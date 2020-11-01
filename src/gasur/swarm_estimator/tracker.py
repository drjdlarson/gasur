"""Implements RFS tracking algorithms.

This module contains the classes and data structures
for RFS tracking related algorithms.
"""
import numpy as np
from numpy.linalg import cholesky, inv
import numpy.random as rnd
import matplotlib.pyplot as plt
from matplotlib.patches import Ellipse
import abc
from copy import deepcopy

from gncpy.math import log_sum_exp
from gasur.utilities.distributions import GaussianMixture, StudentsTMixture
from gasur.utilities.graphs import k_shortest, murty_m_best
from gasur.utilities.plotting import calc_error_ellipse


class RandomFiniteSetBase(metaclass=abc.ABCMeta):
    """ Generic base class for RFS based filters.

    Attributes:
        filter (gncpy.filters.BayesFilter): Filter handling dynamics
        prob_detection (float): Modeled probability an object is detected
        prob_survive (float): Modeled probability of object survival
        birth_terms (list): List of terms in the birth model
    """

    def __init__(self, **kwargs):
        self.filter = None
        self.prob_detection = 1
        self.prob_survive = 1
        self.birth_terms = []
        self.clutter_rate = 0
        self.clutter_den = 0
        super().__init__(**kwargs)

    @property
    def prob_miss_detection(self):
        """ Compliment of :py:attr:`gasur.swarm_estimator.RandomFiniteSetBase.prob_detection`
        """
        return 1 - self.prob_detection

    @property
    def prob_death(self):
        """ Compliment of :py:attr:`gasur.swarm_estimator.RandomFinitSetBase.prob_survive`
        """
        return 1 - self.prob_survive

    @property
    def num_birth_terms(self):
        """ Number of terms in the birth model
        """
        return len(self.birth_terms)

    @abc.abstractmethod
    def predict(self, **kwargs):
        pass

    @abc.abstractmethod
    def correct(self, **kwargs):
        pass

    @abc.abstractmethod
    def extract_states(self, **kwargs):
        pass


class ProbabilityHypothesisDensity(RandomFiniteSetBase):
    """ Probability Hypothesis Density Filter


    """

    def __init__(self, **kwargs):
        self.gating_on = False
        self.inv_chi2_gate = 0
        self.extract_threshold = 0.5
        self.prune_threshold = 1*10**(-15)
        self.save_covs = False
        self.max_hyps = 3000

        self._gaussMix = GaussianMixture()
        self._states = []  # local copy for internal modification
        self._meas_tab = []  # list of lists, one per timestep, inner is all meas at time
        self._covs = []  # local copy for internal modification

        super().__init__(**kwargs)

    @property
    def states(self):
        """ Read only list of extracted states.

        This is a list with 1 element per timestep, and each element is a list
        of the best states extracted at that timestep. The order of each
        element corresponds to the label order.
        """
        if len(self._states) > 0:
            return self._states[-1]
        else:
            return []

    @property
    def covariances(self):
        """ Read only list of extracted covariances.

        This is a list with 1 element per timestep, and each element is a list
        of the best covariances extracted at that timestep. The order of each
        element corresponds to the state order.

        Raises:
            RuntimeWarning: If the class is not saving the covariances, and
                returns an empty list
        """
        if not self.save_covs:
            raise RuntimeWarning("Not saving covariances")
            return []
        if len(self._covs) > 0:
            return self._covs
        else:
            return []

    def predict(self, **kwargs):
        """ Prediction step of the PHD filter.

        This predicts new hypothesis, and propogates them to the next time
        step. It also updates the cardinality distribution. Because this calls
        the inner filter's predict function, the keyword arguments must contain
        any information needed by that function.

        Keyword Args:

        """
        self._gaussMix = self.predict_prob_density(probDensity=self._gaussMix,
                                           **kwargs)
                
        for (gm, _) in self.birth_terms:
            self._gaussMix.weights.extend(gm.weights)
            self._gaussMix.means.extend(gm.means)
            self._gaussMix.covariances.extend(gm.covariances)

    def predict_prob_density(self, **kwargs):
        """ Loops over all elements in a probability distribution and preforms
        the filter prediction.

        Keyword Args:
            probDensity (:py:class:`gasur.utilities.distributions.GaussianMixture`): A
                probability density to run prediction on

        Returns:
            gm (:py:class:`gasur.utilities.distributions.GaussianMixture`): The
                predicted probability density
        """
        probDensity = kwargs['probDensity']
        gm_tup = zip(probDensity.means,
                     probDensity.covariances)
        c_in = np.zeros((self.filter.get_input_mat().shape[1], 1))
        gm = GaussianMixture()
        gm.weights = [self.prob_survive*x for x in probDensity.weights.copy()]
        for ii, (m, P) in enumerate(gm_tup):
            self.filter.cov = P
            n_mean = self.filter.predict(cur_state=m, cur_input=c_in,
                                         **kwargs)
            gm.covariances.append(self.filter.cov.copy())
            gm.means.append(n_mean)

        return gm

    def correct(self, **kwargs):
        """ Correction step of the PHD filter.

        This corrects the hypotheses based on the measurements and gates the
        measurements according to the class settings. It also updates the
        cardinality distribution. Because this calls the inner filter's correct
        function, the keyword arguments must contain any information needed by
        that function.

        Keyword Args:
            meas (list): List of Nm x 1 numpy arrays that contain all the
                measurements needed for this correction
        """
        meas = deepcopy(kwargs['meas'])
        del kwargs['meas']
        
        gmix = deepcopy(self._gaussMix)
        gmix.weights = [self.prob_miss_detection*x for x in gmix.weights]
        for (_, z) in enumerate(meas):
            (gm, cost) = self.correct_prob_density(meas=z, 
                                               probDensity=self._gaussMix,
                                               **kwargs)
        
        gm.weights.extend(gmix.weights)
        self._gaussMix.weights = gm.weights.copy()
        gm.means.extend(gmix.means)
        self._gaussMix.means = gm.means.copy()
        gm.covariances.extend(gmix.covariances)
        self._gaussMix.covariances = gm.covariances.copy()
        
    def correct_prob_density(self, meas, **kwargs):
        """ Loops over all elements in a probability distribution and preforms
        the filter correction.

        Keyword Args:
            probDensity (:py:class:`gasur.utilities.distributions.GaussianMixture`): A
                probability density to run correction on
            meas (list): List of measurements, each is a N x 1 numpy array

        Returns:
            tuple containing

                - gm (:py:class:`gasur.utilities.distributions.GaussianMixture`): The
                  corrected probability density
                - cost (float): Total cost of for the m best assignment
        """
        probDensity = kwargs['probDensity']

        gm = GaussianMixture()
        
        
        for jj in range(0, len(probDensity.means)):
            self.filter.cov = probDensity.covariances[jj]
            state = probDensity.means[jj]
            (mean, qz) = self.filter.correct(meas=meas, cur_state=state,
                                             **kwargs)
            cov = self.filter.cov
            w = qz * probDensity.weights[jj]
            gm.means.append(mean)
            gm.covariances.append(cov)
            gm.weights.append(w)
        lst = gm.weights
        lst = [x + np.finfo(float).eps for x in lst]
        gm.weights = lst
        cost = sum(gm.weights)
        for jj in range(0, len(gm.weights)):
            gm.weights[jj] /= cost
        return (gm, cost)

    def prune(self, **kwargs):
        """ Removes hypotheses below a threshold.

        This should be called once per time step after the correction and
        before the state extraction.
        """
        idx = np.where(np.asarray(self._gaussMix.weights)<self.prune_threshold)
        idx = np.ndarray.flatten(idx[0])
        if len(idx)!=0:
            for index in sorted(idx, reverse=True):
                del self._gaussMix.means[index]
                del self._gaussMix.weights[index]
                del self._gaussMix.covariances[index]
        
        
        
    def cap(self, **kwargs):
        """ Removes least likely hypotheses until a maximum number is reached.

        This should be called once per time step after pruning and
        before the state extraction.
        """
        if len(self._gaussMix.weights) > self.max_hyps:
            self._gaussMix.weights.sort(reverse=True)
            del self._gaussMix.weights[self.max_hyps:-1]
            self._gaussMix.means.sort(reverse=True)
            del self._gaussMix.means[self.max_hyps:-1]
            self._gaussMix.covariances.sort(reverse=True)
            del self._gaussMix.covariances[self.max_hyps:-1]
        
    def extract_states(self, **kwargs):
        """ Extracts the best state estimates.

        This extracts the best states from the distribution. It should be
        called once per time step after the correction function. This calls
        both the inner filters predict and correct functions so the keyword
        arguments must contain any additional variables needed by those
        functions.
        """
        idx = np.where(np.asarray(self._gaussMix.weights)>=self.extract_threshold)
        idx = np.ndarray.flatten(idx[0])
        if len(idx)!=0:
            for jj in range(0, len(idx)):
                self._states.append(self._gaussMix.means[idx[jj]])
                self._covs.append(self._gaussMix.covariances[idx[jj]])

    def _gate_meas(self, meas, means, covs, **kwargs):
        if len(meas) == 0:
            return []

        valid = []
        for (m, p) in zip(means, covs):
            meas_mat = self.filter.get_meas_mat(m, **kwargs)
            est = self.filter.get_est_meas(m, **kwargs)
            meas_pred_cov = meas_mat @ p @ meas_mat.T + self.filter.meas_noise
            meas_pred_cov = (meas_pred_cov + meas_pred_cov.T) / 2
            v_s = cholesky(meas_pred_cov.T)
            inv_sqrt_m_cov = inv(v_s)

            for (ii, z) in enumerate(meas):
                if ii in valid:
                    continue
                inov = z - est
                dist = np.sum((inv_sqrt_m_cov.T @ inov)**2)
                if dist < self.inv_chi2_gate:
                    valid.append(ii)

        valid.sort()
        return [meas[ii] for ii in valid]

    def plot_states(self, plt_inds, **kwargs):
        """ Plots the best estimate for the states.

        This assumes that the states have been extracted. It's designed to plot
        two of the state variables (typically x/y position). The error ellipses
        are calculated according to :cite:`Hoover1984_AlgorithmsforConfidenceCirclesandEllipses`

        Args:
            plt_inds (list): List of indices in the state vector to plot

        Keyword Args:
            f_hndl (Matplotlib figure): Current to figure to plot on. Always
                plots on axes[0], pass None to create a new figure
            true_states (list): list where each element is a list of numpy
                N x 1 arrays of each true state. If not given true states
                are not plotted.
            sig_bnd (int): If set and the covariances are saved, the sigma
                bounds are scaled by this number and plotted for each track
            rng (Generator): A numpy random generator, leave as None for
                default.
            meas_inds (list): List of indices in the measurement vector to plot
                if this is specified all available measurements will be
                plotted. Note, x-axis is first, then y-axis. Also note, if
                gating is on then gated measurements will not be plotted.
            lgnd_loc (string): Location of the legend. Set to none to skip
                creating a legend.

        Returns:
            (Matplotlib figure): Instance of the matplotlib figure used
        """

        f_hndl = kwargs.get('f_hndl', None)
        true_states = kwargs.get('true_states', None)
        sig_bnd = kwargs.get('sig_bnd', None)
        rng = kwargs.get('rng', None)
        meas_inds = kwargs.get('meas_inds', None)
        lgnd_loc = kwargs.get('lgnd_loc', None)

        if rng is None:
            rng = rnd.default_rng(1)

        plt_meas = meas_inds is not None
        show_sig = sig_bnd is not None and self.save_covs

        s_lst = deepcopy(self._states)
        x_dim = None

        if f_hndl is None:
            f_hndl = plt.figure()
            f_hndl.add_subplot(1, 1, 1)

        # get state dimension
        for states in s_lst:
            if len(states) > 0:
                x_dim = states[0].size
                break

        # get array of all state values for each label
        added_sig_lbl = False
        added_true_lbl = False
        added_state_lbl = False
        added_meas_lbl = False
        r = rng.random()
        b = rng.random()
        g = rng.random()
        color = (r, g, b)
        for tt, states in enumerate(s_lst):
            if len(states) == 0:
                continue

            x = np.array(states)
            if show_sig:
                sigs = [None] * len(states)
                for ii, cov in enumerate(self._covs[tt]):
                    sig = np.zeros((2, 2))
                    sig[0, 0] = cov[ii][plt_inds[0], plt_inds[0]]
                    sig[0, 1] = cov[ii][plt_inds[0], plt_inds[1]]
                    sig[1, 0] = cov[ii][plt_inds[1], plt_inds[0]]
                    sig[1, 1] = cov[ii][plt_inds[1], plt_inds[1]]
                    sigs[ii] = sig

            # plot
            if show_sig:
                for ii, sig in enumerate(sigs):
                    if sig is None:
                        continue
                    w, h, a = calc_error_ellipse(sig, sig_bnd)
                    if not added_sig_lbl:
                        s = r'${}\sigma$ Error Ellipses'.format(sig_bnd)
                        e = Ellipse(xy=x[plt_inds, ii], width=w,
                                    height=h, angle=a, zorder=-10000,
                                    label=s)
                        added_sig_lbl = True
                    else:
                        e = Ellipse(xy=x[plt_inds, ii], width=w,
                                    height=h, angle=a, zorder=-10000)
                    e.set_clip_box(f_hndl.axes[0].bbox)
                    e.set_alpha(0.2)
                    e.set_facecolor(color)
                    f_hndl.axes[0].add_patch(e)

            if not added_state_lbl:
                f_hndl.axes[0].scatter(x[plt_inds[0], :], x[plt_inds[1], :],
                                       color=color, edgecolors=(0, 0, 0),
                                       label='States')
                added_state_lbl = True
            else:
                f_hndl.axes[0].scatter(x[plt_inds[0], :], x[plt_inds[1], :],
                                       color=color, edgecolors=(0, 0, 0))

        # if true states are available then plot them
        if true_states is not None:
            if x_dim is None:
                for states in true_states:
                    if len(states) > 0:
                        x_dim = states[0].size
                        break

            max_true = max([len(x) for x in true_states])
            x = np.nan * np.ones((x_dim, len(true_states), max_true))
            for tt, states in enumerate(true_states):
                for ii, state in enumerate(states):
                    x[:, [tt], ii] = state.copy()

            for ii in range(0, max_true):
                if not added_true_lbl:
                    f_hndl.axes[0].plot(x[plt_inds[0], :, ii],
                                        x[plt_inds[1], :, ii],
                                        color='k', marker='.',
                                        label='True Trajectories')
                    added_true_lbl = True
                else:
                    f_hndl.axes[0].plot(x[plt_inds[0], :, ii],
                                        x[plt_inds[1], :, ii],
                                        color='k', marker='.')

        if plt_meas:
            meas_x = []
            meas_y = []
            for meas_tt in self._meas_tab:
                mx_ii = [m[meas_inds[0]].item() for m in meas_tt]
                my_ii = [m[meas_inds[1]].item() for m in meas_tt]
                meas_x.extend(mx_ii)
                meas_y.extend(my_ii)
            color = (128/255, 128/255, 128/255)
            meas_x = np.asarray(meas_x)
            meas_y = np.asarray(meas_y)
            if not added_meas_lbl:
                f_hndl.axes[0].scatter(meas_x, meas_y, zorder=-1, alpha=0.35,
                                       color=color, marker='^',
                                       label='Measurements')
            else:
                f_hndl.axes[0].scatter(meas_x, meas_y, zorder=-1, alpha=0.35,
                                       color=color, marker='^')

        f_hndl.axes[0].grid(True)
        f_hndl.axes[0].set_title("Labeled State Trajectories")
        f_hndl.axes[0].set_ylabel("y-position")
        f_hndl.axes[0].set_xlabel("x-position")
        if lgnd_loc is not None:
            plt.legend(loc=lgnd_loc)
        plt.tight_layout()

        return f_hndl


class GeneralizedLabeledMultiBernoulli(RandomFiniteSetBase):
    """ Delta-Generalized Labeled Multi-Bernoulli filter.

    This is based on :cite:`Vo2013_LabeledRandomFiniteSetsandMultiObjectConjugatePriors`
    and :cite:`Vo2014_LabeledRandomFiniteSetsandtheBayesMultiTargetTrackingFilter`
    It does not account for agents spawned from existing tracks, only agents
    birthed from the given birth model.

    Attributes:
        req_births (int): Number of requested birth hypotheses
        req_surv (int): Number of requested surviving hypotheses
        req_upd (int): Number of requested updated hypotheses
        gating_on (bool): Determines if measurements are gated
        inv_chi2_gate (float): Chi squared threshold for gating the
            measurements
        birth_terms (list): List of tuples where the first element is
            a :py:class:`gasur.utilities.distributions.GaussianMixture` and
            the second is the birth probability for that term
        prune_threshold (float): Minimum association probability to keep when
            pruning
        max_hyps (int): Maximum number of hypotheses to keep when capping
        save_covs (bool): Save covariance matrix for each state during state
            extraction
    """

    class _TabEntry:
        def __init__(self):
            self.label = ()  # time step born, index of birth model born from
            self.probDensity = None  # must be a distribution class
            self.meas_assoc_hist = []  # list indices into measurement list per time step

    class _HypothesisHelper:
        def __init__(self):
            self.assoc_prob = 0
            self.track_set = []  # indices in lookup table

        @property
        def num_tracks(self):
            return len(self.track_set)

    def __init__(self, **kwargs):
        self.req_births = 0
        self.req_surv = 0
        self.req_upd = 0
        self.gating_on = False
        self.inv_chi2_gate = 0
        self.prune_threshold = 1*10**(-15)
        self.max_hyps = 3000
        self.save_covs = False

        self._track_tab = []  # list of all possible tracks
        self._states = []  # local copy for internal modification
        self._labels = []  # local copy for internal modification
        self._meas_tab = []  # list of lists, one per timestep, inner is all meas at time
        self._meas_asoc_mem = []
        self._lab_mem = []
        self._covs = []  # local copy for internal modification

        hyp0 = self._HypothesisHelper()
        hyp0.assoc_prob = 1
        hyp0.track_set = []
        self._hypotheses = [hyp0]  # list of _HypothesisHelper objects

        self._card_dist = []  # probability of having index # as cardinality

        super().__init__(**kwargs)

    @property
    def states(self):
        """ Read only list of extracted states.

        This is a list with 1 element per timestep, and each element is a list
        of the best states extracted at that timestep. The order of each
        element corresponds to the label order.
        """
        return self._states

    @property
    def labels(self):
        """ Read only list of extracted labels.

        This is a list with 1 element per timestep, and each element is a list
        of the best labels extracted at that timestep. The order of each
        element corresponds to the state order.
        """
        return self._labels

    @property
    def covariances(self):
        """ Read only list of extracted covariances.

        This is a list with 1 element per timestep, and each element is a list
        of the best covariances extracted at that timestep. The order of each
        element corresponds to the state order.

        Raises:
            RuntimeWarning: If the class is not saving the covariances, and
                returns an empty list
        """
        if not self.save_covs:
            raise RuntimeWarning("Not saving covariances")
            return []
        return self._covs

    def predict(self, **kwargs):
        """ Prediction step of the GLMB filter.

        This predicts new hypothesis, and propogates them to the next time
        step. It also updates the cardinality distribution. Because this calls
        the inner filter's predict function, the keyword arguments must contain
        any information needed by that function.

        Keyword Args:
            time_step (int): Current time step number for the new labels
        """

        # Find cost for each birth track, and setup lookup table
        time_step = kwargs['time_step']

        log_cost = []
        birth_tab = []
        for ii, (gm, p) in enumerate(self.birth_terms):
            cost = p / (1 - p)
            log_cost.append(-np.log(cost))
            entry = self._TabEntry()
            entry.probDensity = deepcopy(gm)
            entry.label = (time_step, ii)
            birth_tab.append(entry)

        # get K best hypothesis, and their index in the lookup table
        (paths, hyp_cost) = k_shortest(np.array(log_cost), self.req_births)

        # calculate association probabilities for birth hypothesis
        tot_cost = 0
        for c in hyp_cost:
            tot_cost = tot_cost + np.exp(-c).item()
        birth_hyps = []
        for (p, c) in zip(paths, hyp_cost):
            hyp = self._HypothesisHelper()
            # NOTE: this may suffer from underflow and can be improved
            hyp.assoc_prob = np.exp(-c).item() / tot_cost
            hyp.track_set = p
            birth_hyps.append(hyp)

        # Init and propagate surviving track table
        surv_tab = []
        for (ii, track) in enumerate(self._track_tab):
            gm = self.predict_prob_density(probDensity=track.probDensity,
                                           **kwargs)

            entry = self._TabEntry()
            entry.probDensity = gm
            entry.meas_assoc_hist = deepcopy(track.meas_assoc_hist)
            entry.label = track.label
            surv_tab.append(entry)

        # loop over postierior components
        surv_hyps = []
        sum_sqrt_w = 0
        for hyp in self._hypotheses:
            sum_sqrt_w = sum_sqrt_w + np.sqrt(hyp.assoc_prob)
        for hyp in self._hypotheses:
            if hyp.num_tracks == 0:
                new_hyp = self._HypothesisHelper()
                new_hyp.assoc_prob = np.log(hyp.assoc_prob)
                new_hyp.track_set = hyp.track_set
                surv_hyps.append(new_hyp)
            else:
                cost = self.prob_survive / self.prob_death
                log_cost = [-np.log(cost)] * hyp.num_tracks
                k = np.round(self.req_surv * np.sqrt(hyp.assoc_prob)
                             / sum_sqrt_w)
                (paths, hyp_cost) = k_shortest(np.array(log_cost), k)

                for (p, c) in zip(paths, hyp_cost):
                    new_hyp = self._HypothesisHelper()
                    new_hyp.assoc_prob = hyp.num_tracks \
                        * np.log(self.prob_death) + np.log(hyp.assoc_prob) \
                        - c.item()
                    if len(p) > 0:
                        new_hyp.track_set = [hyp.track_set[ii] for ii in p]
                    else:
                        new_hyp.track_set = []
                    surv_hyps.append(new_hyp)

        lse = log_sum_exp([x.assoc_prob for x in surv_hyps])
        for ii in range(0, len(surv_hyps)):
            surv_hyps[ii].assoc_prob = np.exp(surv_hyps[ii].assoc_prob - lse)

        # Get  predicted hypothesis by convolution
        self._track_tab = birth_tab + surv_tab
        self._hypotheses = []
        tot_w = 0
        for b_hyp in birth_hyps:
            for s_hyp in surv_hyps:
                new_hyp = self._HypothesisHelper()
                new_hyp.assoc_prob = b_hyp.assoc_prob * s_hyp.assoc_prob
                tot_w = tot_w + new_hyp.assoc_prob
                surv_lst = []
                for x in s_hyp.track_set:
                    surv_lst.append(x + len(birth_tab))
                new_hyp.track_set = b_hyp.track_set + surv_lst
                self._hypotheses.append(new_hyp)

        for ii in range(0, len(self._hypotheses)):
            self._hypotheses[ii].assoc_prob = (self._hypotheses[ii].assoc_prob
                                               / tot_w)
        self._card_dist = self.calc_card_dist(self._hypotheses)
        self._clean_predictions()

    def predict_prob_density(self, **kwargs):
        """ Loops over all elements in a probability distribution and preforms
        the filter prediction.

        Keyword Args:
            probDensity (:py:class:`gasur.utilities.distributions.GaussianMixture`): A
                probability density to run prediction on

        Returns:
            gm (:py:class:`gasur.utilities.distributions.GaussianMixture`): The
                predicted probability density
        """
        probDensity = kwargs['probDensity']
        gm_tup = zip(probDensity.means,
                     probDensity.covariances)
        c_in = np.zeros((self.filter.get_input_mat().shape[1], 1))
        gm = GaussianMixture()
        gm.weights = probDensity.weights.copy()
        for ii, (m, P) in enumerate(gm_tup):
            self.filter.cov = P
            n_mean = self.filter.predict(cur_state=m, cur_input=c_in,
                                         **kwargs)
            gm.covariances.append(self.filter.cov.copy())
            gm.means.append(n_mean)

        return gm

    def correct(self, **kwargs):
        """ Correction step of the GLMB filter.

        This corrects the hypotheses based on the measurements and gates the
        measurements according to the class settings. It also updates the
        cardinality distribution. Because this calls the inner filter's correct
        function, the keyword arguments must contain any information needed by
        that function.

        Keyword Args:
            meas (list): List of Nm x 1 numpy arrays that contain all the
                measurements needed for this correction
        """

        meas = deepcopy(kwargs['meas'])
        del kwargs['meas']

        # gate measurements by tracks
        if self.gating_on:
            means = []
            covs = []
            for ent in self._track_tab:
                means.extend(ent.probDensity.means)
                covs.extend(ent.probDensity.covariances)
            meas = self._gate_meas(meas, means, covs, **kwargs)

        self._meas_tab.append(meas)
        num_meas = len(meas)

        # missed detection tracks
        num_pred = len(self._track_tab)
        up_tab = []
        for ii in range(0, (num_meas + 1) * num_pred):
            up_tab.append(self._TabEntry())

        for ii, track in enumerate(self._track_tab):
            up_tab[ii] = deepcopy(track)
            up_tab[ii].meas_assoc_hist.append(None)

        # measurement updated tracks
        all_cost_m = np.zeros((num_pred, num_meas))
        for emm, z in enumerate(meas):
            for ii, ent in enumerate(self._track_tab):
                s_to_ii = num_pred * emm + ii + num_pred
                (up_tab[s_to_ii].probDensity, cost) = \
                    self.correct_prob_density(meas=z,
                                              probDensity=ent.probDensity,
                                              **kwargs)

                # update association history with current measurement index
                up_tab[s_to_ii].meas_assoc_hist = ent.meas_assoc_hist + [emm]
                up_tab[s_to_ii].label = ent.label
                all_cost_m[ii, emm] = cost

        # component updates
        up_hyp = []
        if num_meas == 0:
            for hyp in self._hypotheses:
                hyp.assoc_prob = -self.clutter_rate + hyp.num_tracks \
                    * np.log(self.prob_miss_detection) + np.log(hyp.assoc_prob)
                up_hyp.append(hyp)
        else:
            clutter = self.clutter_rate * self.clutter_den
            if self.prob_miss_detection == 0:
                p_d_ratio = np.inf
            else:
                p_d_ratio = self.prob_detection / self.prob_miss_detection
            ss_w = 0
            for p_hyp in self._hypotheses:
                ss_w += np.sqrt(p_hyp.assoc_prob)
            for p_hyp in self._hypotheses:
                if p_hyp.num_tracks == 0:  # all clutter
                    new_hyp = self._HypothesisHelper()
                    new_hyp.assoc_prob = -self.clutter_rate + num_meas \
                        * np.log(clutter) + np.log(p_hyp.assoc_prob)
                    new_hyp.track_set = p_hyp.track_set
                    up_hyp.append(new_hyp)

                else:
                    if clutter == 0:
                        cost_m = np.inf * all_cost_m[p_hyp.track_set, :]
                    else:
                        cost_m = p_d_ratio * all_cost_m[p_hyp.track_set, :] \
                            / clutter
                    neg_log = -np.log(cost_m)
                    m = np.round(self.req_upd * np.sqrt(p_hyp.assoc_prob)
                                 / ss_w)
                    m = int(m.item())
                    [assigns, costs] = murty_m_best(neg_log, m)

                    for (a, c) in zip(assigns, costs):
                        new_hyp = self._HypothesisHelper()
                        new_hyp.assoc_prob = -self.clutter_rate + num_meas \
                            * np.log(clutter) + p_hyp.num_tracks \
                            * np.log(self.prob_miss_detection) \
                            + np.log(p_hyp.assoc_prob) - c
                        lst1 = [num_pred * x for x in a]
                        lst2 = p_hyp.track_set.copy()
                        new_hyp.track_set = [sum(x) for x in zip(lst1, lst2)]
                        up_hyp.append(new_hyp)

        lse = log_sum_exp([x.assoc_prob for x in up_hyp])
        for ii in range(0, len(up_hyp)):
            up_hyp[ii].assoc_prob = np.exp(up_hyp[ii].assoc_prob - lse)

        self._track_tab = up_tab
        self._hypotheses = up_hyp
        self._card_dist = self.calc_card_dist(self._hypotheses)
        self._clean_updates()

    def correct_prob_density(self, meas, **kwargs):
        """ Loops over all elements in a probability distribution and preforms
        the filter correction.

        Keyword Args:
            probDensity (:py:class:`gasur.utilities.distributions.GaussianMixture`): A
                probability density to run correction on
            meas (list): List of measurements, each is a N x 1 numpy array

        Returns:
            tuple containing

                - gm (:py:class:`gasur.utilities.distributions.GaussianMixture`): The
                  corrected probability density
                - cost (float): Total cost of for the m best assignment
        """
        probDensity = kwargs['probDensity']

        gm = GaussianMixture()
        for jj in range(0, len(probDensity.means)):
            self.filter.cov = probDensity.covariances[jj]
            state = probDensity.means[jj]
            (mean, qz) = self.filter.correct(meas=meas, cur_state=state,
                                             **kwargs)
            cov = self.filter.cov
            w = qz * probDensity.weights[jj]
            gm.means.append(mean)
            gm.covariances.append(cov)
            gm.weights.append(w)
        lst = gm.weights
        lst = [x + np.finfo(float).eps for x in lst]
        gm.weights = lst
        cost = sum(gm.weights)
        for jj in range(0, len(gm.weights)):
            gm.weights[jj] /= cost
        return (gm, cost)

    def extract_most_prob_states(self, thresh, **kwargs):
        loc_self = deepcopy(self)
        state_sets = []
        cov_sets = []
        label_sets = []
        probs = []

        idx = loc_self.extract_states(**kwargs)
        if idx is None:
            return (state_sets, label_sets, cov_sets, probs)

        state_sets.append(loc_self.states.copy())
        label_sets.append(loc_self.labels.copy())
        if loc_self.save_covs:
            cov_sets.append(loc_self.covariances.copy())
        probs.append(loc_self._hypotheses[idx].assoc_prob)
        loc_self._hypotheses[idx].assoc_prob = 0
        while True:
            idx = loc_self.extract_states(**kwargs)
            if idx is None:
                break

            if loc_self._hypotheses[idx].assoc_prob >= thresh:
                state_sets.append(loc_self.states.copy())
                label_sets.append(loc_self.labels.copy())
                if loc_self.save_covs:
                    cov_sets.append(loc_self.covariances.copy())
                probs.append(loc_self._hypotheses[idx].assoc_prob)
                loc_self._hypotheses[idx].assoc_prob = 0
            else:
                break

        return (state_sets, label_sets, cov_sets, probs)

    def extract_states(self, **kwargs):
        """ Extracts the best state estimates.

        This extracts the best states from the distribution. It should be
        called once per time step after the correction function. This calls
        both the inner filters predict and correct functions so the keyword
        arguments must contain any additional variables needed by those
        functions.

        Returns:
            idx_cmp (int): Index of the hypothesis table used when extracting
                states
        """

        card = np.argmax(self._card_dist)
        tracks_per_hyp = np.array([x.num_tracks for x in self._hypotheses])
        weight_per_hyp = np.array([x.assoc_prob for x in self._hypotheses])

        if len(tracks_per_hyp) == 0:
            self._states = [[]]
            self._labels = [[]]
            self._covs = [[]]
            return None

        idx_cmp = np.argmax(weight_per_hyp * (tracks_per_hyp == card))
        meas_hists = []
        labels = []
        for ptr in self._hypotheses[idx_cmp].track_set:
            meas_hists.append(self._track_tab[ptr].meas_assoc_hist.copy())
            labels.append(self._track_tab[ptr].label)

        both = set(self._lab_mem).intersection(labels)
        surv_ii = [labels.index(x) for x in both]
        either = set(self._lab_mem).symmetric_difference(labels)
        dead_ii = [self._lab_mem.index(a) for a in either
                   if a in self._lab_mem]
        new_ii = [labels.index(a) for a in either if a in labels]

        self._lab_mem = [self._lab_mem[ii] for ii in dead_ii] \
            + [labels[ii] for ii in surv_ii] \
            + [labels[ii] for ii in new_ii]
        self._meas_asoc_mem = [self._meas_asoc_mem[ii] for ii in dead_ii] \
            + [meas_hists[ii] for ii in surv_ii] \
            + [meas_hists[ii] for ii in new_ii]

        self._states = [None] * len(self._meas_tab)
        self._labels = [None] * len(self._meas_tab)
        if self.save_covs:
            self._covs = [None] * len(self._meas_tab)

        # if there are no old or new tracks assume its the first iteration
        if len(self._lab_mem) == 0 and len(self._meas_asoc_mem) == 0:
            self._states = [[]]
            self._labels = [[]]
            self._covs = [[]]
            return None

        for (hist, (b_time, b_idx)) in zip(self._meas_asoc_mem, self._lab_mem):
            pd = deepcopy(self.birth_terms[b_idx][0])

            for (t_after_b, emm) in enumerate(hist):
                # propagate for GM
                pd = self.predict_prob_density(probDensity=pd, **kwargs)

                # measurement correction for GM
                tt = b_time + t_after_b
                if emm is not None:
                    meas = self._meas_tab[tt][emm].copy()
                    (pd, _) = self.correct_prob_density(meas=meas,
                                                        probDensity=pd,
                                                        **kwargs)

                # find best one and add to state table
                idx_trk = np.argmax(pd.weights)
                new_state = pd.means[idx_trk]
                new_cov = pd.covariances[idx_trk]
                new_label = (b_time, b_idx)
                if self._labels[tt] is None:
                    self._states[tt] = [new_state]
                    self._labels[tt] = [new_label]
                    if self.save_covs:
                        self._covs[tt] = [new_cov]
                else:
                    self._states[tt].append(new_state)
                    self._labels[tt].append(new_label)
                    if self.save_covs:
                        self._covs[tt].append(new_cov)
        return idx_cmp

    def prune(self, **kwargs):
        """ Removes hypotheses below a threshold.

        This should be called once per time step after the correction and
        before the state extraction.
        """

        # Find hypotheses with low association probabilities
        temp_assoc_probs = np.array([])
        for ii in range(0, len(self._hypotheses)):
            temp_assoc_probs = np.append(temp_assoc_probs,
                                         self._hypotheses[ii].assoc_prob)
        keep_indices = np.argwhere(temp_assoc_probs > self.prune_threshold).T
        keep_indices = keep_indices.flatten()

        # For re-weighing association probabilities
        new_sum = np.sum(temp_assoc_probs[keep_indices])
        self._hypotheses = [self._hypotheses[ii] for ii in keep_indices]
        for ii in range(0, len(keep_indices)):
            self._hypotheses[ii].assoc_prob = (self._hypotheses[ii].assoc_prob
                                               / new_sum)
        # Re-calculate cardinality
        self._card_dist = self.calc_card_dist(self._hypotheses)

    def cap(self, **kwargs):
        """ Removes least likely hypotheses until a maximum number is reached.

        This should be called once per time step after pruning and
        before the state extraction.
        """

        # Determine if there are too many hypotheses
        if len(self._hypotheses) > self.max_hyps:
            temp_assoc_probs = np.array([])
            for ii in range(0, len(self._hypotheses)):
                temp_assoc_probs = np.append(temp_assoc_probs,
                                             self._hypotheses[ii].assoc_prob)
            sorted_indices = np.argsort(temp_assoc_probs)

            # Reverse order to get descending array
            sorted_indices = sorted_indices[::-1]

            # Take the top n assoc_probs, where n = max_hyps
            keep_indices = np.array([], dtype=np.int64)
            for ii in range(0, self.max_hyps):
                keep_indices = np.append(keep_indices, int(sorted_indices[ii]))

            # Assign to class
            self._hypotheses = [self._hypotheses[ii] for ii in keep_indices]

            # Normalize association probabilities
            new_sum = 0
            for ii in range(0, len(self._hypotheses)):
                new_sum = new_sum + self._hypotheses[ii].assoc_prob

            for ii in range(0, len(self._hypotheses)):
                self._hypotheses[ii].assoc_prob = (self._hypotheses[ii].assoc_prob
                                                   / new_sum)

            # Re-calculate cardinality
            self._card_dist = self.calc_card_dist(self._hypotheses)

    def calc_card_dist(self, hyp_lst):
        """ Calucaltes the cardinality distribution.

        Args:
            hyp_lst (list): list of hypotheses to use when finding the
                distribution

        Returns:
            (list): Each element is the probability that the index is the
            cardinality.
        """

        if len(hyp_lst) == 0:
            return 0

        card_dist = []
        for ii in range(0, max(map(lambda x: x.num_tracks, hyp_lst)) + 1):
            card = 0
            for hyp in hyp_lst:
                if hyp.num_tracks == ii:
                    card = card + hyp.assoc_prob
            card_dist.append(card)
        return card_dist

    def _clean_predictions(self):
        hash_lst = []
        for hyp in self._hypotheses:
            if len(hyp.track_set) == 0:
                lst = []
            else:
                hyp.track_set.sort()
                lst = [int(x) for x in hyp.track_set]
            h = hash('*'.join(map(str, lst)))
            hash_lst.append(h)

        new_hyps = []
        used_hash = []
        for ii, h in enumerate(hash_lst):
            if h not in used_hash:
                used_hash.append(h)
                new_hyps.append(self._hypotheses[ii])
            else:
                new_ii = used_hash.index(h)
                new_hyps[new_ii].assoc_prob += self._hypotheses[ii].assoc_prob
        self._hypotheses = new_hyps

    def _clean_updates(self):
        used = [0] * len(self._track_tab)
        for hyp in self._hypotheses:
            for ii in hyp.track_set:
                used[ii] += 1
        nnz_inds = [idx for idx, val in enumerate(used) if val != 0]
        track_cnt = len(nnz_inds)

        new_inds = [0] * len(self._track_tab)
        for (ii, v) in zip(nnz_inds, [ii for ii in range(0, track_cnt)]):
            new_inds[ii] = v

        new_tab = [deepcopy(self._track_tab[ii]) for ii in nnz_inds]
        new_hyps = []
        for(ii, hyp) in enumerate(self._hypotheses):
            if len(hyp.track_set) > 0:
                hyp.track_set = [new_inds[ii] for ii in hyp.track_set]
            new_hyps.append(hyp)

        self._track_tab = new_tab
        self._hypotheses = new_hyps

    def _gate_meas(self, meas, means, covs, **kwargs):
        if len(meas) == 0:
            return []

        valid = []
        for (m, p) in zip(means, covs):
            meas_mat = self.filter.get_meas_mat(m, **kwargs)
            est = self.filter.get_est_meas(m, **kwargs)
            meas_pred_cov = meas_mat @ p @ meas_mat.T + self.filter.meas_noise
            meas_pred_cov = (meas_pred_cov + meas_pred_cov.T) / 2
            v_s = cholesky(meas_pred_cov.T)
            inv_sqrt_m_cov = inv(v_s)

            for (ii, z) in enumerate(meas):
                if ii in valid:
                    continue
                inov = z - est
                dist = np.sum((inv_sqrt_m_cov.T @ inov)**2)
                if dist < self.inv_chi2_gate:
                    valid.append(ii)

        valid.sort()
        return [meas[ii] for ii in valid]

    def plot_states_labels(self, plt_inds, **kwargs):
        """ Plots the best estimate for the states and labels.

        This assumes that the states have been extracted. It's designed to plot
        two of the state variables (typically x/y position). The error ellipses
        are calculated according to :cite:`Hoover1984_AlgorithmsforConfidenceCirclesandEllipses`

        Args:
            plt_inds (list): List of indices in the state vector to plot

        Keyword Args:
            f_hndl (Matplotlib figure): Current to figure to plot on. Always
                plots on axes[0], pass None to create a new figure
            true_states (list): list where each element is a list of numpy
                N x 1 arrays of each true state. If not given true states
                are not plotted.
            sig_bnd (int): If set and the covariances are saved, the sigma
                bounds are scaled by this number and plotted for each track
            rng (Generator): A numpy random generator, leave as None for
                default.
            meas_inds (list): List of indices in the measurement vector to plot
                if this is specified all available measurements will be
                plotted. Note, x-axis is first, then y-axis. Also note, if
                gating is on then gated measurements will not be plotted.
            lgnd_loc (string): Location of the legend. Set to none to skip
                creating a legend.

        Returns:
            (Matplotlib figure): Instance of the matplotlib figure used
        """

        f_hndl = kwargs.get('f_hndl', None)
        true_states = kwargs.get('true_states', None)
        sig_bnd = kwargs.get('sig_bnd', None)
        rng = kwargs.get('rng', None)
        meas_inds = kwargs.get('meas_inds', None)
        lgnd_loc = kwargs.get('lgnd_loc', None)

        if rng is None:
            rng = rnd.default_rng(1)

        plt_meas = meas_inds is not None
        show_sig = sig_bnd is not None and self.save_covs

        s_lst = deepcopy(self.states)
        l_lst = deepcopy(self.labels)
        x_dim = None

        if f_hndl is None:
            f_hndl = plt.figure()
            f_hndl.add_subplot(1, 1, 1)

        # get state dimension
        for states in s_lst:
            if len(states) > 0:
                x_dim = states[0].size
                break

        # get unique labels
        u_lbls = []
        for lbls in l_lst:
            for lbl in lbls:
                if lbl not in u_lbls:
                    u_lbls.append(lbl)

        # get array of all state values for each label
        added_sig_lbl = False
        added_true_lbl = False
        added_state_lbl = False
        added_meas_lbl = False
        for lbl in u_lbls:
            x = np.nan * np.ones((x_dim, len(s_lst)))
            if show_sig:
                sigs = [None] * len(s_lst)
            for tt, lbls in enumerate(l_lst):
                if lbl in lbls:
                    ii = lbls.index(lbl)
                    x[:, [tt]] = s_lst[tt][ii].copy()

                    if show_sig:
                        sig = np.zeros((2, 2))
                        sig[0, 0] = self._covs[tt][ii][plt_inds[0],
                                                       plt_inds[0]]
                        sig[0, 1] = self._covs[tt][ii][plt_inds[0],
                                                       plt_inds[1]]
                        sig[1, 0] = self._covs[tt][ii][plt_inds[1],
                                                       plt_inds[0]]
                        sig[1, 1] = self._covs[tt][ii][plt_inds[1],
                                                       plt_inds[1]]
                        sigs[tt] = sig

            # plot
            r = rng.random()
            b = rng.random()
            g = rng.random()
            color = (r, g, b)
            if show_sig:
                for tt, sig in enumerate(sigs):
                    if sig is None:
                        continue
                    w, h, a = calc_error_ellipse(sig, sig_bnd)
                    if not added_sig_lbl:
                        s = r'${}\sigma$ Error Ellipses'.format(sig_bnd)
                        e = Ellipse(xy=x[plt_inds, tt], width=w,
                                    height=h, angle=a, zorder=-10000,
                                    label=s)
                        added_sig_lbl = True
                    else:
                        e = Ellipse(xy=x[plt_inds, tt], width=w,
                                    height=h, angle=a, zorder=-10000)
                    e.set_clip_box(f_hndl.axes[0].bbox)
                    e.set_alpha(0.2)
                    e.set_facecolor(color)
                    f_hndl.axes[0].add_patch(e)

            if not added_state_lbl:
                f_hndl.axes[0].scatter(x[plt_inds[0], :], x[plt_inds[1], :],
                                       color=color, edgecolors=(0, 0, 0),
                                       label='States')
                added_state_lbl = True
            else:
                f_hndl.axes[0].scatter(x[plt_inds[0], :], x[plt_inds[1], :],
                                       color=color, edgecolors=(0, 0, 0))

            s = "({}, {})".format(lbl[0], lbl[1])
            tmp = x.copy()
            tmp = tmp[:, ~np.any(np.isnan(tmp), axis=0)]
            f_hndl.axes[0].text(tmp[plt_inds[0], 0], tmp[plt_inds[1], 0], s,
                                color=color)

        # if true states are available then plot them
        if true_states is not None:
            if x_dim is None:
                for states in true_states:
                    if len(states) > 0:
                        x_dim = states[0].size
                        break

            max_true = max([len(x) for x in true_states])
            x = np.nan * np.ones((x_dim, len(true_states), max_true))
            for tt, states in enumerate(true_states):
                for ii, state in enumerate(states):
                    x[:, [tt], ii] = state.copy()

            for ii in range(0, max_true):
                if not added_true_lbl:
                    f_hndl.axes[0].plot(x[plt_inds[0], :, ii],
                                        x[plt_inds[1], :, ii],
                                        color='k', marker='.',
                                        label='True Trajectories')
                    added_true_lbl = True
                else:
                    f_hndl.axes[0].plot(x[plt_inds[0], :, ii],
                                        x[plt_inds[1], :, ii],
                                        color='k', marker='.')

        if plt_meas:
            meas_x = []
            meas_y = []
            for meas_tt in self._meas_tab:
                mx_ii = [m[meas_inds[0]].item() for m in meas_tt]
                my_ii = [m[meas_inds[1]].item() for m in meas_tt]
                meas_x.extend(mx_ii)
                meas_y.extend(my_ii)
            color = (128/255, 128/255, 128/255)
            meas_x = np.asarray(meas_x)
            meas_y = np.asarray(meas_y)
            if not added_meas_lbl:
                f_hndl.axes[0].scatter(meas_x, meas_y, zorder=-1, alpha=0.35,
                                       color=color, marker='^',
                                       label='Measurements')
            else:
                f_hndl.axes[0].scatter(meas_x, meas_y, zorder=-1, alpha=0.35,
                                       color=color, marker='^')

        f_hndl.axes[0].grid(True)
        f_hndl.axes[0].set_title("Labeled State Trajectories")
        f_hndl.axes[0].set_ylabel("y-position")
        f_hndl.axes[0].set_xlabel("x-position")
        if lgnd_loc is not None:
            plt.legend(loc=lgnd_loc)
        plt.tight_layout()

        return f_hndl

    def plot_card_dist(self, **kwargs):
        """ Plots the current cardinality distribution.

        This assumes that the cardinality distribution has been calculated by
        the class.

        Keyword Args:
            f_hndl (Matplotlib figure): Current to figure to plot on. Always
                plots on axes[0], pass None to create a new figure

        Returns:
            (Matplotlib figure): Instance of the matplotlib figure used
        """

        f_hndl = kwargs.get('f_hndl', None)

        if len(self._card_dist) == 0:
            raise RuntimeWarning("Empty Cardinality")
            return f_hndl

        if f_hndl is None:
            f_hndl = plt.figure()
            f_hndl.add_subplot(1, 1, 1)

        x_vals = np.arange(0, len(self._card_dist))
        f_hndl.axes[0].bar(x_vals, self._card_dist)

        f_hndl.axes[0].set_title("Cardinality Distribution")
        f_hndl.axes[0].set_ylabel("Probability")
        f_hndl.axes[0].set_xlabel("Cardinality")
        plt.tight_layout()

        return f_hndl


class STMGeneralizedLabeledMultiBernoulli(GeneralizedLabeledMultiBernoulli):
    def predict_prob_density(self, **kwargs):
        """ Loops over all elements in a probability distribution and preforms
        the filter prediction.

        Keyword Args:
            probDensity (:py:class:`gasur.utilities.distributions.StudentsTMixture`): A
                probability density to run prediction on

        Returns:
            pd (:py:class:`gasur.utilities.distributions.StudentsTMixture`): The
                predicted probability density
        """
        probDensity = kwargs['probDensity']
        pd_tup = zip(probDensity.means,
                     probDensity.sclaings)
        c_in = np.zeros((self.filter.get_input_mat().shape[1], 1))
        pd = StudentsTMixture()
        pd.weights = probDensity.weights.copy()
        for ii, (m, P) in enumerate(pd_tup):
            self.filter.scale = P
            n_mean = self.filter.predict(cur_state=m, cur_input=c_in,
                                         **kwargs)
            pd.scalings.append(self.filter.scale.copy())
            pd.means.append(n_mean)

        return pd

    def correct_prob_density(self, meas, **kwargs):
        """ Loops over all elements in a probability distribution and preforms
        the filter correction.

        Keyword Args:
            probDensity (:py:class:`gasur.utilities.distributions.StudentsTMixture`): A
                probability density to run correction on
            meas (list): List of measurements, each is a N x 1 numpy array

        Returns:
            tuple containing

                - pd (:py:class:`gasur.utilities.distributions.StudentsTMixture`): The
                  corrected probability density
                - cost (float): Total cost of for the m best assignment
        """
        probDensity = kwargs['probDensity']

        pd = StudentsTMixture()
        for jj in range(0, len(probDensity.means)):
            self.filter.scale = probDensity.scalings[jj]
            state = probDensity.means[jj]
            (mean, qz) = self.filter.correct(meas=meas, cur_state=state,
                                             **kwargs)
            scale = self.filter.scalings
            w = qz * probDensity.weights[jj]
            pd.means.append(mean)
            pd.scalings.append(scale)
            pd.weights.append(w)
        lst = pd.weights
        lst = [x + np.finfo(float).eps for x in lst]
        pd.weights = lst
        cost = sum(pd.weights)
        for jj in range(0, len(pd.weights)):
            pd.weights[jj] /= cost
        return (pd, cost)


