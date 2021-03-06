"""Implements RFS tracking algorithms.

This module contains the classes and data structures
for RFS tracking related algorithms.
"""
import numpy as np
from numpy.linalg import cholesky, inv
import numpy.random as rnd
import matplotlib.pyplot as plt
from matplotlib.patches import Ellipse
import matplotlib.animation as animation
import abc
from copy import deepcopy
from warnings import warn

from gncpy.math import log_sum_exp, get_elem_sym_fnc
from gasur.utilities.distributions import GaussianMixture, StudentsTMixture
from gasur.utilities.graphs import k_shortest, murty_m_best
import gncpy.plotting as pltUtil


class RandomFiniteSetBase(metaclass=abc.ABCMeta):
    """ Generic base class for RFS based filters.

    Attributes:
        filter (gncpy.filters.BayesFilter): Filter handling dynamics
        prob_detection (float): Modeled probability an object is detected
        prob_survive (float): Modeled probability of object survival
        birth_terms (list): List of terms in the birth model
        clutter_rate (float): Rate of clutter
        clutter_density (float): Density of clutter distribution
        inv_chi2_gate (float): Chi squared threshold for gating the
            measurements
        debug_plots (bool): Saves data needed for extra debugging plots
    """

    def __init__(self, **kwargs):
        self.filter = None
        self.prob_detection = 1
        self.prob_survive = 1
        self.birth_terms = []
        self.clutter_rate = 0
        self.clutter_den = 0

        self.inv_chi2_gate = 0

        self.debug_plots = False
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

    def _gate_meas(self, meas, means, covs, **kwargs):
        """
        This gates measurements assuming a kalman filter (Gaussian noise).
        See :cite:`Cox1993_AReviewofStatisticalDataAssociationTechniquesforMotionCorrespondence`
        for details on the chi squared test used.

        Args:
            meas (list): each element is a 2d numpy array
            means (list): each element is a 2d numpy array
            covs (list): each element is a 2d numpy array
            **kwargs (kwargs): Passed to the filters measurement matrix and
                estimated measurement functions.

        Returns:
            list: Measurements passing gating criteria.

        """
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


class ProbabilityHypothesisDensity(RandomFiniteSetBase):
    """ Probability Hypothesis Density Filter


    """

    def __init__(self, **kwargs):
        self.gating_on = False
        self.inv_chi2_gate = 0
        self.extract_threshold = 0.5
        self.prune_threshold = 1*10**(-5)
        self.merge_threshold = 4
        self.save_covs = False
        self.max_gauss = 100

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
            return self._covs[-1]
        else:
            return []

    @property
    def cardinality(self):
        if len(self._states) ==  0:
            return 0
        else:
            return len(self._states[-1])

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

        for gm in self.birth_terms:
            self._gaussMix.weights.extend(gm.weights)
            self._gaussMix.means.extend(gm.means)
            self._gaussMix.covariances.extend(gm.covariances)

    def predict_prob_density(self, **kwargs):
        """ Loops over all elements in a probability distribution and performs
        the filter prediction.

        Keyword Args:
            probDensity (:py:class:`gasur.utilities.distributions.GaussianMixture`):
                A probability density to run prediction on

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

        if self.gating_on:
            meas = self._gate_meas(meas, self._gaussMix.means,
                                   self._gaussMix.covariances, **kwargs)

        self._meas_tab.append(meas)

        gmix = deepcopy(self._gaussMix)
        gmix.weights = [self.prob_miss_detection*x for x in gmix.weights]
        gm = self.correct_prob_density(meas=meas, probDensity=self._gaussMix,
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
        det_weights = [self.prob_detection * x for x in probDensity.weights]
        for z in meas:
            w_lst = []
            for jj in range(0, len(probDensity.means)):
                self.filter.cov = probDensity.covariances[jj]
                state = probDensity.means[jj]
                (mean, qz) = self.filter.correct(meas=z, cur_state=state,
                                                 **kwargs)
                cov = self.filter.cov
                w = qz * det_weights[jj]
                gm.means.append(mean)
                gm.covariances.append(cov)
                w_lst.append(w)
            gm.weights.extend([x / (self.clutter_rate * self.clutter_den
                               + sum(w_lst)) for x in w_lst])

        return gm

    def prune(self, **kwargs):
        """ Removes hypotheses below a threshold.

        This should be called once per time step after the correction and
        before the state extraction.
        """
        idx = np.where(np.asarray(self._gaussMix.weights)
                       < self.prune_threshold)
        idx = np.ndarray.flatten(idx[0])
        for index in sorted(idx, reverse=True):
            del self._gaussMix.means[index]
            del self._gaussMix.weights[index]
            del self._gaussMix.covariances[index]

    def merge(self, **kwargs):
        loop_inds = set(range(0, len(self._gaussMix.means)))

        w_lst = []
        m_lst = []
        p_lst = []
        while len(loop_inds) > 0:
            jj = np.argmax(self._gaussMix.weights)
            comp_inds = []
            inv_cov = inv(self._gaussMix.covariances[jj])
            for ii in loop_inds:
                diff = self._gaussMix.means[ii] - self._gaussMix.means[jj]
                val = diff.T @ inv_cov @ diff
                if val <= self.merge_threshold:
                    comp_inds.append(ii)

            w_new = sum([self._gaussMix.weights[ii] for ii in comp_inds])
            m_new = sum([self._gaussMix.weights[ii] * self._gaussMix.means[ii]
                         for ii in comp_inds]) / w_new
            p_new = sum([self._gaussMix.weights[ii]
                         * self._gaussMix.covariances[ii]
                         for ii in comp_inds]) / w_new

            w_lst.append(w_new)
            m_lst.append(m_new)
            p_lst.append(p_new)

            loop_inds = loop_inds.symmetric_difference(comp_inds)
            for ii in comp_inds:
                self._gaussMix.weights[ii] = -1

        self._gaussMix.weights = w_lst
        self._gaussMix.means = m_lst
        self._gaussMix.covariances = p_lst

    def cap(self, **kwargs):
        """ Removes least likely hypotheses until a maximum number is reached.

        This should be called once per time step after pruning and
        before the state extraction.
        """
        if len(self._gaussMix.weights) > self.max_gauss:
            idx = np.argsort(self._gaussMix.weights)
            w = sum(self._gaussMix.weights)
            for index in sorted(idx[0:-self.max_gauss], reverse=True):
                del self._gaussMix.means[index]
                del self._gaussMix.weights[index]
                del self._gaussMix.covariances[index]
            self._gaussMix.weights = [x * (w / sum(self._gaussMix.weights))
                                      for x in self._gaussMix.weights]

    def extract_states(self, **kwargs):
        """ Extracts the best state estimates.

        This extracts the best states from the distribution. It should be
        called once per time step after the correction function. This calls
        both the inner filters predict and correct functions so the keyword
        arguments must contain any additional variables needed by those
        functions.
        """
        inds = np.where(np.asarray(self._gaussMix.weights)
                        >= self.extract_threshold)
        inds = np.ndarray.flatten(inds[0])
        s_lst = []
        c_lst = []
        for jj in inds:
            num_reps = round(self._gaussMix.weights[jj])
            s_lst.extend([self._gaussMix.means[jj]] * num_reps)
            if self.save_covs:
                c_lst.extend([self._gaussMix.covariances[jj]] * num_reps)
        self._states.append(s_lst)
        if self.save_covs:
            self._covs.append(c_lst)

    def __ani_state_plotting(self, f_hndl, tt, states, show_sig, plt_inds, sig_bnd,
                             color, marker, state_lbl, added_sig_lbl,
                             added_state_lbl, scat=None):
        if scat is None:
            if not added_state_lbl:
                scat = f_hndl.axes[0].scatter([], [], color=color,
                                              edgecolors=(0, 0, 0),
                                              marker=marker)
            else:
                scat = f_hndl.axes[0].scatter([], [], color=color,
                                              edgecolors=(0, 0, 0),
                                              marker=marker, label=state_lbl)
        if len(states) == 0:
            return scat

        x = np.concatenate(states, axis=1)
        if show_sig:
            sigs = [None] * len(states)
            for ii, cov in enumerate(self._covs[tt]):
                sig = np.zeros((2, 2))
                sig[0, 0] = cov[plt_inds[0], plt_inds[0]]
                sig[0, 1] = cov[plt_inds[0], plt_inds[1]]
                sig[1, 0] = cov[plt_inds[1], plt_inds[0]]
                sig[1, 1] = cov[plt_inds[1], plt_inds[1]]
                sigs[ii] = sig

            # plot
            for ii, sig in enumerate(sigs):
                if sig is None:
                    continue
                w, h, a = pltUtil.calc_error_ellipse(sig, sig_bnd)
                if not added_sig_lbl:
                    s = r'${}\sigma$ Error Ellipses'.format(sig_bnd)
                    e = Ellipse(xy=x[plt_inds, ii], width=w,
                                height=h, angle=a, zorder=-10000,
                                animated=True, label=s)
                else:
                    e = Ellipse(xy=x[plt_inds, ii], width=w,
                                height=h, angle=a, zorder=-10000,
                                animated=True)
                e.set_clip_box(f_hndl.axes[0].bbox)
                e.set_alpha(0.15)
                e.set_facecolor(color)
                f_hndl.axes[0].add_patch(e)

        scat.set_offsets(x[plt_inds[0:2], :].T)
        return scat

    def plot_states(self, plt_inds, state_lbl='States', state_color=None,
                    **kwargs):
        """ Plots the best estimate for the states.

        This assumes that the states have been extracted. It's designed to plot
        two of the state variables (typically x/y position). The error ellipses
        are calculated according to :cite:`Hoover1984_AlgorithmsforConfidenceCirclesandEllipses`

        Keyword arguments are processed with
        :meth:`gasur.utilities.plotting.init_plotting_opts`. This function
        implements

            - f_hndl
            - true_states
            - sig_bnd
            - rng
            - meas_inds
            - lgnd_loc
            - marker

        Args:
            plt_inds (list): List of indices in the state vector to plot
            state_lbl (string): Value to appear in legend for the states. Only
                appears if the legend is shown

        Returns:
            (Matplotlib figure): Instance of the matplotlib figure used
        """

        opts = pltUtil.init_plotting_opts(**kwargs)
        f_hndl = opts['f_hndl']
        true_states = opts['true_states']
        sig_bnd = opts['sig_bnd']
        rng = opts['rng']
        meas_inds = opts['meas_inds']
        lgnd_loc = opts['lgnd_loc']
        marker = opts['marker']

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
        if state_color is None:
            color = (r, g, b)
        else:
            color = state_color
        for tt, states in enumerate(s_lst):
            if len(states) == 0:
                continue

            x = np.concatenate(states, axis=1)
            if show_sig:
                sigs = [None] * len(states)
                for ii, cov in enumerate(self._covs[tt]):
                    sig = np.zeros((2, 2))
                    sig[0, 0] = cov[plt_inds[0], plt_inds[0]]
                    sig[0, 1] = cov[plt_inds[0], plt_inds[1]]
                    sig[1, 0] = cov[plt_inds[1], plt_inds[0]]
                    sig[1, 1] = cov[plt_inds[1], plt_inds[1]]
                    sigs[ii] = sig

                # plot
                for ii, sig in enumerate(sigs):
                    if sig is None:
                        continue
                    w, h, a = pltUtil.calc_error_ellipse(sig, sig_bnd)
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
                    e.set_alpha(0.15)
                    e.set_facecolor(color)
                    f_hndl.axes[0].add_patch(e)

            if not added_state_lbl:
                f_hndl.axes[0].scatter(x[plt_inds[0], :], x[plt_inds[1], :],
                                       color=color, edgecolors=(0, 0, 0),
                                       marker=marker, label=state_lbl)
                added_state_lbl = True
            else:
                f_hndl.axes[0].scatter(x[plt_inds[0], :], x[plt_inds[1], :],
                                       color=color, edgecolors=(0, 0, 0),
                                       marker=marker)

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
                                       edgecolors=(0, 0, 0),
                                       label='Measurements')
            else:
                f_hndl.axes[0].scatter(meas_x, meas_y, zorder=-1, alpha=0.35,
                                       color=color, marker='^',
                                       edgecolors=(0, 0, 0))

        f_hndl.axes[0].grid(True)
        pltUtil.set_title_label(f_hndl, 0, opts, ttl="State Estimates",
                                x_lbl="x-position", y_lbl="y-position")
        if lgnd_loc is not None:
            plt.legend(loc=lgnd_loc)
        plt.tight_layout()

        return f_hndl

    def animate_state_plot(self, plt_inds, state_lbl='States', state_color=None,
                           interval=250, repeat=True, repeat_delay=1000,
                           save_path=None, **kwargs):
        opts = pltUtil.init_plotting_opts(**kwargs)
        f_hndl = opts['f_hndl']
        sig_bnd = opts['sig_bnd']
        rng = opts['rng']
        meas_inds = opts['meas_inds']
        lgnd_loc = opts['lgnd_loc']
        marker = opts['marker']

        plt_meas = meas_inds is not None
        show_sig = sig_bnd is not None and self.save_covs

        f_hndl.axes[0].grid(True)
        pltUtil.set_title_label(f_hndl, 0, opts, ttl="State Estimates",
                                x_lbl="x-position", y_lbl="y-position")

        fr_number = f_hndl.axes[0].annotate("0", (0, 1),
                                            xycoords="axes fraction",
                                            xytext=(10, -10),
                                            textcoords="offset points",
                                            ha="left", va="top",
                                            animated=False)

        added_sig_lbl = False
        added_state_lbl = False
        added_meas_lbl = False
        r = rng.random()
        b = rng.random()
        g = rng.random()
        if state_color is None:
            s_color = (r, g, b)
        else:
            s_color = state_color

        state_scat = f_hndl.axes[0].scatter([], [], color=s_color,
                                            edgecolors=(0, 0, 0),
                                            marker=marker, label=state_lbl)
        meas_scat = None
        if plt_meas:
            m_color = (128/255, 128/255, 128/255)

            if meas_scat is None:
                if not added_meas_lbl:
                    lbl = 'Measurements'
                    meas_scat = f_hndl.axes[0].scatter([], [], zorder=-1,
                                                       alpha=0.35,
                                                       color=m_color,
                                                       marker='^',
                                                       edgecolors='k',
                                                       label=lbl)
                    added_meas_lbl = True
                else:
                    meas_scat = f_hndl.axes[0].scatter([], [], zorder=-1,
                                                       alpha=0.35,
                                                       color=m_color,
                                                       marker='^',
                                                       edgecolors='k')

        def update(tt, *fargs):
            nonlocal added_sig_lbl
            nonlocal added_state_lbl
            nonlocal added_meas_lbl
            nonlocal state_scat
            nonlocal meas_scat
            nonlocal fr_number

            fr_number.set_text("Timestep: {j}".format(j=tt))

            states = self._states[tt]
            state_scat = self.__ani_state_plotting(f_hndl, tt, states,
                                                   show_sig, plt_inds,
                                                   sig_bnd, s_color, marker,
                                                   state_lbl, added_sig_lbl,
                                                   added_state_lbl,
                                                   scat=state_scat)
            added_sig_lbl = True
            added_state_lbl = True

            if plt_meas:
                meas_tt = self._meas_tab[tt]

                meas_x = [m[meas_inds[0]].item() for m in meas_tt]
                meas_y = [m[meas_inds[1]].item() for m in meas_tt]

                meas_x = np.asarray(meas_x)
                meas_y = np.asarray(meas_y)
                meas_scat.set_offsets(np.array([meas_x, meas_y]).T)

        # plt.figure(f_hndl.number)
        anim = animation.FuncAnimation(f_hndl, update,
                                       frames=len(self._states),
                                       interval=interval,
                                       repeat_delay=repeat_delay,
                                       repeat=repeat)

        if lgnd_loc is not None:
            plt.legend(loc=lgnd_loc)

        if save_path is not None:
            writer = animation.PillowWriter(fps=30)
            anim.save(save_path, writer=writer)

        return anim


class CardinalizedPHD(ProbabilityHypothesisDensity):
    """ Cardinalized Probability Hypothesis Density Filter


    """

    def __init__(self, **kwargs):
        self.agents_per_state = []
        self._max_expected_card = 10

        self._card_dist = np.zeros(self.max_expected_card + 1)  # local copy for internal modification
        self._card_dist[0] = 1
        self._card_time_hist = []  # local copy for internal modification
        self._n_states_per_time = []

        super().__init__(**kwargs)

    @property
    def max_expected_card(self):
        return self._max_expected_card

    @max_expected_card.setter
    def max_expected_card(self, x):
        self._card_dist = np.zeros(x + 1)
        self._card_dist[0] = 1
        self._max_expected_card = x

    @property
    def cardinality(self):
        return np.argmax(self._card_dist)

    def predict(self, **kwargs):
        """ Prediction step of the CPHD filter.

        This predicts new hypothesis, and propogates them to the next time
        step. It also updates the cardinality distribution. Because this calls
        the inner filter's predict function, the keyword arguments must contain
        any information needed by that function.

        Keyword Args:

        """
        super().predict(**kwargs)

        survive_cdn_predict = np.zeros(self.max_expected_card + 1)
        for j in range(0, self.max_expected_card):
            terms = np.zeros((self.max_expected_card + 1, 1))
            for i in range(j, self.max_expected_card + 1):
                temp = []
                temp.append(np.sum(np.log(range(1, i + 1))))
                temp.append(-np.sum(np.log(range(1, j + 1))))
                temp.append(np.sum(np.log(range(1, i - j + 1))))
                temp.append(j * np.log(self.prob_survive))
                temp.append((i - j) * np.log(self.prob_death))
                terms[i, 0] = np.exp(np.sum(temp)) * self._card_dist[i]
            survive_cdn_predict[j] = np.sum(terms)

        cdn_predict = np.zeros(self.max_expected_card + 1)
        for n in range(0, self.max_expected_card + 1):
            terms = np.zeros((self.max_expected_card + 1, 1))
            for j in range(0, n + 1):
                temp = []
                birth = np.zeros(len(self.birth_terms))
                for b in range(0, len(self.birth_terms)):
                    birth[b] = self.birth_terms[b].weights[0]
                temp.append(np.sum(birth))
                temp.append((n - j) * np.log(np.sum(birth)))
                temp.append(-np.sum(np.log(range(1, n - j + 1))))
                terms[j, 0] = np.exp(np.sum(temp)) * survive_cdn_predict[j]
            cdn_predict[n] = np.sum(terms)
        self._card_dist = (cdn_predict/np.sum(cdn_predict)).copy()

    def correct(self, **kwargs):
        """ Correction step of the CPHD filter.

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

        if self.gating_on:
            meas = self._gate_meas(meas, self._gaussMix.means,
                                   self._gaussMix.covariances, **kwargs)

        self._meas_tab.append(meas)

        gmix = deepcopy(self._gaussMix)  # predicted gm

        self._gaussMix = self.correct_prob_density(meas=meas, probDensity=gmix,
                                                   **kwargs)

    def correct_prob_density(self, meas, **kwargs):
        """ Loops over all elements in a probability distribution and preforms
        the filter correction.

        Keyword Args:
            probDensity (:py:class:`gasur.utilities.distributions.GaussianMixture`): A
                probability density to run correction on
            meas (list): List of measurements, each is a N x 1 numpy array

        Returns:
                - gm (:py:class:`gasur.utilities.distributions.GaussianMixture`): The
                  corrected probability density
        """
        probDensity = kwargs['probDensity']
        w_pred = np.zeros((len(probDensity.weights), 1))
        for i in range(0, len(probDensity.weights)):
            w_pred[i] = probDensity.weights[i]

        xdim = len(probDensity.means[0])

        plen = len(probDensity.means)
        zlen = len(meas)

        qz_temp = np.zeros((plen, zlen))
        mean_temp = np.zeros((zlen, xdim, plen))
        cov_temp = np.zeros((plen, xdim, xdim))

        for z_ind in range(0, zlen):
            for p_ind in range(0, plen):
                self.filter.cov = probDensity.covariances[p_ind]
                state = probDensity.means[p_ind]
                (mean, qz) = self.filter.correct(meas=meas[z_ind],
                                                 cur_state=state,
                                                 **kwargs)
                cov = self.filter.cov
                qz_temp[p_ind, z_ind] = qz
                mean_temp[z_ind, :, p_ind] = np.ndarray.flatten(mean)
                cov_temp[[p_ind], :, :] = cov

        xivals = np.zeros(zlen)
        pdc = self.prob_detection/self.clutter_den
        for e in range(0, zlen):
            xivals[e] = pdc * np.dot(w_pred.T, qz_temp[:, [e]])
            # xilog = []
            # for c in range(0, len(w_pred)):
            #     xilog.append(np.log(w_pred[[c]]).item())
            #     xilog.append(np.log(qz_temp[c, e]))
            # xivals[e] = np.exp(np.log(pdc) + np.sum(xilog))

        esfvals_E = get_elem_sym_fnc(xivals)
        esfvals_D = np.zeros((zlen, zlen))

        for j in range(0, zlen):
            xi_temp = xivals.copy()
            xi_temp = np.delete(xi_temp, j)
            esfvals_D[:, [j]] = get_elem_sym_fnc(xi_temp)

        ups0_E = np.zeros((self.max_expected_card + 1, 1))
        ups1_E = np.zeros((self.max_expected_card + 1, 1))
        ups1_D = np.zeros((self.max_expected_card + 1, zlen))

        tot_w_pred = sum(w_pred)
        for nn in range(0, self.max_expected_card + 1):
            terms0_E = np.zeros((min(zlen, nn) + 1))
            for jj in range(0, min(zlen, nn) + 1):
                t1 = -self.clutter_rate + (zlen - jj) \
                    * np.log(self.clutter_rate)
                t2 = sum([np.log(x) for x in range(1, nn + 1)])
                t3 = -1 * sum([np.log(x) for x in range(1, nn - jj + 1)])
                t4 = (nn - jj) * np.log(self.prob_death)
                t5 = -jj * np.log(tot_w_pred)
                terms0_E[jj] = np.exp(t1 + t2 + t3 + t4 + t5) * esfvals_E[jj]
            ups0_E[nn] = np.sum(terms0_E)

            terms1_E = np.zeros((min(zlen, nn) + 1))
            for jj in range(0, min(zlen, nn) + 1):
                if nn >= jj + 1:
                    t1 = -self.clutter_rate + (zlen - jj) \
                        * np.log(self.clutter_rate)
                    t2 = sum([np.log(x) for x in range(1, nn + 1)])
                    t3 = -1 * sum([np.log(x)
                                   for x in range(1, nn - (jj + 1) + 1)])
                    t4 = (nn - (jj + 1)) * np.log(self.prob_death)
                    t5 = -(jj + 1) * np.log(tot_w_pred)
                    terms1_E[jj] = np.exp(t1 + t2 + t3 + t4 + t5) \
                        * esfvals_E[jj]
            ups1_E[nn] = np.sum(terms1_E)

            if zlen != 0:
                terms1_D = np.zeros((min(zlen - 1, nn) + 1, zlen))
                for ell in range(1, zlen + 1):
                    for jj in range(0, min((zlen - 1), nn) + 1):
                        if nn >= jj + 1:
                            t1 = -self.clutter_rate + ((zlen - 1) - jj) \
                                * np.log(self.clutter_rate)
                            t2 = sum([np.log(x) for x in range(1, nn + 1)])
                            t3 = -1 * sum([np.log(x)
                                           for x in range(1,
                                                          nn - (jj + 1) + 1)])
                            t4 = (nn - (jj + 1)) * np.log(self.prob_death)
                            t5 = -(jj + 1) * np.log(tot_w_pred)
                            terms1_D[jj, ell - 1] = np.exp(t1 + t2 + t3
                                                           + t4 + t5) \
                                * esfvals_D[jj, ell - 1]
                ups1_D[nn, :] = np.sum(terms1_D, axis=0)

        gmix = deepcopy(probDensity)
        w_update = ((ups1_E.T @ self._card_dist)
                    / (ups0_E.T @ self._card_dist)) * self.prob_miss_detection * w_pred
        # w_update = np.exp(w_update)

        gmix.weights = [x.item() for x in w_update]

        for ee in range(0, zlen):
            wt_1 = ((ups1_D[:, [ee]].T @ self._card_dist) / (ups0_E.T @ self._card_dist)).reshape((1, 1))
            wt_2 = self.prob_detection * qz_temp[:, [ee]] / self.clutter_den * w_pred
            w_temp = wt_1 * wt_2
            for ww in range(0, w_temp.shape[0]):
                gmix.weights.append(w_temp[ww].item())
                gmix.means.append(mean_temp[ee, :, ww].reshape((xdim, 1)))
                gmix.covariances.append(cov_temp[ww, :, :])

        cdn_update = self._card_dist.copy()
        for ii in range(0, len(cdn_update)):
            cdn_update[ii] = ups0_E[ii] * self._card_dist[ii]

        self._card_dist = cdn_update / np.sum(cdn_update)
        self._card_time_hist.append((np.argmax(self._card_dist).item(),
                                     np.std(self._card_dist)))

        return gmix

    def extract_states(self, **kwargs):
        """ Extracts the best state estimates.

        This extracts the best states from the distribution. It should be
        called once per time step after the correction function. This calls
        both the inner filters predict and correct functions so the keyword
        arguments must contain any additional variables needed by those
        functions.
        """
        s_weights = np.argsort(self._gaussMix.weights)[::-1]
        s_lst = []
        c_lst = []
        self.agents_per_state = []
        ii = 0
        tot_agents = 0
        while ii < s_weights.size and tot_agents < self.cardinality:
            idx = s_weights[ii]

            n_agents = round(self._gaussMix.weights[idx])
            if n_agents <= 0:
                msg = "Gaussian weights are 0 before reaching cardinality"
                warn(msg, RuntimeWarning)
                break

            tot_agents += n_agents
            self.agents_per_state.append(n_agents)

            s_lst.append(self._gaussMix.means[idx])
            if self.save_covs:
                c_lst.append(self._gaussMix.covariances[idx])

            ii += 1

        self._states.append(s_lst)
        if self.save_covs:
            self._covs.append(c_lst)
        if self.debug_plots:
            self._n_states_per_time.append(ii)

    def plot_card_dist(self, **kwargs):
        """ Plots the current cardinality distribution.

        This assumes that the cardinality distribution has been calculated by
        the class.

        Keyword arguments are processed with
        :meth:`gasur.utilities.plotting.init_plotting_opts`. This function
        implements

            - f_hndl

        Returns:
            (Matplotlib figure): Instance of the matplotlib figure used
        """

        opts = pltUtil.init_plotting_opts(**kwargs)
        f_hndl = opts['f_hndl']

        if len(self._card_dist) == 0:
            raise RuntimeWarning("Empty Cardinality")
            return f_hndl

        if f_hndl is None:
            f_hndl = plt.figure()
            f_hndl.add_subplot(1, 1, 1)

        x_vals = np.arange(0, len(self._card_dist))
        f_hndl.axes[0].bar(x_vals, self._card_dist)

        pltUtil.set_title_label(f_hndl, 0, opts,
                                ttl="Cardinality Distribution",
                                x_lbl="Cardinality", y_lbl="Probability")
        plt.tight_layout()

        return f_hndl

    def plot_card_time_hist(self, **kwargs):
        """ Plots the current cardinality time history.

        This assumes that the cardinality distribution has been calculated by
        the class.

        Keyword arguments are processed with
        :meth:`gasur.utilities.plotting.init_plotting_opts`. This function
        implements

            - f_hndl
            - sig_bnd
            - time_vec
            - lgnd_loc

        Keyword Args:
            true_card (array like): List of the true cardinality at each time

        Returns:
            (Matplotlib figure): Instance of the matplotlib figure used
        """

        opts = pltUtil.init_plotting_opts(**kwargs)
        f_hndl = opts['f_hndl']
        sig_bnd = opts['sig_bnd']
        time_vec = opts['time_vec']
        lgnd_loc = opts['lgnd_loc']

        true_card = kwargs.get('true_card', None)

        if len(self._card_time_hist) == 0:
            raise RuntimeWarning("Empty Cardinality")
            return f_hndl

        if sig_bnd is not None:
            stds = [sig_bnd * x[1] for x in self._card_time_hist]
        card = [x[0] for x in self._card_time_hist]

        if f_hndl is None:
            f_hndl = plt.figure()
            f_hndl.add_subplot(1, 1, 1)

        if time_vec is None:
            x_vals = [ii for ii in range(0, len(card))]
        else:
            x_vals = time_vec

        if true_card is not None:
            if len(true_card) != len(x_vals):
                c_len = len(true_card)
                t_len = len(x_vals)
                msg = "True Cardinality vector length ({})".format(c_len) \
                    + " does not match time vector length ({})".format(t_len)
                warn(msg)
            else:
                f_hndl.axes[0].plot(x_vals, true_card, color='g',
                                    label='True Cardinality',
                                    linestyle='-')

        f_hndl.axes[0].plot(x_vals, card, label='Cardinality', color='k',
                            linestyle='--')

        if sig_bnd is not None:
            lbl = r'${}\sigma$ Bound'.format(sig_bnd)
            f_hndl.axes[0].plot(x_vals, [x + s for (x, s) in zip(card, stds)],
                                linestyle='--', color='r', label=lbl)
            f_hndl.axes[0].plot(x_vals, [x - s for (x, s) in zip(card, stds)],
                                linestyle='--', color='r')

        if lgnd_loc is not None:
            plt.legend(loc=lgnd_loc)

        plt.grid(True)
        pltUtil.set_title_label(f_hndl, 0, opts,
                                ttl="Cardinality History",
                                x_lbl="Time", y_lbl="Cardinality")

        plt.tight_layout()

        return f_hndl

    def plot_number_states_per_time(self, **kwargs):
        opts = pltUtil.init_plotting_opts(**kwargs)
        f_hndl = opts['f_hndl']
        lgnd_loc = opts['lgnd_loc']

        if not self.debug_plots:
            msg = 'Debug plots turned off'
            warn(msg)
            return f_hndl

        if f_hndl is None:
            f_hndl = plt.figure()
            f_hndl.add_subplot(1, 1, 1)

        if lgnd_loc is not None:
            plt.legend(loc=lgnd_loc)

        x_vals = [ii for ii in range(0, len(self._n_states_per_time))]

        f_hndl.axes[0].plot(x_vals, self._n_states_per_time)
        plt.grid(True)
        pltUtil.set_title_label(f_hndl, 0, opts,
                                ttl="Gaussians per Timestep",
                                x_lbl="Time", y_lbl="Number of Gaussians")

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

    @property
    def cardinality(self):
        """ The cardinality estimate
        """
        return np.argmax(self._card_dist)

    def _gen_birth_tab(self, time_step):
        log_cost = []
        birth_tab = []
        for ii, (distrib, p) in enumerate(self.birth_terms):
            cost = p / (1 - p)
            log_cost.append(-np.log(cost))
            entry = self._TabEntry()
            entry.probDensity = deepcopy(distrib)
            entry.label = (time_step, ii)
            birth_tab.append(entry)

        return birth_tab, log_cost

    def _gen_birth_hyps(self, paths, hyp_costs):
        birth_hyps = []
        tot_b_prob = sum([np.log(1 - x[1]) for x in self.birth_terms])
        for (p, c) in zip(paths, hyp_costs):
            hyp = self._HypothesisHelper()
            # NOTE: this may suffer from underflow and can be improved
            hyp.assoc_prob = tot_b_prob - c.item()
            hyp.track_set = p
            birth_hyps.append(hyp)
        lse = log_sum_exp([x.assoc_prob for x in birth_hyps])
        for ii in range(0, len(birth_hyps)):
            birth_hyps[ii].assoc_prob = np.exp(birth_hyps[ii].assoc_prob - lse)

        return birth_hyps

    def _gen_surv_tab(self, **kwargs):
        surv_tab = []
        for (ii, track) in enumerate(self._track_tab):
            entry = self.predict_track_tab_entry(track, **kwargs)

            surv_tab.append(entry)

        return surv_tab

    def _gen_surv_hyps(self, avg_prob_survive, avg_prob_death):
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
                cost = avg_prob_survive[hyp.track_set] \
                    / avg_prob_death[hyp.track_set]
                log_cost = -np.log(cost)  # this is length hyp.num_tracks
                k = np.round(self.req_surv * np.sqrt(hyp.assoc_prob)
                             / sum_sqrt_w)
                (paths, hyp_cost) = k_shortest(np.array(log_cost), k)

                pdeath_log = np.sum([np.log(avg_prob_death[ii])
                                     for ii in hyp.track_set])

                for (p, c) in zip(paths, hyp_cost):
                    new_hyp = self._HypothesisHelper()
                    new_hyp.assoc_prob = pdeath_log \
                        + np.log(hyp.assoc_prob) - c.item()
                    if len(p) > 0:
                        new_hyp.track_set = [hyp.track_set[ii] for ii in p]
                    else:
                        new_hyp.track_set = []
                    surv_hyps.append(new_hyp)

        lse = log_sum_exp([x.assoc_prob for x in surv_hyps])
        for ii in range(0, len(surv_hyps)):
            surv_hyps[ii].assoc_prob = np.exp(surv_hyps[ii].assoc_prob - lse)

        return surv_hyps

    def _calc_avg_prob_surv_death(self, **kwargs):
        avg_prob_survive = self.prob_survive * np.ones(len(self._track_tab))
        avg_prob_death = 1 - avg_prob_survive

        return avg_prob_survive, avg_prob_death

    def _set_pred_hyps(self, birth_tab, birth_hyps, surv_hyps):
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
            n_val = self._hypotheses[ii].assoc_prob / tot_w
            self._hypotheses[ii].assoc_prob = n_val

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
        birth_tab, log_cost = self._gen_birth_tab(time_step)

        # get K best hypothesis, and their index in the lookup table
        (paths, hyp_costs) = k_shortest(np.array(log_cost), self.req_births)

        # calculate association probabilities for birth hypothesis
        birth_hyps = self._gen_birth_hyps(paths, hyp_costs)

        # Init and propagate surviving track table
        surv_tab = self._gen_surv_tab(**kwargs)

        # Calculation for average survival/death probabilities
        (avg_prob_survive,
         avg_prob_death) = self._calc_avg_prob_surv_death(**kwargs)

        # loop over postierior components
        surv_hyps = self._gen_surv_hyps(avg_prob_survive, avg_prob_death)

        # Get  predicted hypothesis by convolution
        self._track_tab = birth_tab + surv_tab
        self._set_pred_hyps(birth_tab, birth_hyps, surv_hyps)

        self._card_dist = self.calc_card_dist(self._hypotheses)
        self._clean_predictions()

    def predict_track_tab_entry(self, tab, **kwargs):
        """ Loops over all elements in a probability distribution and preforms
        the filter correction.

        Args:
            tab (Track Table class): An entry in the track table, class is
                internal to the parent class.

        Returns:
            tuple containing

                - newTab (Track table class): A track table class instance with
                  predicted probability density
        """
        probDensity = tab.probDensity
        newTab = deepcopy(tab)

        pd = self.predict_prob_density(probDensity, **kwargs)
        newTab.probDensity = pd
        return newTab

    def predict_prob_density(self, probDensity, **kwargs):
        """ Loops over all elements in a probability distribution and preforms
        the filter prediction.

        Args:
            probDensity (:py:class:`gasur.utilities.distributions.GaussianMixture`): A
                probability density to run prediction on

        Returns:
            gm (:py:class:`gasur.utilities.distributions.GaussianMixture`): The
                predicted probability density
        """
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

    def _gen_cor_tab(self, num_meas, meas, **kwargs):
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
                (up_tab[s_to_ii], cost) = \
                    self.correct_track_tab_entry(z, ent, **kwargs)

                # update association history with current measurement index
                up_tab[s_to_ii].meas_assoc_hist += [emm]
                all_cost_m[ii, emm] = cost
        return up_tab, all_cost_m

    def _gen_cor_hyps(self, num_meas, avg_prob_detect, avg_prob_miss_detect,
                      all_cost_m):
        num_pred = len(self._track_tab)
        up_hyps = []
        if num_meas == 0:
            for hyp in self._hypotheses:
                pmd_log = np.sum([np.log(avg_prob_miss_detect[ii])
                                  for ii in hyp.track_set])
                hyp.assoc_prob = -self.clutter_rate + pmd_log \
                    + np.log(hyp.assoc_prob)
                up_hyps.append(hyp)
        else:
            clutter = self.clutter_rate * self.clutter_den
            ss_w = 0
            for p_hyp in self._hypotheses:
                ss_w += np.sqrt(p_hyp.assoc_prob)
            for p_hyp in self._hypotheses:
                if p_hyp.num_tracks == 0:  # all clutter
                    new_hyp = self._HypothesisHelper()
                    new_hyp.assoc_prob = -self.clutter_rate + num_meas \
                        * np.log(clutter) + np.log(p_hyp.assoc_prob)
                    new_hyp.track_set = p_hyp.track_set.copy()
                    up_hyps.append(new_hyp)

                else:
                    pd = [avg_prob_detect[ii] for ii in p_hyp.track_set]
                    pmd = [avg_prob_miss_detect[ii] for ii in p_hyp.track_set]
                    ratio = np.array([p_d / q_d for p_d, q_d in zip(pd, pmd)])

                    ratio = ratio.reshape((ratio.size, 1))
                    ratio = np.tile(ratio, (1, num_meas))

                    cost_m = ratio * all_cost_m[p_hyp.track_set, :] \
                        / clutter
                    if (np.abs(cost_m) == np.inf).any():
                        continue

                    neg_log = -np.log(cost_m)
                    m = np.round(self.req_upd * np.sqrt(p_hyp.assoc_prob)
                                 / ss_w)
                    m = int(m.item())
                    [assigns, costs] = murty_m_best(neg_log, m)

                    pmd_log = np.sum([np.log(avg_prob_miss_detect[ii])
                                      for ii in p_hyp.track_set])
                    for (a, c) in zip(assigns, costs):
                        new_hyp = self._HypothesisHelper()
                        new_hyp.assoc_prob = -self.clutter_rate + num_meas \
                            * np.log(clutter) + pmd_log \
                            + np.log(p_hyp.assoc_prob) - c
                        lst1 = [num_pred * x for x in a]
                        lst2 = p_hyp.track_set.copy()
                        new_hyp.track_set = [sum(x) for x in zip(lst1, lst2)]
                        up_hyps.append(new_hyp)

        lse = log_sum_exp([x.assoc_prob for x in up_hyps])
        for ii in range(0, len(up_hyps)):
            up_hyps[ii].assoc_prob = np.exp(up_hyps[ii].assoc_prob - lse)

        return up_hyps

    def _calc_avg_prob_det_mdet(self, **kwargs):
        avg_prob_detect = self.prob_detection * np.ones(len(self._track_tab))
        avg_prob_miss_detect = 1 - avg_prob_detect

        return avg_prob_detect, avg_prob_miss_detect

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
        cor_tab, all_cost_m = self._gen_cor_tab(num_meas, meas, **kwargs)

        # Calculation for average detection/missed probabilities
        avg_prob_det, avg_prob_mdet = self._calc_avg_prob_det_mdet(**kwargs)

        # component updates
        cor_hyps = self._gen_cor_hyps(num_meas, avg_prob_det, avg_prob_mdet,
                                      all_cost_m)

        # save values and cleanup
        self._track_tab = cor_tab
        self._hypotheses = cor_hyps
        self._card_dist = self.calc_card_dist(self._hypotheses)
        self._clean_updates()

    def correct_track_tab_entry(self, meas, tab, **kwargs):
        """ Loops over all elements in a probability distribution and preforms
        the filter correction.

        Args:
            tab (Track Table class): An entry in the track table, class is
                internal to the parent class.
            meas (list): List of measurements, each is a N x 1 numpy array

        Returns:
            tuple containing

                - newTab (Track table class): A track table class instance with
                  corrected probability density
                - cost (float): Total cost of for the m best assignment
        """
        probDensity = tab.probDensity
        newTab = deepcopy(tab)

        pd, cost = self.correct_prob_density(meas, probDensity, **kwargs)
        newTab.probDensity = pd
        return newTab, cost

    def correct_prob_density(self, meas, probDensity, **kwargs):
        """ Loops over all elements in a probability distribution and preforms
        the filter correction.

        Args:
            probDensity (:py:class:`gasur.utilities.distributions.GaussianMixture`): A
                probability density to run correction on
            meas (list): List of measurements, each is a N x 1 numpy array

        Returns:
            tuple containing

                - gm (:py:class:`gasur.utilities.distributions.GaussianMixture`): The
                  corrected probability density
                - cost (float): Total cost of for the m best assignment
        """
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
                pd = self.predict_prob_density(pd, **kwargs)

                # measurement correction for GM
                tt = b_time + t_after_b
                if emm is not None:
                    meas = self._meas_tab[tt][emm].copy()
                    pd = self.correct_prob_density(meas, pd, **kwargs)[0]

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

    def plot_states_labels(self, plt_inds, **kwargs):
        """ Plots the best estimate for the states and labels.

        This assumes that the states have been extracted. It's designed to plot
        two of the state variables (typically x/y position). The error ellipses
        are calculated according to :cite:`Hoover1984_AlgorithmsforConfidenceCirclesandEllipses`

        Keywrod arguments are processed with
        :meth:`gasur.utilities.plotting.init_plotting_opts`. This function
        implements

            - f_hndl
            - true_states
            - sig_bnd
            - rng
            - meas_inds
            - lgnd_loc

        Args:
            plt_inds (list): List of indices in the state vector to plot

        Returns:
            (Matplotlib figure): Instance of the matplotlib figure used
        """

        opts = pltUtil.init_plotting_opts(**kwargs)
        f_hndl = opts['f_hndl']
        true_states = opts['true_states']
        sig_bnd = opts['sig_bnd']
        rng = opts['rng']
        meas_inds = opts['meas_inds']
        lgnd_loc = opts['lgnd_loc']

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
            if states is not None and len(states) > 0:
                x_dim = states[0].size
                break

        # get unique labels
        u_lbls = []
        for lbls in l_lst:
            if lbls is None:
                continue
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
                if lbls is None:
                    continue
                if lbl in lbls:
                    ii = lbls.index(lbl)
                    if s_lst[tt][ii] is not None:
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
                    w, h, a = pltUtil.calc_error_ellipse(sig, sig_bnd)
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
        pltUtil.set_title_label(f_hndl, 0, opts,
                                ttl="Labeled State Trajectories",
                                x_lbl="x-position", y_lbl="y-position")
        if lgnd_loc is not None:
            plt.legend(loc=lgnd_loc)
        plt.tight_layout()

        return f_hndl

    def plot_card_dist(self, **kwargs):
        """ Plots the current cardinality distribution.

        This assumes that the cardinality distribution has been calculated by
        the class.

        Keywrod arguments are processed with
        :meth:`gasur.utilities.plotting.init_plotting_opts`. This function
        implements

            - f_hndl

        Returns:
            (Matplotlib figure): Instance of the matplotlib figure used
        """

        opts = pltUtil.init_plotting_opts(**kwargs)
        f_hndl = opts['f_hndl']

        if len(self._card_dist) == 0:
            raise RuntimeWarning("Empty Cardinality")
            return f_hndl

        if f_hndl is None:
            f_hndl = plt.figure()
            f_hndl.add_subplot(1, 1, 1)

        x_vals = np.arange(0, len(self._card_dist))
        f_hndl.axes[0].bar(x_vals, self._card_dist)

        pltUtil.set_title_label(f_hndl, 0, opts,
                                ttl="Cardinality Distribution",
                                x_lbl="Cardinality", y_lbl="Probability")
        plt.tight_layout()

        return f_hndl


class STMGeneralizedLabeledMultiBernoulli(GeneralizedLabeledMultiBernoulli):
    def predict_prob_density(self, probDensity, **kwargs):
        """ Loops over all elements in a probability distribution and preforms
        the filter prediction.

        Keyword Args:
            probDensity (:py:class:`gasur.utilities.distributions.StudentsTMixture`): A
                probability density to run prediction on

        Returns:
            pd (:py:class:`gasur.utilities.distributions.StudentsTMixture`): The
                predicted probability density
        """
        self.filter.dof = probDensity.dof
        pd_tup = zip(probDensity.means,
                     probDensity.scalings)
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

    def correct_prob_density(self, meas, probDensity, **kwargs):
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
        self.filter.dof = probDensity.dof
        pd = StudentsTMixture()
        for jj in range(0, len(probDensity.means)):
            self.filter.scale = probDensity.scalings[jj]
            state = probDensity.means[jj]
            (mean, qz) = self.filter.correct(meas=meas, cur_state=state,
                                             **kwargs)
            scale = self.filter.scale.copy()
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

    def _gate_meas(self, meas, means, covs, **kwargs):
        if len(meas) == 0:
            return []

        scalings = []
        for ent in self._track_tab:
            scalings.extend(ent.probDensity.scalings)

        valid = []
        for (m, p) in zip(means, scalings):
            meas_mat = self.filter.get_meas_mat(m, **kwargs)
            est = self.filter.get_est_meas(m, **kwargs)
            factor = self.filter.meas_noise_dof * (self.filter.dof - 2) \
                / (self.filter.dof * (self.filter.meas_noise_dof - 2))
            P_zz = meas_mat @ p @ meas_mat.T + factor * self.filter.meas_noise
            inv_P = inv(P_zz)

            for (ii, z) in enumerate(meas):
                if ii in valid:
                    continue
                innov = z - est
                dist = innov.T @ inv_P @ innov
                if dist < self.inv_chi2_gate:
                    valid.append(ii)

        valid.sort()
        return [meas[ii] for ii in valid]


class SMCGeneralizedLabeledMultiBernoulli(GeneralizedLabeledMultiBernoulli):
    """ This implements a Sequential Monte Carlo GLMB filter.

    This is based on :cite:`Vo2014_LabeledRandomFiniteSetsandtheBayesMultiTargetTrackingFilter`
    It does not account for agents spawned from existing tracks, only agents
    birthed from the given birth model.

    Attributes:
        compute_prob_detection (function): Must take a likst of particles as
            the first argument and kwargs as the next. Returns the average
            probability of detection for the list of particles
        compute_prob_survive (function): Must take a likst of particles as
            the first argument and kwargs as the next. Returns the average
            probability of survival for the list of particles
    """

    class _TabEntry:
        def __init__(self):
            self.label = ()  # time step born, index of birth model born from
            self.probDensity = None  # must be a distribution class
            self.meas_assoc_hist = []  # list indices into measurement list per time step
            self.prop_parts = []

    def __init__(self, **kwargs):
        self.compute_prob_detection = kwargs.get('compute_prob_detection',
                                                 None)
        self.compute_prob_survive = kwargs.get('compute_prob_survive', None)

        super().__init__(**kwargs)

    def _calc_avg_prob_surv_death(self, **kwargs):
        avg_prob_survive = np.zeros(len(self._track_tab))
        for tabidx, ent in enumerate(self._track_tab):
            p_surv = self.compute_prob_survive(ent.probDensity.particles,
                                               **kwargs)
            avg_prob_survive[tabidx] = np.sum(np.array(ent.probDensity.weights)
                                              * p_surv)

        avg_prob_death = 1 - avg_prob_survive

        return avg_prob_survive, avg_prob_death

    def predict_track_tab_entry(self, tab, **kwargs):
        """ Loops over all elements in a probability distribution and preforms
        the filter correction.

        Args:
            tab (Track Table class): An entry in the track table, class is
                internal to the parent class.

        Returns:
            tuple containing

                - newTab (Track table class): A track table class instance with
                  predicted probability density
        """
        newTab = super().predict_track_tab_entry(tab, **kwargs)
        newTab.prop_parts = deepcopy(self.filter._prop_parts)
        return newTab

    def predict_prob_density(self, probDensity, **kwargs):
        """ This predicts the next probability density.


        Args:
            probDensity (:py:mod:`gasur.utilities.distributions`): Probability
                distribution to use for prediction
            **kwargs (dict): Passed through to the filters predict function.

        Returns:
            newProbDen (:py:mod:`gasur.utilities.distributions`): Predicted
                probability distribution.

        """
        self.filter.init_from_dist(deepcopy(probDensity))
        # cls_type = type(probDensity)
        # newProbDen = cls_type()

        self.filter.predict(**kwargs)

        newProbDen = self.filter.extract_dist()

        new_weights = [w * self.prob_survive for p, w in probDensity]
        tot = sum(new_weights)
        w_lst = [w / tot for w in new_weights]
        newProbDen.update_weights(w_lst)

        return newProbDen

    def _calc_avg_prob_det_mdet(self, **kwargs):
        avg_prob_detect = np.zeros(len(self._track_tab))
        for tabidx, ent in enumerate(self._track_tab):
            p_detect = self.compute_prob_detection(ent.probDensity.particles,
                                                   **kwargs)
            avg_prob_detect[tabidx] = np.sum(np.array(ent.probDensity.weights)
                                             * p_detect)

        avg_prob_miss_detect = 1 - avg_prob_detect

        return avg_prob_detect, avg_prob_miss_detect

    def correct_track_tab_entry(self, meas, tab, **kwargs):
        """ Loops over all elements in a probability distribution and preforms
        the filter correction.

        Args:
            tab (Track Table class): An entry in the track table, class is
                internal to the parent class.
            meas (list): List of measurements, each is a N x 1 numpy array

        Returns:
            tuple containing

                - newTab (Track table class): A track table class instance with
                  corrected probability density
                - cost (float): Total cost of for the m best assignment
        """
        if len(tab.prop_parts) == 0:
            # assume this is a new birth and prob_parts hasn't been initialized
            prop_parts = deepcopy(tab.probDensity.particles)
        else:
            prop_parts = deepcopy(tab.prop_parts)
        self.filter._prop_parts = prop_parts
        newTab, cost = super().correct_track_tab_entry(meas, tab, **kwargs)
        newTab.prop_parts = self.filter._prop_parts
        return newTab, cost

    def correct_prob_density(self, meas, probDensity, **kwargs):
        """ This corrects the probability density and resamples.


        Args:
            meas (numpy array): Measurement vector as a 2D numpy array
            probDensity (:py:mod:`gasur.utilities.distributions`): Probability
                distribution to use for prediction
            **kwargs (dict): Passed through to the filters correct function.

        Returns:
            tuple containing

                - (:py:mod:`gasur.utilities.distributions`): Corrected
                and resampledprobability distribution.
                - (float): Sum of the unnormalized weights
        """
        self.filter.init_from_dist(deepcopy(probDensity))
        # cls_type = type(probDensity)
        # newProbDen = cls_type()

        likelihood, inds_removed = self.filter.correct(meas, **kwargs)[1:3]
        newProbDen = self.filter.extract_dist()

        new_weights = self.prob_detection * np.array(likelihood)
        tot = sum(new_weights)
        if tot > 0 and np.abs(tot) != np.inf:
            new_weights = [w / tot for w in new_weights]
        else:
            new_weights = [np.inf] * len(new_weights)
            tot = np.inf  # division by 0 would give inf

        newProbDen.update_weights(new_weights)

        return newProbDen, tot

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
                pd = self.predict_prob_density(pd, **kwargs)

                # measurement correction for GM
                tt = b_time + t_after_b
                if emm is not None:
                    meas = self._meas_tab[tt][emm].copy()
                    pd = self.correct_prob_density(meas, pd, **kwargs)[0]

                # find best one and add to state table
                new_state = pd.mean
                new_cov = pd.covariance
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

    def _gate_meas(self, meas, means, covs, **kwargs):
        warn('Mesurement gating not yet implemented for SMC-GLMB')
        return meas
