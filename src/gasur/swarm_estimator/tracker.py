"""Implements RFS tracking algorithms.

This module contains the classes and data structures
for RFS tracking related algorithms.
"""
import numpy as np
from numpy.linalg import cholesky, inv
import numpy.random as rnd
import numpy.matlib as matlib
from scipy.optimize import linear_sum_assignment
import matplotlib.pyplot as plt
from matplotlib.patches import Ellipse
import matplotlib.animation as animation
import abc
from copy import deepcopy
from warnings import warn

from gasur.utilities.distributions import GaussianMixture, StudentsTMixture
from gasur.utilities.graphs import k_shortest, murty_m_best
from gasur.utilities.sampling import gibbs
from gncpy.math import log_sum_exp, get_elem_sym_fnc
import gncpy.plotting as pltUtil
import gncpy.filters as gfilts
import gncpy.errors as gerr


class RandomFiniteSetBase(metaclass=abc.ABCMeta):
    """Generic base class for RFS based filters.

    Attributes
    ----------
    filter : gncpy.filters.BayesFilter
        Filter handling dynamics
    prob_detection : float
        Modeled probability an object is detected
    prob_survive : float
        Modeled probability of object survival
    birth_terms : list
        List of terms in the birth model
    clutter_rate : float
        Rate of clutter
    clutter_density : float
        Density of clutter distribution
    inv_chi2_gate : float
        Chi squared threshold for gating the measurements
    save_covs : bool
        Save covariance matrix for each state during state extraction
    debug_plots : bool
        Saves data needed for extra debugging plots
    ospa : numpy array
        Calculated OSPA value for the given truth data. Must be manually updated
        by a function call.
    ospa_localization : numpy array
        Calculated OSPA value for the given truth data. Must be manually updated
        by a function call.
    ospa_cardinality : numpy array
        Calculated OSPA value for the given truth data. Must be manually updated
        by a function call.
    """

    def __init__(self, in_filter=None, prob_detection=1, prob_survive=1,
                 birth_terms=None, clutter_rate=0, clutter_den=0,
                 inv_chi2_gate=0, save_covs=False, debug_plots=False, **kwargs):
        if birth_terms is None:
            birth_terms = []
        self.filter = deepcopy(in_filter)
        self.prob_detection = prob_detection
        self.prob_survive = prob_survive
        self.birth_terms = deepcopy(birth_terms)
        self.clutter_rate = clutter_rate
        if isinstance(clutter_den, np.ndarray):
            clutter_den = clutter_den.item()
        self.clutter_den = clutter_den

        self.inv_chi2_gate = inv_chi2_gate

        self.save_covs = save_covs
        self.debug_plots = debug_plots

        self.ospa = None
        self.ospa_localization = None
        self.ospa_cardinality = None

        self._states = []  # local copy for internal modification
        self._meas_tab = []  # list of lists, one per timestep, inner is all meas at time
        self._covs = []  # local copy for internal modification

        super().__init__(**kwargs)

    @property
    def prob_miss_detection(self):
        """Compliment of :py:attr:`.swarm_estimator.RandomFiniteSetBase.prob_detection`."""
        return 1 - self.prob_detection

    @property
    def prob_death(self):
        """Compliment of :attr:`gasur.swarm_estimator.RandomFinitSetBase.prob_survive`."""
        return 1 - self.prob_survive

    @property
    def num_birth_terms(self):
        """Number of terms in the birth model."""
        return len(self.birth_terms)

    @abc.abstractmethod
    def predict(self, t, **kwargs):
        """Abstract method for the prediction step.

        This must be overridden in the inherited class. It is recommended to
        keep the same structure/order for the arguments for consistency
        between the inherited classes.
        """
        pass

    @abc.abstractmethod
    def correct(self, t, m, **kwargs):
        """Abstract method for the correction step.

        This must be overridden in the inherited class. It is recommended to
        keep the same structure/order for the arguments for consistency
        between the inherited classes.
        """
        pass

    @abc.abstractmethod
    def extract_states(self, **kwargs):
        """Abstract method for extracting states."""
        pass

    @abc.abstractmethod
    def cleanup(self, **kwargs):
        """Abstract method that performs the cleanup step of the filter.

        This must be overridden in the inherited class. It is recommended to
        keep the same structure/order for the arguments for consistency
        between the inherited classes.
        """
        pass

    def _gate_meas(self, meas, means, covs, meas_mat_args={},
                   est_meas_args={}):
        """Gates measurements based on current estimates.

        Notes
        -----
        Gating is performed based on a Gaussian noise model.
        See :cite:`Cox1993_AReviewofStatisticalDataAssociationTechniquesforMotionCorrespondence`
        for details on the chi squared test used.

        Parameters
        ----------
        meas : list
            2d numpy arrrays of each measurement.
        means : list
            2d numpy arrays of each mean.
        covs : list
            2d numpy array of each covariance.
        meas_mat_args : dict, optional
            keyword arguments to pass to the inner filters get measurement
            matrix function. The default is {}.
        est_meas_args : TYPE, optional
            keyword arguments to pass to the inner filters get estimate
            matrix function. The default is {}.

        Returns
        -------
        list
            2d numpy arrays of valid measurements.

        """
        if len(meas) == 0:
            return []
        valid = []
        for (m, p) in zip(means, covs):
            meas_mat = self.filter.get_meas_mat(m, **meas_mat_args)
            est = self.filter.get_est_meas(m, **est_meas_args)
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

    def calculate_ospa(self, truth, c, p):
        """Calculates the OSPA distance between the truth at all timesteps.

        Notes
        -----
        This calculates the Optimal SubPattern Assignment metric for the
        extracted states and the supplied truth point distributions. The
        calculation is based on
        :cite:`Schuhmacher2008_AConsistentMetricforPerformanceEvaluationofMultiObjectFilters`
        with much of the math defined in
        :cite:`Schuhmacher2008_ANewMetricbetweenDistributionsofPointProcesses`.
        A value is calculated for each timestep available in the data.

        Parameters
        ----------
        truth : list
            Each element represents a timestep and is a list of N x 1 numpy array,
            one per true agent in the swarm.
        c : float
            Distance cutoff for considering a point properly assigned. This
            influences how cardinality errors are penalized. For :math:`p = 1`
            it is the penalty given false point estimate.
        p : int
            The power of the distance term. Higher values penalize outliers
            more.
        """
        num_timesteps = len(self._states)
        self.ospa = np.nan * np.ones(num_timesteps)
        self.ospa_localization = np.nan * np.ones(num_timesteps)
        self.ospa_cardinality = np.nan * np.ones(num_timesteps)
        for ii, (x_lst, y_lst) in enumerate(zip(self._states, truth)):
            x_empty = x_lst is None or len(x_lst) == 0
            y_empty = y_lst is None or len(y_lst) == 0

            if x_empty and y_empty:
                self.ospa[ii] = 0
                self.ospa_localization[ii] = 0
                self.ospa_cardinality[ii] = 0
                continue

            if x_empty or y_empty:
                self.ospa[ii] = 0
                self.ospa_localization[ii] = 0
                self.ospa_cardinality[ii] = c
                continue

            # create row matrices of data
            x = np.stack([vec.flatten() for vec in x_lst])
            y = np.stack([vec.flatten() for vec in y_lst])

            n = x.shape[0]
            m = y.shape[0]

            x_mat = np.tile(x, (m, 1))
            # set y_mat to repeat each value of y n times in a row
            y_mat = np.tile(y, (1, x.shape[0])).reshape((n * m, y.shape[1]))

            # get distances and set cutoff
            dists = np.sqrt(np.sum((x_mat - y_mat)**2, axis=1)).reshape((m, n))
            dists = np.minimum(dists, c)**p

            # use hungarian to find minimum distances for getting total cost
            row_ind, col_ind = linear_sum_assignment(dists)
            cost = dists[row_ind, col_ind].sum()

            inv_max_card = 1 / np.max([n, m])
            card_diff = np.abs(n - m)
            inv_p = 1 / p
            c_p = c**p
            self.ospa[ii] = (inv_max_card * (c_p * card_diff + cost))**inv_p
            self.ospa_localization[ii] = (inv_max_card * cost)**inv_p
            self.ospa_cardinality[ii] = (inv_max_card * c_p * card_diff)**inv_p

    def plot_ospa_history(self, time_units='index', time=None, **kwargs):
        """Plots the OSPA history.

        This requires that the OSPA has been calcualted by the approriate
        function first.

        Parameters
        ----------
        time_units : string, optional
            Text representing the units of time in the plot. The default is
            'index'.
        time : numpy array, optional
            Vector to use for the x-axis of the plot. If none is given then
            vector indices are used. The default is None.
        **kwargs : dict
            Additional plotting options for :meth:`gncpy.plotting.init_plotting_opts`
            function. Values implemented here are `f_hndl`, and any values
            relating to title/axis text formatting.

        Returns
        -------
        fig : matplotlib figure
            Figure object the data was plotted on.
        """
        if self.ospa is None:
            warn('OSPA must be calculated before plotting')
            return

        opts = pltUtil.init_plotting_opts(**kwargs)
        fig = opts['f_hndl']

        if fig is None:
            fig = plt.figure()
            fig.add_subplot(1, 1, 1)

        if time is None:
            time = np.arange(self.ospa.size, dtype=int)

        fig.axes[0].grid(True)
        fig.axes[0].plot(time, self.ospa)

        pltUtil.set_title_label(fig, 0, opts, ttl="OSPA Metric",
                                x_lbl='Time ({})'.format(time_units),
                                y_lbl="OSPA")
        fig.tight_layout()

        return fig


class ProbabilityHypothesisDensity(RandomFiniteSetBase):
    """Implements the Probability Hypothesis Density filter.

    The kwargs in the constructor are passed through to the parent constructor.

    Notes
    -----
    The filter implementation is based on :cite:`Vo2006_TheGaussianMixtureProbabilityHypothesisDensityFilter`

    Attributes
    ----------
    gating_on : bool
        flag indicating if measurement gating should be performed. The
        default is False.
    inv_chi2_gate : float
        threshold for the chi squared test in the measurement gating. The
        default is 0.
    extract_threshold : float
        threshold for extracting the state. The default is 0.5.
    prune_threshold : float
        threshold for removing hypotheses. The default is 10**-5.
    merge_threshold : float
        threshold for merging hypotheses. The default is 4.
    max_gauss : int
        max number of gaussians to use. The default is 100.

    """

    def __init__(self, gating_on=False, inv_chi2_gate=0, extract_threshold=0.5,
                 prune_threshold=10**-5, merge_threshold=4, max_gauss=100,
                 **kwargs):
        self.gating_on = gating_on
        self.inv_chi2_gate = inv_chi2_gate
        self.extract_threshold = extract_threshold
        self.prune_threshold = prune_threshold
        self.merge_threshold = merge_threshold
        self.max_gauss = max_gauss

        self._gaussMix = GaussianMixture()

        super().__init__(**kwargs)

    @property
    def states(self):
        """Read only list of extracted states.

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
        """Read only list of extracted covariances.

        This is a list with 1 element per timestep, and each element is a list
        of the best covariances extracted at that timestep. The order of each
        element corresponds to the state order.

        Warns
        -----
            RuntimeWarning
                If the class is not saving the covariances, and returns an
                empty list
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
        """Read only cardinality of the RFS."""
        if len(self._states) == 0:
            return 0
        else:
            return len(self._states[-1])

    def predict(self, timestep, filt_args={}):
        """Prediction step of the PHD filter.

        This predicts new hypothesis, and propogates them to the next time
        step. It also updates the cardinality distribution. Because this calls
        the inner filter's predict function, the keyword arguments must contain
        any information needed by that function.


        Parameters
        ----------
        timestep: float
            current timestep
        filt_args : dict, optional
            Passed to the inner filter. The default is {}.

        Returns
        -------
        None.

        """
        self._gaussMix = self._predict_prob_density(timestep, self._gaussMix,
                                                    filt_args)

        for gm in self.birth_terms:
            self._gaussMix.weights.extend(gm.weights)
            self._gaussMix.means.extend(gm.means)
            self._gaussMix.covariances.extend(gm.covariances)

    def _predict_prob_density(self, timestep, probDensity, filt_args):
        """Predicts the probability density.

        Loops over all elements in a probability distribution and performs
        the filter prediction.

        Parameters
        ----------
        timestep: float
            current timestep
        probDensity : :class:`gasur.utilities.distributions.GaussianMixture`
            Probability density to perform prediction on.
        filt_args : dict
            Passed directly to the inner filter.

        Returns
        -------
        gm : :class:`gasur.utilities.distributions.GaussianMixture`
            predicted Gaussian mixture.

        """
        gm_tup = zip(probDensity.means,
                     probDensity.covariances)
        gm = GaussianMixture()
        gm.weights = [self.prob_survive * x for x in probDensity.weights.copy()]
        for ii, (m, P) in enumerate(gm_tup):
            self.filter.cov = P
            n_mean = self.filter.predict(timestep, m, **filt_args)
            gm.covariances.append(self.filter.cov.copy())
            gm.means.append(n_mean)

        return gm

    def correct(self, timestep, meas_in, meas_mat_args={}, est_meas_args={},
                filt_args={}):
        """Correction step of the PHD filter.

        This corrects the hypotheses based on the measurements and gates the
        measurements according to the class settings. It also updates the
        cardinality distribution.


        Parameters
        ----------
        timestep: float
            current timestep
        meas_in : list
            2d numpy arrays representing a measurement.
        meas_mat_args : dict, optional
            keyword arguments to pass to the inner filters get measurement
            matrix function. Only used if gating is on. The default is {}.
        est_meas_args : TYPE, optional
            keyword arguments to pass to the inner filters estimate
            measurements function. Only used if gating is on. The default is {}.
        filt_args : dict, optional
            keyword arguments to pass to the inner filters correct function.
            The default is {}.

        .. todo::
            Fix the measurement gating

        Returns
        -------
        None.

        """
        meas = deepcopy(meas_in)

        if self.gating_on:
            meas = self._gate_meas(meas, self._gaussMix.means,
                                   self._gaussMix.covariances, meas_mat_args,
                                   est_meas_args)

        self._meas_tab.append(meas)

        gmix = deepcopy(self._gaussMix)
        gmix.weights = [self.prob_miss_detection * x for x in gmix.weights]
        gm = self._correct_prob_density(timestep, meas, self._gaussMix, filt_args)

        gm.weights.extend(gmix.weights)
        self._gaussMix.weights = gm.weights.copy()
        gm.means.extend(gmix.means)
        self._gaussMix.means = gm.means.copy()
        gm.covariances.extend(gmix.covariances)
        self._gaussMix.covariances = gm.covariances.copy()

    def _correct_prob_density(self, timestep, meas, probDensity, filt_args):
        """Corrects the probability densities.

        Loops over all elements in a probability distribution and preforms
        the filter correction.

        Parameters
        ----------
        meas : list
            2d numpy arrays of each measurement.
        probDensity : :py:class:`gasur.utilities.distributions.GaussianMixture`
            probability density to run correction on.
        filt_args : dict
            arguements to pass to the inner filter correct function.

        Returns
        -------
        gm : :py:class:`gasur.utilities.distributions.GaussianMixture`
            corrected probability density.

        """
        gm = GaussianMixture()
        det_weights = [self.prob_detection * x for x in probDensity.weights]
        for z in meas:
            w_lst = []
            for jj in range(0, len(probDensity.means)):
                self.filter.cov = probDensity.covariances[jj]
                state = probDensity.means[jj]
                (mean, qz) = self.filter.correct(timestep, z, state, **filt_args)
                cov = self.filter.cov
                w = qz * det_weights[jj]
                gm.means.append(mean)
                gm.covariances.append(cov)
                w_lst.append(w)
            gm.weights.extend([x / (self.clutter_rate * self.clutter_den
                               + sum(w_lst)) for x in w_lst])

        return gm

    def _prune(self):
        """Removes hypotheses below a threshold.

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

    def _merge(self):
        """Merges nearby hypotheses."""
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

    def _cap(self):
        """Removes least likely hypotheses until a maximum number is reached.

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

    def extract_states(self):
        """Extracts the best state estimates.

        This extracts the best states from the distribution. It should be
        called once per time step after the correction function.
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

    def cleanup(self, enable_prune=True, enable_cap=True, enable_merge=True,
                enable_extract=True):
        """Performs the cleanup step of the filter.

        This can prune, cap, and extract states. It must be called once per
        timestep. If this is called with `enable_extract` set to true then
        the extract states method does not need to be called separately. It is
        recommended to call this function instead of
        :meth:`gasur.swarm_estimator.tracker.GeneralizedLabeledMultiBernoulli.extract_states`
        directly.

        Parameters
        ----------
        enable_prune : bool, optional
            Flag indicating if prunning should be performed. The default is True.
        enable_cap : bool, optional
            Flag indicating if capping should be performed. The default is True.
        enable_merge : bool, optional
            Flag indicating if merging should be performed. The default is True.
        enable_extract : bool, optional
            Flag indicating if state extraction should be performed. The default is True.

        Returns
        -------
        None.

        """
        if enable_prune:
            self._prune()

        if enable_merge:
            self._merge()

        if enable_cap:
            self._cap()

        if enable_extract:
            self.extract_states()

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
        """Plots the best estimate for the states.

        This assumes that the states have been extracted. It's designed to plot
        two of the state variables (typically x/y position). The error ellipses
        are calculated according to :cite:`Hoover1984_AlgorithmsforConfidenceCirclesandEllipses`

        Keyword arguments are processed with
        :meth:`gncpy.plotting.init_plotting_opts`. This function
        implements

            - f_hndl
            - true_states
            - sig_bnd
            - rng
            - meas_inds
            - lgnd_loc
            - marker

        Parameters
        ----------
        plt_inds : list
            List of indices in the state vector to plot
        state_lbl : string
            Value to appear in legend for the states. Only appears if the
            legend is shown

        Returns
        -------
        Matplotlib figure
            Instance of the matplotlib figure used
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
            color = (128 / 255, 128 / 255, 128 / 255)
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
        """Creates an animated plot of the states.

        Parameters
        ----------
        plt_inds : list
            indices of the state vector to plot.
        state_lbl : string, optional
            label for the states. The default is 'States'.
        state_color : tuple, optional
            3-tuple for rgb value. The default is None.
        interval : int, optional
            interval of the animation in ms. The default is 250.
        repeat : bool, optional
            flag indicating if the animation loops. The default is True.
        repeat_delay : int, optional
            delay between loops in ms. The default is 1000.
        save_path : string, optional
            file path and name to save the gif, does not save if not given.
            The default is None.
        **kwargs : dict, optional
            Standard plotting options for
            :meth:`gncpy.plotting.init_plotting_opts`. This function
            implements

                - f_hndl
                - sig_bnd
                - rng
                - meas_inds
                - lgnd_loc
                - marker

        Returns
        -------
        anim :
            handle to the animation.

        """
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
            m_color = (128 / 255, 128 / 255, 128 / 255)

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
    """Implements the Cardinalized Probability Hypothesis Density filter.

    The kwargs in the constructor are passed through to the parent constructor.

    Notes
    -----
    The filter implementation is based on
    :cite:`Vo2006_TheCardinalizedProbabilityHypothesisDensityFilterforLinearGaussianMultiTargetModels`
    and :cite:`Vo2007_AnalyticImplementationsoftheCardinalizedProbabilityHypothesisDensityFilter`.

    Attributes
    ----------
    agents_per_state : list, optional
        number of agents per state. The default is [].
    """

    def __init__(self, agents_per_state=None, max_expected_card=10, **kwargs):
        if agents_per_state is None:
            agents_per_state = []
        self.agents_per_state = agents_per_state
        self._max_expected_card = max_expected_card

        self._card_dist = np.zeros(self.max_expected_card + 1)  # local copy for internal modification
        self._card_dist[0] = 1
        self._card_time_hist = []  # local copy for internal modification
        self._n_states_per_time = []

        super().__init__(**kwargs)

    @property
    def max_expected_card(self):
        """Maximum expected cardinality. The default is 10."""
        return self._max_expected_card

    @max_expected_card.setter
    def max_expected_card(self, x):
        self._card_dist = np.zeros(x + 1)
        self._card_dist[0] = 1
        self._max_expected_card = x

    @property
    def cardinality(self):
        """Cardinality of the RFS."""
        return np.argmax(self._card_dist)

    def predict(self, timestep, **kwargs):
        """Prediction step of the CPHD filter.

        This predicts new hypothesis, and propogates them to the next time
        step. It also updates the cardinality distribution.


        Parameters
        ----------
        timestep: float
            current timestep
        **kwargs : dict, optional
            See :meth:gasur.swarm_estimator.tracker.ProbabilityHypothesisDensity.predict`
            for the available arguments.

        Returns
        -------
        None.

        """
        super().predict(timestep, **kwargs)

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
        self._card_dist = (cdn_predict / np.sum(cdn_predict)).copy()

    def correct(self, timestep, meas_in, meas_mat_args={}, est_meas_args={},
                filt_args={}):
        """Correction step of the CPHD filter.

        This corrects the hypotheses based on the measurements and gates the
        measurements according to the class settings. It also updates the
        cardinality distribution.


        Parameters
        ----------
        timestep: float
            current timestep
        meas_in : list
            2d numpy arrays representing a measurement.
        meas_mat_args : dict, optional
            keyword arguments to pass to the inner filters get measurement
            matrix function. Only used if gating is on. The default is {}.
        est_meas_args : TYPE, optional
            keyword arguments to pass to the inner filters estimate
            measurements function. Only used if gating is on. The default is {}.
        filt_args : TYPE, optional
            keyword arguments to pass to the inner filters correct function.
            The default is {}.

        Returns
        -------
        None.

        """
        meas = deepcopy(meas_in)

        if self.gating_on:
            meas = self._gate_meas(meas, self._gaussMix.means,
                                   self._gaussMix.covariances, meas_mat_args,
                                   est_meas_args)

        self._meas_tab.append(meas)

        gmix = deepcopy(self._gaussMix)  # predicted gm

        self._gaussMix = self._correct_prob_density(timestep, meas, gmix, filt_args)

    def _correct_prob_density(self, timestep, meas, probDensity, filt_args):
        """Helper function for correction step.

        Loops over all elements in a probability distribution and preforms
        the filter correction.
        """
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

                (mean, qz) = self.filter.correct(timestep, meas[z_ind], state,
                                                 **filt_args)
                cov = self.filter.cov
                qz_temp[p_ind, z_ind] = qz
                mean_temp[z_ind, :, p_ind] = np.ndarray.flatten(mean)
                cov_temp[[p_ind], :, :] = cov

        xivals = np.zeros(zlen)
        pdc = self.prob_detection / self.clutter_den
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

    def extract_states(self):
        """Extracts the best state estimates.

        This extracts the best states from the distribution. It should be
        called once per time step after the correction function.
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
        """Plots the current cardinality distribution.

        This assumes that the cardinality distribution has been calculated by
        the class.

        Parameters
        ----------
        **kwargs : dict, optional
            Keyword arguments are processed with
            :meth:`gncpy.plotting.init_plotting_opts`. This function
            implements

                - f_hndl

        Returns
        -------
        Matplotlib figure
            Instance of the matplotlib figure used

        Raises
        ------
        RuntimeWarning
            If the cardinality distribution is empty.
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

    def plot_card_history(self, true_card=None, **kwargs):
        """Plots the current cardinality time history.

        This assumes that the cardinality distribution has been calculated by
        the class.

        Parameters
        ----------
        true_card : array like
            List of the true cardinality at each time
        **kwargs : dict, optional
            Keyword arguments are processed with
            :meth:`gncpy.plotting.init_plotting_opts`. This function
            implements

                - f_hndl
                - sig_bnd
                - time_vec
                - lgnd_loc

        Returns
        -------
        Matplotlib figure
            Instance of the matplotlib figure used
        """
        opts = pltUtil.init_plotting_opts(**kwargs)
        f_hndl = opts['f_hndl']
        sig_bnd = opts['sig_bnd']
        time_vec = opts['time_vec']
        lgnd_loc = opts['lgnd_loc']

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
        """Plots the number of states per timestep.

        This is a debug plot for if there are 0 weights in the GM but the
        cardinality is not reached. Debug plots must be turned on prior to
        running the filter.


        Parameters
        ----------
        **kwargs : dict, optional
            Keyword arguments are processed with
            :meth:`gncpy.plotting.init_plotting_opts`. This function
            implements

                - f_hndl
                - lgnd_loc

        Returns
        -------
        f_hndl : matplotlib figure
            handle to the current figure.

        """
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
    """Delta-Generalized Labeled Multi-Bernoulli filter.

    Notes
    -----
    This is based on :cite:`Vo2013_LabeledRandomFiniteSetsandMultiObjectConjugatePriors`
    and :cite:`Vo2014_LabeledRandomFiniteSetsandtheBayesMultiTargetTrackingFilter`
    It does not account for agents spawned from existing tracks, only agents
    birthed from the given birth model.

    Attributes
    ----------
    req_births : int
        Number of requested birth hypotheses
    req_surv : int
        Number of requested surviving hypotheses
    req_upd : int
        Number of requested updated hypotheses
    gating_on : bool
        Determines if measurements are gated
    birth_terms :list
        List of tuples where the first element is a
        :py:class:`gncpy.distributions.GaussianMixture` and
        the second is the birth probability for that term
    prune_threshold : float
        Minimum association probability to keep when pruning
    max_hyps : int
        Maximum number of hypotheses to keep when capping
    decimal_places : int
        Number of decimal places to keep in label. The default is 2.
    """

    class _TabEntry:
        def __init__(self):
            self.label = ()  # time step born, index of birth model born from
            self.probDensity = None  # must be a distribution class
            self.meas_assoc_hist = []  # list indices into measurement list per time step

            """ linear index corresponding to timestep, manually updated. Used
            to index things since timestep in label can have decimals."""
            self.time_index = None

    class _HypothesisHelper:
        def __init__(self):
            self.assoc_prob = 0
            self.track_set = []  # indices in lookup table

        @property
        def num_tracks(self):
            return len(self.track_set)

    class _ExtractHistHelper:
        def __init__(self):
            self.label = ()
            self.meas_ind_hist = []
            self.b_time_index = None

    def __init__(self, req_births=None, req_surv=None, req_upd=None,
                 gating_on=False, prune_threshold=10**-15, max_hyps=3000,
                 decimal_places=2, **kwargs):
        self.req_births = req_births
        self.req_surv = req_surv
        self.req_upd = req_upd
        self.gating_on = gating_on
        self.prune_threshold = prune_threshold
        self.max_hyps = max_hyps
        self.decimal_places = decimal_places

        self._track_tab = []  # list of all possible tracks
        self._labels = []  # local copy for internal modification
        self._extractable_hists = []
        self._pred_timesteps = []
        self._cor_timesteps = []

        hyp0 = self._HypothesisHelper()
        hyp0.assoc_prob = 1
        hyp0.track_set = []
        self._hypotheses = [hyp0]  # list of _HypothesisHelper objects

        self._card_dist = []  # probability of having index # as cardinality

        """ linear index corresponding to timestep, manually updated. Used
            to index things since timestep in label can have decimals. Must
            be updated once per time step."""
        self._time_index_cntr = 0

        super().__init__(**kwargs)

    @property
    def states(self):
        """Read only list of extracted states.

        This is a list with 1 element per timestep, and each element is a list
        of the best states extracted at that timestep. The order of each
        element corresponds to the label order.
        """
        return self._states

    @property
    def labels(self):
        """Read only list of extracted labels.

        This is a list with 1 element per timestep, and each element is a list
        of the best labels extracted at that timestep. The order of each
        element corresponds to the state order.
        """
        return self._labels

    @property
    def covariances(self):
        """Read only list of extracted covariances.

        This is a list with 1 element per timestep, and each element is a list
        of the best covariances extracted at that timestep. The order of each
        element corresponds to the state order.

        Raises
        ------
        RuntimeWarning
            If the class is not saving the covariances, and returns an empty list.
        """
        if not self.save_covs:
            raise RuntimeWarning("Not saving covariances")
            return []
        return self._covs

    @property
    def cardinality(self):
        """Cardinality estimate."""
        return np.argmax(self._card_dist)

    def _gen_birth_tab(self, timestep):
        log_cost = []
        birth_tab = []
        for ii, (distrib, p) in enumerate(self.birth_terms):
            cost = p / (1 - p)
            log_cost.append(-np.log(cost))
            entry = self._TabEntry()
            # entry.probDensity = deepcopy(distrib)
            entry.probDensity = distrib
            entry.label = (round(timestep, self.decimal_places), ii)
            entry.time_index = self._time_index_cntr
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

    def _predict_prob_density(self, timestep, probDensity, filt_args):
        """Loops over probability distribution and preforms prediction."""
        gm_tup = zip(probDensity.means,
                     probDensity.covariances)
        gm = GaussianMixture()
        gm.weights = probDensity.weights.copy()
        for ii, (m, P) in enumerate(gm_tup):
            self.filter.cov = P
            n_mean = self.filter.predict(timestep, m, **filt_args)
            gm.covariances.append(self.filter.cov.copy())
            gm.means.append(n_mean)

        return gm

    def _predict_track_tab_entry(self, tab, timestep, filt_args):
        """Updates table entries probability density."""
        newTab = deepcopy(tab)
        newTab.probDensity = self._predict_prob_density(timestep, tab.probDensity, filt_args)
        return newTab

    def _gen_surv_tab(self, timestep, filt_args):
        surv_tab = []
        for (ii, track) in enumerate(self._track_tab):
            entry = self._predict_track_tab_entry(track, timestep, filt_args)

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

    def _calc_avg_prob_surv_death(self):
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

    def _calc_card_dist(self, hyp_lst):
        """Calucaltes the cardinality distribution."""
        if len(hyp_lst) == 0:
            return [1, ]

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
                sorted_inds = hyp.track_set.copy()
                sorted_inds.sort()
                lst = [int(x) for x in sorted_inds]
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

    def predict(self, timestep, filt_args={}):
        """Prediction step of the GLMB filter.

        This predicts new hypothesis, and propogates them to the next time
        step. It also updates the cardinality distribution.

        Parameters
        ----------
        timestep: float
            Current timestep.
        filt_args : dict, optional
            Passed to the inner filter. The default is {}.

        Returns
        -------
        None.
        """
        self._pred_timesteps.append(timestep)

        # Find cost for each birth track, and setup lookup table
        birth_tab, log_cost = self._gen_birth_tab(timestep)

        # get K best hypothesis, and their index in the lookup table
        (paths, hyp_costs) = k_shortest(np.array(log_cost), self.req_births)

        # calculate association probabilities for birth hypothesis
        birth_hyps = self._gen_birth_hyps(paths, hyp_costs)

        # Init and propagate surviving track table
        surv_tab = self._gen_surv_tab(timestep, filt_args)

        # Calculation for average survival/death probabilities
        (avg_prob_survive,
         avg_prob_death) = self._calc_avg_prob_surv_death()

        # loop over postierior components
        surv_hyps = self._gen_surv_hyps(avg_prob_survive, avg_prob_death)

        self._card_dist = self._calc_card_dist(surv_hyps)

        # Get  predicted hypothesis by convolution
        self._track_tab = birth_tab + surv_tab
        self._set_pred_hyps(birth_tab, birth_hyps, surv_hyps)

        self._clean_predictions()

    def _correct_prob_density(self, timestep, meas, probDensity, filt_args):
        """Loops over a probability distribution and preforms correction."""
        gm = GaussianMixture()
        for jj in range(0, len(probDensity.means)):
            self.filter.cov = probDensity.covariances[jj]
            state = probDensity.means[jj]
            (mean, qz) = self.filter.correct(timestep, meas, state, **filt_args)
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

    def _correct_track_tab_entry(self, meas, tab, timestep, filt_args):
        newTab = deepcopy(tab)
        newTab.probDensity, cost = self._correct_prob_density(timestep, meas,
                                                              tab.probDensity,
                                                              filt_args)

        return newTab, cost

    def _gen_cor_tab(self, num_meas, meas, timestep, filt_args):
        num_pred = len(self._track_tab)
        up_tab = [None] * (num_meas + 1) * num_pred

        for ii, track in enumerate(self._track_tab):
            up_tab[ii] = deepcopy(track)
            up_tab[ii].meas_assoc_hist.append(None)

        # measurement updated tracks
        all_cost_m = np.zeros((num_pred, num_meas))
        for emm, z in enumerate(meas):
            for ii, ent in enumerate(self._track_tab):
                s_to_ii = num_pred * emm + ii + num_pred
                (up_tab[s_to_ii], cost) = \
                    self._correct_track_tab_entry(z, ent, timestep, filt_args)

                # update association history with current measurement index
                up_tab[s_to_ii].meas_assoc_hist.append(emm)
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
                    pd = np.array([avg_prob_detect[ii] for ii in p_hyp.track_set])
                    pmd = np.array([avg_prob_miss_detect[ii] for ii in p_hyp.track_set])
                    ratio = pd / pmd

                    ratio = ratio.reshape((ratio.size, 1))
                    ratio = np.tile(ratio, (1, num_meas))

                    cost_m = np.zeros(all_cost_m[p_hyp.track_set, :].shape)
                    for ii, ts in enumerate(p_hyp.track_set):
                        cost_m[ii, :] = ratio[ii] * all_cost_m[ts, :] / clutter

                    max_row_inds, max_col_inds = np.where(cost_m >= np.inf)
                    if max_row_inds.size > 0:
                        cost_m[max_row_inds, max_col_inds] = np.finfo(float).max

                    min_row_inds, min_col_inds = np.where(cost_m <= 0.)
                    if min_row_inds.size > 0:
                        cost_m[min_row_inds, min_col_inds] = 1

                    neg_log = -np.log(cost_m)
                    # if max_row_inds.size > 0:
                    #     neg_log[max_row_inds, max_col_inds] = -np.inf
                    if min_row_inds.size > 0:
                        neg_log[min_row_inds, min_col_inds] = np.inf

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
                        # lst1 = [num_pred * x for x in a]
                        # lst2 = p_hyp.track_set.copy()
                        # if len(lst1) != len(lst2):
                        #     raise RuntimeWarning('Lists not the same length')
                        # new_hyp.track_set = [sum(x) for x in zip(lst1, lst2)]
                        new_hyp.track_set = list(np.array(p_hyp.track_set) + num_pred * a)
                        up_hyps.append(new_hyp)

        lse = log_sum_exp([x.assoc_prob for x in up_hyps])
        for ii in range(0, len(up_hyps)):
            up_hyps[ii].assoc_prob = np.exp(up_hyps[ii].assoc_prob - lse)

        return up_hyps

    def _calc_avg_prob_det_mdet(self):
        avg_prob_detect = self.prob_detection * np.ones(len(self._track_tab))
        avg_prob_miss_detect = 1 - avg_prob_detect

        return avg_prob_detect, avg_prob_miss_detect

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

    def correct(self, timestep, meas, filt_args={}):
        """Correction step of the GLMB filter.

        Notes
        -----
        This corrects the hypotheses based on the measurements and gates the
        measurements according to the class settings. It also updates the
        cardinality distribution.

        Parameters
        ----------
        timestep: float
            Current timestep.
        meas_in : list
            List of Nm x 1 numpy arrays each representing a measuremnt.
        filt_args : dict, optional
            keyword arguments to pass to the inner filters correct function.
            The default is {}.

        .. todo::
            Fix the measurement gating

        Returns
        -------
        None
        """
        self._cor_timesteps.append(timestep)

        # gate measurements by tracks
        if self.gating_on:
            means = []
            covs = []
            for ent in self._track_tab:
                means.extend(ent.probDensity.means)
                covs.extend(ent.probDensity.covariances)
            meas = self._gate_meas(meas, means, covs)

        self._meas_tab.append(deepcopy(meas))
        num_meas = len(meas)

        # missed detection tracks
        cor_tab, all_cost_m = self._gen_cor_tab(num_meas, meas, timestep, filt_args)

        # Calculation for average detection/missed probabilities
        avg_prob_det, avg_prob_mdet = self._calc_avg_prob_det_mdet()

        # component updates
        cor_hyps = self._gen_cor_hyps(num_meas, avg_prob_det, avg_prob_mdet,
                                      all_cost_m)

        # save values and cleanup
        self._track_tab = cor_tab
        self._hypotheses = cor_hyps
        self._card_dist = self._calc_card_dist(self._hypotheses)
        self._clean_updates()

    def _update_extract_hist(self, idx_cmp):
        used_meas_inds = [[]] * self._time_index_cntr
        used_labels = []
        new_extract_hists = [None] * len(self._hypotheses[idx_cmp].track_set)
        for ii, ptr in enumerate(self._hypotheses[idx_cmp].track_set):
            new_extract_hists[ii] = self._ExtractHistHelper()
            new_extract_hists[ii].label = self._track_tab[ptr].label
            new_extract_hists[ii].meas_ind_hist = self._track_tab[ptr].meas_assoc_hist.copy()
            new_extract_hists[ii].b_time_index = self._track_tab[ptr].time_index

            used_labels.append(self._track_tab[ptr].label)

            for t_inds_after_b, meas_ind in enumerate(self._track_tab[ptr].meas_assoc_hist):
                tt = self._track_tab[ptr].time_index + t_inds_after_b
                if meas_ind is not None and meas_ind not in used_meas_inds[tt]:
                    used_meas_inds[tt].append(meas_ind)

        good_inds = []
        for ii, existing in enumerate(self._extractable_hists):
            used = existing.label in used_labels
            if used:
                continue

            for t_inds_after_b, meas_ind in enumerate(existing.meas_ind_hist):
                tt = existing.b_time_index + t_inds_after_b
                if meas_ind in used_meas_inds[tt]:
                    used = True
                    break

            if not used:
                good_inds.append(ii)

        self._extractable_hists = [self._extractable_hists[ii] for ii in good_inds]
        self._extractable_hists.extend(new_extract_hists)

    def _extract_helper(self, pd):
        idx_trk = np.argmax(pd.weights)
        new_state = pd.means[idx_trk]
        new_cov = pd.covariances[idx_trk]

        return new_state, new_cov

    def extract_states(self, pred_args={}, cor_args={}, update=True,
                       calc_states=True):
        """Extracts the best state estimates.

        This extracts the best states from the distribution. It should be
        called once per time step after the correction function. This calls
        both the inner filters predict and correct functions so the keyword
        arguments must contain any additional variables needed by those
        functions.

        Parameters
        ----------
        pred_args : dict, optional
            Additional arguments to pass to the inner filters prediction
            function. The default is {}.
        cor_args : dict, optional
            Additional arguments to pass to the inner filters correction
            function. The default is {}.
        update : bool, optional
            Flag indicating if the label history should be updated. This should
            be done once per timestep and can be disabled if calculating states
            after the final timestep. The default is True.
        calc_states : bool, optional
            Flag indicating if the states should be calculated based on the
            label history. This only needs to be done before the states are used.
            It can simply be called once after the end of the simulation. The
            default is true.

        .. todo::
            Improve the history tracking so it is not as convoluted and does
            a better comparison for labels to protect against numerical issues
            with floats in the timestamp.

        Returns
        -------
        idx_cmp : int
            Index of the hypothesis table used when extracting states.
        """
        card = np.argmax(self._card_dist)
        tracks_per_hyp = np.array([x.num_tracks for x in self._hypotheses])
        weight_per_hyp = np.array([x.assoc_prob for x in self._hypotheses])

        self._states = [[]] * self._time_index_cntr
        self._labels = [[]] * self._time_index_cntr
        if self.save_covs:
            self._covs = [[]] * self._time_index_cntr

        if len(tracks_per_hyp) == 0:
            return None

        idx_cmp = np.argmax(weight_per_hyp * (tracks_per_hyp == card))
        if update:
            self._update_extract_hist(idx_cmp)

        if calc_states:
            for existing in self._extractable_hists:
                b_time, b_idx = existing.label
                pd = deepcopy(self.birth_terms[b_idx][0])

                for t_inds_after_b, meas_ind in enumerate(existing.meas_ind_hist):
                    tt = existing.b_time_index + t_inds_after_b
                    timestep = self._pred_timesteps[tt]
                    pd = self._predict_prob_density(timestep, pd, pred_args)

                    if meas_ind is not None:
                        meas = self._meas_tab[tt][meas_ind].copy()
                        pd = self._correct_prob_density(timestep, meas, pd, cor_args)[0]

                    new_state, new_cov = self._extract_helper(pd)

                    # may happen if filter fails so stop trying to extract track
                    if new_state is None:
                        break

                    if len(self._labels[tt]) == 0:
                        self._states[tt] = [new_state]
                        self._labels[tt] = [existing.label]
                        if self.save_covs:
                            self._covs[tt] = [new_cov]
                    else:
                        self._states[tt].append(new_state)
                        self._labels[tt].append(existing.label)
                        if self.save_covs:
                            self._covs[tt].append(new_cov)

        if not update and not calc_states:
            warn('Extracting states performed no actions')

        return idx_cmp

    def extract_most_prob_states(self, thresh, pred_args={}, cor_args={}):
        """Extracts the most probable hypotheses up to a threshold.

        Parameters
        ----------
        thresh : float
            Minimum association probability to extract.
        pred_args : dict, optional
            Additional arguments to pass to the inner filters prediction
            function. The default is {}.
        cor_args : dict, optional
            Additional arguments to pass to the inner filters correction
            function. The default is {}.

        Returns
        -------
        state_sets : list
            Each element is the state list from the normal
            :meth:`gasur.swarm_estimator.tracker.GeneralizedLabeledMultiBernoulli.extract_states`.
        label_sets : list
            Each element is the label list from the normal
            :meth:`gasur.swarm_estimator.tracker.GeneralizedLabeledMultiBernoulli.extract_states`
        cov_sets : list
            Each element is the covariance list from the normal
            :meth:`gasur.swarm_estimator.tracker.GeneralizedLabeledMultiBernoulli.extract_states`
            if the covariances are saved.
        probs : list
            Each element is the association probability for the extracted states.
        """
        loc_self = deepcopy(self)
        state_sets = []
        cov_sets = []
        label_sets = []
        probs = []

        idx = loc_self.extract_states(pred_args=pred_args, cor_args=cor_args)
        if idx is None:
            return (state_sets, label_sets, cov_sets, probs)

        state_sets.append(loc_self.states.copy())
        label_sets.append(loc_self.labels.copy())
        if loc_self.save_covs:
            cov_sets.append(loc_self.covariances.copy())
        probs.append(loc_self._hypotheses[idx].assoc_prob)
        loc_self._hypotheses[idx].assoc_prob = 0
        while True:
            idx = loc_self.extract_states(pred_args=pred_args, cor_args=cor_args)
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

    def _prune(self):
        """Removes hypotheses below a threshold.

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
        self._card_dist = self._calc_card_dist(self._hypotheses)

    def _cap(self):
        """Removes least likely hypotheses until a maximum number is reached.

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
            self._card_dist = self._calc_card_dist(self._hypotheses)

    def cleanup(self, enable_prune=True, enable_cap=True, enable_extract=True,
                extract_kwargs={}):
        """Performs the cleanup step of the filter.

        This can prune, cap, and extract states. It must be called once per
        timestep, even if all three functions are disabled. This is to ensure
        that internal counters for tracking linear timestep indices are properly
        incremented. If this is called with `enable_extract` set to true then
        the extract states method does not need to be called separately. It is
        recommended to call this function instead of
        :meth:`gasur.swarm_estimator.tracker.GeneralizedLabeledMultiBernoulli.extract_states`
        directly.

        Parameters
        ----------
        enable_prune : bool, optional
            Flag indicating if prunning should be performed. The default is True.
        enable_cap : bool, optional
            Flag indicating if capping should be performed. The default is True.
        enable_extract : bool, optional
            Flag indicating if state extraction should be performed. The default is True.
        pred_args : dict, optional
            Additional arguments to pass to the inner filter's prediction
            function. The default is {}. Only used if extracting states.
        cor_args : dict, optional
            Additional arguments to pass to the inner filter's correction
            function. The default is {}. Only used if extracting states.

        Returns
        -------
        None.

        """
        self._time_index_cntr += 1

        if enable_prune:
            self._prune()

        if enable_cap:
            self._cap()

        if enable_extract:
            self.extract_states(**extract_kwargs)

    def plot_states_labels(self, plt_inds, **kwargs):
        """Plots the best estimate for the states and labels.

        This assumes that the states have been extracted. It's designed to plot
        two of the state variables (typically x/y position). The error ellipses
        are calculated according to :cite:`Hoover1984_AlgorithmsforConfidenceCirclesandEllipses`

        Keywrod arguments are processed with
        :meth:`gncpy.plotting.init_plotting_opts`. This function
        implements

            - f_hndl
            - true_states
            - sig_bnd
            - rng
            - meas_inds
            - lgnd_loc

        Parameters
        ----------
        plt_inds : list
            List of indices in the state vector to plot

        Returns
        -------
        Matplotlib figure
            Instance of the matplotlib figure used
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

        cmap = pltUtil.get_cmap(len(u_lbls))

        # get array of all state values for each label
        added_sig_lbl = False
        added_true_lbl = False
        added_state_lbl = False
        added_meas_lbl = False
        for c_idx, lbl in enumerate(u_lbls):
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
            color = cmap(c_idx)

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

            settings = {'color': color, 'markeredgecolor': 'k', 'marker': '.'}
            if not added_state_lbl:
                settings['label'] = 'States'
                # f_hndl.axes[0].scatter(x[plt_inds[0], :], x[plt_inds[1], :],
                #                        color=color, edgecolors='k',
                #                        label='States')
                added_state_lbl = True
            # else:
            f_hndl.axes[0].plot(x[plt_inds[0], :], x[plt_inds[1], :],
                                **settings)

            s = "({}, {})".format(lbl[0], lbl[1])
            tmp = x.copy()
            tmp = tmp[:, ~np.any(np.isnan(tmp), axis=0)]
            f_hndl.axes[0].text(tmp[plt_inds[0], 0], tmp[plt_inds[1], 0], s,
                                color=color)

        # if true states are available then plot them
        if true_states is not None and any([len(x) > 0 for x in true_states]):
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
            color = (128 / 255, 128 / 255, 128 / 255)
            meas_x = np.asarray(meas_x)
            meas_y = np.asarray(meas_y)
            if meas_x.size > 0:
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
        """Plots the current cardinality distribution.

        This assumes that the cardinality distribution has been calculated by
        the class.

        Keywrod arguments are processed with
        :meth:`gncpy.plotting.init_plotting_opts`. This function
        implements

            - f_hndl

        Returns
        -------
        Matplotlib figure
            Instance of the matplotlib figure used
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

    def plot_card_history(self, time_units='index', time=None, **kwargs):
        """Plots the cardinality history.

        Parameters
        ----------
        time_units : string, optional
            Text representing the units of time in the plot. The default is
            'index'.
        time : numpy array, optional
            Vector to use for the x-axis of the plot. If none is given then
            vector indices are used. The default is None.
        **kwargs : dict
            Additional plotting options for :meth:`gncpy.plotting.init_plotting_opts`
            function. Values implemented here are `f_hndl`, and any values
            relating to title/axis text formatting.

        Returns
        -------
        fig : matplotlib figure
            Figure object the data was plotted on.
        """
        card_history = np.array([len(state_set) for state_set in self.states])

        opts = pltUtil.init_plotting_opts(**kwargs)
        fig = opts['f_hndl']

        if fig is None:
            fig = plt.figure()
            fig.add_subplot(1, 1, 1)

        if time is None:
            time = np.arange(self.card_history.size, dtype=int)

        fig.axes[0].grid(True)
        fig.axes[0].step(time, card_history, where='post')

        pltUtil.set_title_label(fig, 0, opts, ttl="Cardinality History",
                                x_lbl='Time ({})'.format(time_units),
                                y_lbl="Cardinality")
        fig.tight_layout()

        return fig


class STMGeneralizedLabeledMultiBernoulli(GeneralizedLabeledMultiBernoulli):
    """Implementation of a STM-GLMB filter."""

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def _predict_prob_density(self, timestep, probDensity, filt_args):
        """Loops over probability distribution and preforms prediction."""
        self.filter.dof = probDensity.dof
        pd_tup = zip(probDensity.means,
                     probDensity.scalings)
        pd = StudentsTMixture()
        pd.weights = probDensity.weights.copy()
        for ii, (m, P) in enumerate(pd_tup):
            self.filter.scale = P
            n_mean = self.filter.predict(timestep, m, **filt_args)
            pd.scalings.append(self.filter.scale.copy())
            pd.means.append(n_mean)

        return pd

    def _correct_prob_density(self, timestep, meas, probDensity, filt_args):
        """Loops over a probability distribution and preforms correction."""
        self.filter.dof = probDensity.dof
        pd = StudentsTMixture()
        for jj in range(0, len(probDensity.means)):
            self.filter.scale = probDensity.scalings[jj]
            state = probDensity.means[jj]
            (mean, qz) = self.filter.correct(timestep, meas, state, **filt_args)
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
    """Implementation of a Sequential Monte Carlo GLMB filter.

    This is based on :cite:`Vo2014_LabeledRandomFiniteSetsandtheBayesMultiTargetTrackingFilter`
    It does not account for agents spawned from existing tracks, only agents
    birthed from the given birth model.

    Attributes
    ----------
    compute_prob_detection : callable
        Function that takes a list of particles as the first argument and `*args`
        as the next. Returns the probability of detection for each particle as a list.
    compute_prob_survive : callable
        Function that takes a list of particles as the first argument and `*args` as
        the next. Returns the average probability of survival for each particle as a list.
    """

    # inherit from parents local class to extend it
    class _TabEntry(GeneralizedLabeledMultiBernoulli._TabEntry):
        def __init__(self):
            self.prop_parts = []
            self.candDist = None

            super().__init__()

    def __init__(self, compute_prob_detection=None, compute_prob_survive=None,
                 **kwargs):
        self.compute_prob_detection = compute_prob_detection
        self.compute_prob_survive = compute_prob_survive

        # for wrappers for predict/correct function to handle extra args for private functions
        self._prob_surv_args = ()
        self._prob_det_args = ()

        super().__init__(**kwargs)

    def _predict_prob_density(self, timestep, probDensity, filt_args):
        """Predicts the next probability density."""
        self.filter.init_from_dist(probDensity, make_copy=True)

        self.filter.predict(timestep, **filt_args)

        newProbDen = self.filter.extract_dist(make_copy=False)

        ps = self.compute_prob_survive(newProbDen.particles, *self._prob_surv_args)
        new_weights = [w * ps[ii] for ii, (p, w) in enumerate(newProbDen)]
        tot = sum(new_weights)
        if np.abs(tot) == np.inf:
            w_lst = [np.inf] * len(new_weights)
        else:
            w_lst = [w / tot for w in new_weights]
        newProbDen.update_weights(w_lst)
        return newProbDen

    def _predict_track_tab_entry(self, tab, timestep, filt_args):
        if self.filter.require_copy_can_dist:
            self.filter.candDist = deepcopy(tab.candDist)

        newTab = super()._predict_track_tab_entry(tab, timestep, filt_args)

        if self.filter.require_copy_prop_parts:
            newTab.prop_parts = list(np.stack(self.filter.prop_parts).copy())
        if self.filter.require_copy_can_dist:
            newTab.candDist = deepcopy(self.filter.candDist)

        return newTab

    def _calc_avg_prob_surv_death(self):
        avg_prob_survive = np.zeros(len(self._track_tab))
        for tabidx, ent in enumerate(self._track_tab):
            p_surv = self.compute_prob_survive(ent.probDensity.particles,
                                               *self._prob_surv_args)
            avg_prob_survive[tabidx] = np.sum(np.array(ent.probDensity.weights)
                                              * p_surv)

        avg_prob_death = 1 - avg_prob_survive

        return avg_prob_survive, avg_prob_death

    def predict(self, timestep, prob_surv_args=(), **kwargs):
        """Prediction step of the SMC-GLMB filter.

        This is a wrapper for the parent class to allow for extra parameters.
        See :meth:`.tracker.GeneralizedLabeledMultiBernoulli.predict` for
        additional details.

        Parameters
        ----------
        timestep : float
            Current timestep.
        prob_surv_args : tuple, optional
            Additional arguments for the `compute_prob_survive` function.
            The default is ().
        **kwargs : dict, optional
            See :meth:`.tracker.GeneralizedLabeledMultiBernoulli.predict`
        """
        self._prob_surv_args = prob_surv_args
        return super().predict(timestep, **kwargs)

    def _correct_prob_density(self, timestep, meas, probDensity, filt_args):
        """Corrects the probability density and resamples."""
        self.filter.init_from_dist(probDensity, make_copy=True)
        try:
            meas_likely = self.filter.correct(timestep, meas, **filt_args)[1]

            newProbDen = self.filter.extract_dist(make_copy=False)

            # manually upate the weights to allow for probability of detection
            pd = self.compute_prob_detection(newProbDen.particles, *self._prob_det_args)
            pd_weight = pd * np.array(newProbDen.weights)
            newProbDen.update_weights((pd_weight / np.sum(pd_weight)).tolist())

            # determine the partial cost, the remainder is calculated later from
            # the hypothesis
            cost = np.sum(meas_likely * pd_weight)
        except gerr.ParticleDepletionError:
            newProbDen = self.filter.extract_dist(make_copy=False)
            cost = np.inf

        return newProbDen, cost

    def _correct_track_tab_entry(self, meas, tab, timestep, filt_args):
        if self.filter.require_copy_prop_parts:
            if len(tab.prop_parts) == 0:
                # assume this is a new birth and prob_parts hasn't been initialized
                prop_parts = list(np.stack(tab.probDensity.particles).copy())
            else:
                prop_parts = list(np.stack(tab.prop_parts).copy())
            self.filter.prop_parts = prop_parts

        if self.filter.require_copy_can_dist:
            if tab.candDist is None:
                # this has only been initialized by the birth and not predicted
                self.filter.candDist = deepcopy(tab.probDensity)
            else:
                self.filter.candDist = deepcopy(tab.candDist)
        newTab, cost = super()._correct_track_tab_entry(meas, tab, timestep,
                                                        filt_args)
        if self.filter.require_copy_prop_parts:
            newTab.prop_parts = self.filter.prop_parts

        if self.filter.require_copy_can_dist:
            newTab.candDist = self.filter.candDist

        return newTab, cost

    def _calc_avg_prob_det_mdet(self):
        avg_prob_detect = np.zeros(len(self._track_tab))
        for tabidx, ent in enumerate(self._track_tab):
            p_detect = self.compute_prob_detection(ent.probDensity.particles,
                                                   *self._prob_det_args)
            avg_prob_detect[tabidx] = np.sum(np.array(ent.probDensity.weights)
                                             * p_detect)

        avg_prob_miss_detect = 1 - avg_prob_detect

        return avg_prob_detect, avg_prob_miss_detect

    def correct(self, timestep, meas, prob_det_args=(), **kwargs):
        """Correction step of the SMC-GLMB filter.

        This is a wrapper for the parent class to allow for extra parameters.
        See :meth:`.tracker.GeneralizedLabeledMultiBernoulli.correct` for
        additional details.

        Parameters
        ----------
        timestep : float
            Current timestep.
        prob_det_args : tuple, optional
            Additional arguments for the `compute_prob_detection` function.
            The default is ().
        **kwargs : dict, optional
            See :meth:`.tracker.GeneralizedLabeledMultiBernoulli.correct`
        """
        self._prob_det_args = prob_det_args
        return super().correct(timestep, meas, **kwargs)

    def _extract_helper(self, pd):
        new_state = pd.mean
        if new_state.size == 0:
            new_state = None
        new_cov = pd.covariance

        return new_state, new_cov

    def extract_states(self, prob_surv_args=(), prob_det_args=(), **kwargs):
        """Extracts the state estimates.

        This is a wrapper for the parent method to allow for extra arguments.
        See :meth:`.tracker.GeneralizedLabeledMultiBernoulli.extract_states`
        for details.

        Parameters
        ----------
        prob_surv_args : tuple, optional
            Additional arguments for the `compute_prob_survive` function.
            The default is ().
        prob_det_args : tuple, optional
            Additional arguments for the `compute_prob_detection` function.
            The default is ().
        **kwargs : dict, optional
            See :meth:`.tracker.GeneralizedLabeledMultiBernoulli.extract_states`

        """
        self._prob_surv_args = prob_surv_args
        self._prob_det_args = prob_det_args
        return super().extract_states(**kwargs)

    def extract_most_prob_states(self, thresh, **kwargs):
        """Extracts themost probable states.

        .. todo::
            Implement this function for the SMC-GLMB filter

        Raises
        ------
        RuntimeWarning
            Function must be implemented.
        """
        warn('Not implemented for this class')

class JointGeneralizedLabeledMultiBernoulli(GeneralizedLabeledMultiBernoulli):

    def predict(self, timestep, filt_args={}):
        """ Prediction step of the JGLMB filter.

            This predicts new hypothesis, and propogates them to the next time
            step. It also updates the cardinality distribution. Because this calls
            the inner filter's predict function, the keyword arguments must contain
            any information needed by that function.

            Keyword Args:
                time_step (int): Current time step number for the new labels
            """

        # Birth Track Table
        self._pred_timesteps.append(timestep)

        birth_tab = self._gen_birth_tab(timestep)[0]

        # Survival Track Table
        surv_tab = self._gen_surv_tab(timestep, filt_args)

        # Prediction Track Table
        self._track_tab = birth_tab + surv_tab

    def _unique_faster(self, keys):
        difference = np.diff(np.append(keys, np.nan), n=1, axis=0)
        keyind = np.not_equal(difference, 0)
        mindices = (keys[0][np.where(keyind)]).astype(int)
        return mindices

    def correct(self, timestep, meas, filt_args={}):
        """ Correction step of the JGLMB filter.

            This corrects the hypotheses based on the measurements and gates the
            measurements according to the class settings. It also updates the
            cardinality distribution. Because this calls the inner filter's correct
            function, the keyword arguments must contain any information needed by
            that function.

            Keyword Args:
            meas (list): List of Nm x 1 numpy arrays that contain all the
                    measurements needed for this correction
            """

        self._cor_timesteps.append(timestep)
        # gating by tracks
        if self.gating_on:
            for ent in self._track_tab:
                ent.gatemeas = self._gate_meas(meas, ent.probDensity.means,
                                               ent.probDensity.covariances)
        else:
            for ent in self._track_tab:
                ent.gatemeas = np.arange(0, len(meas))
                # ent.gatemeas = np.arange(0, np.shape(meas)[1]) # maybe np.shape... +1?

        # Pre-calculation of average survival/death probabilities
        avg_surv = self.prob_survive * np.ones(len(self._track_tab))
        avg_death = 1 - avg_surv

        # Pre-calculation of average detection/missed probabilities
        avg_detect = self.prob_detection * np.ones(len(self._track_tab))
        avg_miss = 1 - avg_detect

        # num_meas = np.shape(meas)[1]
        self._meas_tab.append(deepcopy(meas))
        num_meas = len(meas)

        # missed detection tracks
        num_pred = len(self._track_tab)
        up_tab = []
        #take a look at this, might be the source of the issue.
        # current thought: look at how matlab track table updates,
        # compare to ryan's glmb and vo jglmb/glmb, figure out how the up_tab
        # initialization/propagation should occur, likely something is going
        # wrong here because the filter is getting the best possible estimate,
        # but there's a lack of time_index which communicates some issues.
        # Could also be a problem with the meas_assoc_hist thing, but probably not.
        for ii in range(0, (num_meas + 1) * num_pred):
            up_tab.append(self._TabEntry())

        for ii, track in enumerate(self._track_tab):
            up_tab[ii] = deepcopy(track)
            up_tab[ii].meas_assoc_hist.append(None)

        # measurement updated tracks
        all_cost_m = np.zeros((num_pred, num_meas))
        for emm, z in enumerate(meas):
            for ii, ent in enumerate(self._track_tab):
                s_to_ii = num_pred * emm + ii + num_pred - 1
                [up_tab[s_to_ii].probDensity, cost] = \
                    self._correct_prob_density(timestep, z, ent.probDensity,
                                               filt_args)

                # update association history with current measurement index
                up_tab[s_to_ii].meas_assoc_hist = ent.meas_assoc_hist + [emm]
                up_tab[s_to_ii].label = ent.label
                all_cost_m[ii, emm] = cost
        clutter = self.clutter_rate * self.clutter_den
        # Joint Cost Matrix
        if num_meas == 0:
            joint_cost = np.concatenate([np.diag(avg_death),
                                         np.diag(avg_surv * avg_miss)], axis=1)
        else:
            joint_cost = np.concatenate([np.diag(avg_death),
                                     np.diag(avg_surv * avg_miss)], axis=1)

            other_jc_terms = np.matlib.repmat(avg_surv * avg_detect, 1, num_meas).T * all_cost_m / (clutter)

            joint_cost = np.append(joint_cost, other_jc_terms, axis=1)



        # joint_cost = np.concatenate([joint_cost,
        #                              np.matlib.repmat(avg_surv*avg_detect,
        #                                         1, num_meas)*all_cost_m/(clutter)], axis=1)
        # if num_meas == 0:
        #     joint_cost = np.append(joint_cost,
        #                            np.matlib.repmat(avg_surv*avg_detect,
        #                                             1, 1)*all_cost_m/(clutter))

        # Gated Measurement index matrix
        gate_meas_indices = np.zeros((len(self._track_tab), num_meas))
        for ii in range(0, len(self._track_tab)):
            for jj in range(0, len(self._track_tab[ii].gatemeas)):
                gate_meas_indices[ii][jj] = self._track_tab[ii].gatemeas[jj]
        gate_meas_indc = gate_meas_indices >= 0

        # Component updates
        ss_w = 0
        up_hyp = []
        for p_hyp in self._hypotheses:
            ss_w += np.sqrt(p_hyp.assoc_prob)
        for p_hyp in self._hypotheses:
            cpreds = len(self._track_tab)
            num_births = np.shape(self.birth_terms)[0]
            num_exists = len(p_hyp.track_set)
            num_tracks = num_births + num_exists
            tindices = np.concatenate((np.arange(0, num_births),
                                           num_births + np.array(p_hyp.track_set))).astype(int)
            lselmask = np.zeros((len(self._track_tab), num_meas), dtype='bool')
            lselmask[tindices, ] = gate_meas_indc[tindices, ]
            # keys = gate_meas_indices[lselmask].ravel().sort()
            keys = np.array([np.sort(gate_meas_indices[lselmask])])
            mindices = self._unique_faster(keys)

            # keys = np.sort(gate_meas_indices[lselmask])
            # difference = np.diff([keys, np.nan], n=1, axis=0)
            # keyind = np.not_equal(difference, 0)
            # mindices = keys[np.where(keyind)]

            if num_meas == 0:
                cost_m = joint_cost[tindices, [tindices, cpreds + tindices]].T
            else:
                cost_m = np.zeros((len(tindices), len(np.append(np.append(tindices, cpreds + tindices),
                                                        [2 * cpreds + mindices]))))
                cmi = 0
                for ind in tindices:
                    cost_m[cmi,:] = joint_cost[ind, np.append(np.append(tindices, cpreds + tindices), [2 * cpreds + mindices])]
                    cmi = cmi + 1
                # cost_m = np.array([joint_cost[tindices, np.append(np.append(tindices,
                #                                                             cpreds + tindices),
                #                                         [2 * cpreds + mindices])]])
            # if num_meas == 0:
            #     cost_m = np.array([joint_cost[tindices, [tindices, cpreds + tindices]]]).T
            # else:
            #     cost_m = np.array([joint_cost[tindices, np.append([tindices,
            #                                                         cpreds + tindices],
            #                                                       [2 * cpreds + mindices]).astype(int)]]).T
            # cost_m = np.concatenate((joint_cost[tindices],
            #                          joint_cost[cpreds+tindices],
            #                          joint_cost[2*cpreds+mindices]))
            neg_log = -np.log(cost_m)
            m = np.round(self.req_upd * np.sqrt(p_hyp.assoc_prob)/ ss_w)
            m = int(m.item())+1

            [assigns, costs] = gibbs(neg_log, m) # (rename)
            #this whole section may need to be redone or re-evaluated based on how indexing works in python vs matlab
            assigns[assigns<num_tracks] = -np.inf*np.ones(np.shape(assigns[assigns<num_tracks]))
            for ii in range(np.shape(assigns)[0]):
                if len(np.shape(assigns)) < 2:
                    if assigns[ii] >= num_tracks and assigns[ii] < 2*num_tracks:
                        assigns[ii] = -1
                else:
                    for jj in range(np.shape(assigns)[1]):
                        if assigns[ii][jj] >= num_tracks and assigns[ii][jj] < 2*num_tracks:
                           assigns[ii][jj] = -1
            assigns[assigns >= 2*num_tracks-1] = assigns[assigns >= 2*num_tracks-1]-2*num_tracks
            if assigns[assigns>=0].size != 0:
                assigns[assigns>=0] = mindices[assigns.astype(int)[assigns.astype(int)>=0]]

            for c in range(0, len(costs)):
                update_hyp_cmp_temp = assigns[c, ]
                update_hyp_cmp_idx = cpreds*(update_hyp_cmp_temp + 1) - 1 + \
                np.append(np.array([np.arange(0, num_births)]), num_births + np.array([p_hyp.track_set]))
                                   # + self._hypotheses.track_set)
                new_hyp = self._HypothesisHelper()
                new_hyp.assoc_prob = self.clutter_rate + num_meas *np.log(clutter) \
                    + np.log(p_hyp.assoc_prob)
                new_hyp.track_set = update_hyp_cmp_idx[update_hyp_cmp_idx>=0].astype(int)
                up_hyp.append(new_hyp)

        lse = log_sum_exp([x.assoc_prob for x in up_hyp])
        for ii in range(0, len(up_hyp)):
            up_hyp[ii].assoc_prob = np.exp(up_hyp[ii].assoc_prob - lse)

        self._track_tab = up_tab
        self._hypotheses = up_hyp
        self._card_dist = self._calc_card_dist(self._hypotheses)
        self._clean_predictions()
        self._clean_updates()
