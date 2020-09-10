import numpy as np
from numpy.linalg import cholesky, inv
import numpy.random as rnd
import matplotlib.pyplot as plt
import abc
from copy import deepcopy

from gncpy.math import log_sum_exp
from gasur.utilities.distributions import GaussianMixture
from gasur.utilities.graphs import k_shortest, murty_m_best


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


class GeneralizedLabeledMultiBernoulli(RandomFiniteSetBase):
    """ Delta-Generalized Labeled Multi-Bernoulli filter.

    This is based on :cite:`Vo2013_LabeledRandomFiniteSetsandMultiObjectConjugatePriors`
    and :cite:`Vo2014_LabeledRandomFiniteSetsandtheBayesMultiTargetTrackingFilter`

    Attributes:
        birth_terms (list): List of tuples where the first element is
            a :py:class:`gasur.utilities.distributions.GaussianMixture` and
            the second is the birth probability for that term
        req_births (int): Modeled maximum number of births
    """

    class _TabEntry:
        def __init__(self):
            self.label = ()  # time step born, index of birth model born from
            self.probDensity = GaussianMixture()
            self.meas_assoc_hist = []  # list indices into measurement list per time step

    class _HypothesisHelper:
        def __init__(self):
            self.assoc_prob = 0
            self.track_set = []  # indices in lookup table

        @property
        def num_tracks(self):
            return len(self.track_set)

    def __init__(self, **kwargs):
        self.req_births = 0  # filter.H_bth
        self.req_surv = 0  # filter.H_surv
        self.req_upd = 0  # filter.H_upd
        self.gating_on = False
        self.inv_chi2_gate = 0  # filter.gamma

        self._track_tab = []  # list of _TabEntry objects
        self._states = [] # list of lists, one per time step, inner list is all states alive at that time
        self._labels = [] # list of list, one per time step, inner list is all labels alive at that time
        self._meas_tab = []  # list of lists, one per timestep, inner is all meas at time
        self._meas_asoc_mem = []
        self._lab_mem = []

        hyp0 = self._HypothesisHelper()
        hyp0.assoc_prob = 1
        hyp0.track_set = []
        self._hypotheses = [hyp0]  # list of _HypothesisHelper objects

        self._card_dist = []  # probability of having index # as cardinality
        self.prune_threshold = 1*10**(-15) # hypothesis pruning threshold
        self.max_hyps = 3000 # hypothesis capping threshold

        super().__init__(**kwargs)

    @property
    def states(self):
        """ Read only property
        """
        return self._states

    @property
    def labels(self):
        """ Read only property
        """
        return self._labels

    def predict(self, **kwargs):
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
            gm_tup = zip(track.probDensity.means,
                         track.probDensity.covariances)
            c_in = np.zeros((self.filter.get_input_mat().shape[1], 1))
            gm = GaussianMixture()
            gm.weights = track.probDensity.weights.copy()
            for ii, (m, P) in enumerate(gm_tup):
                self.filter.cov = P
                n_mean = self.filter.predict(cur_state=m, cur_input=c_in,
                                             **kwargs)
                gm.covariances.append(self.filter.cov.copy())
                gm.means.append(n_mean)

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
        self.clean_predictions()

    def correct(self, **kwargs):
        meas = kwargs['meas']
        del kwargs['meas']

        # gate measurements by tracks
        if self.gating_on:
            means = []
            covs = []
            for ent in self._track_tab:
                means.extend(ent.probDensity.means)
                covs.extend(ent.probDensity.covariances)
            meas = self.gate_meas(meas, means, covs, **kwargs)

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
                up_tab[s_to_ii].probDensity.means = []
                up_tab[s_to_ii].probDensity.covariances = []
                up_tab[s_to_ii].probDensity.weights = []
                for jj in range(0, len(ent.probDensity.means)):
                    self.filter.cov = ent.probDensity.covariances[jj]
                    state = ent.probDensity.means[jj]
                    (mean, qz) = self.filter.correct(meas=z, cur_state=state,
                                                     **kwargs)
                    cov = self.filter.cov
                    w = qz * ent.probDensity.weights[jj]
                    up_tab[s_to_ii].probDensity.means.append(mean)
                    up_tab[s_to_ii].probDensity.covariances.append(cov)
                    up_tab[s_to_ii].probDensity.weights.append(w)
                lst = up_tab[s_to_ii].probDensity.weights
                lst = [x + np.finfo(float).eps for x in lst]
                up_tab[s_to_ii].probDensity.weights = lst
                tmp_sum = sum(up_tab[s_to_ii].probDensity.weights)
                for jj in range(0, len(up_tab[s_to_ii].probDensity.weights)):
                    up_tab[s_to_ii].probDensity.weights[jj] /= tmp_sum

                # update association history with current measurement index
                up_tab[s_to_ii].meas_assoc_hist = ent.meas_assoc_hist + [emm]
                up_tab[s_to_ii].label = ent.label
                all_cost_m[ii, emm] = tmp_sum

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
        lst = [x.assoc_prob for x in self._hypotheses]
        self.clean_updates()

    def extract_states(self, **kwargs):
        card = np.argmax(self._card_dist)
        tracks_per_hyp = np.array([x.num_tracks for x in self._hypotheses])
        weight_per_hyp = np.array([x.assoc_prob for x in self._hypotheses])

        if len(tracks_per_hyp) == 0:
            self._states = [[]]
            self._labels = [[]]
            return

        idx_cmp = np.argmax(weight_per_hyp * (tracks_per_hyp == card))
        meas_hists = []
        labels = []
        for ptr in self._hypotheses[idx_cmp].track_set:
            meas_hists.append(self._track_tab[ptr].meas_assoc_hist.copy())
            labels.append(self._track_tab[ptr].label)

        both = set(self._lab_mem).intersection(labels)
        surv_ii = [labels.index(x) for x in both]
        either = set(self._lab_mem).symmetric_difference(labels)
        dead_ii = [self._lab_mem.index(a) for a in either if a in self._lab_mem]
        new_ii = [labels.index(a) for a in either if a in labels]

        self._lab_mem = [self._lab_mem[ii] for ii in dead_ii] \
            + [labels[ii] for ii in surv_ii] \
            + [labels[ii] for ii in new_ii]
        self._meas_asoc_mem = [self._meas_asoc_mem[ii] for ii in dead_ii] \
            + [meas_hists[ii] for ii in surv_ii] \
            + [meas_hists[ii] for ii in new_ii]

        self._states = [None] * len(self._meas_tab)
        self._labels = [None] * len(self._meas_tab)
        c_in = np.zeros((self.filter.get_input_mat().shape[1], 1))

        # if there are no old or new tracks assume its the first iteration
        if len(self._lab_mem) == 0 and len(self._meas_asoc_mem) == 0:
            self._states = [[]]
            self._labels = [[]]
            return

        for (hist, (b_time, b_idx)) in zip(self._meas_asoc_mem, self._lab_mem):
            weights = self.birth_terms[b_idx][0].weights.copy()
            means = self.birth_terms[b_idx][0].means.copy()
            covs = self.birth_terms[b_idx][0].covariances.copy()

            for (t_after_b, emm) in enumerate(hist):
                # propagate for GM
                for ii in range(0, len(weights)):
                    self.filter.cov = covs[ii]
                    means[ii] = self.filter.predict(cur_state=means[ii],
                                                    cur_input=c_in, **kwargs)
                    covs[ii] = self.filter.cov.copy()

                # measurement correction for GM
                tt = b_time + t_after_b
                if emm is not None:
                    meas = self._meas_tab[tt][emm].copy()
                    for ii in range(0, len(weights)):
                        state = means[ii]
                        self.filter.cov = covs[ii]
                        (means[ii], qz) = self.filter.correct(meas=meas,
                                                              cur_state=state,
                                                              **kwargs)
                        covs[ii] = self.filter.cov.copy()
                        weights[ii] = weights[ii] * qz + np.finfo(float).eps
                    s_w = sum(weights)
                    weights = [x / s_w for x in weights]

                # find best one and add to state table
                idx_trk = np.argmax(weights)
                new_state = means[idx_trk]
                new_label = (b_time, b_idx)
                if self._labels[tt] is None:
                    self._states[tt] = [new_state]
                    self._labels[tt] = [new_label]
                else:
                    self._states[tt].append(new_state)
                    self._labels[tt].append(new_label)

    def prune(self, **kwargs):
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

    def clean_predictions(self):
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

    def clean_updates(self):
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

    def gate_meas(self, meas, means, covs, **kwargs):
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
        f_hndl = kwargs.get('f_hndl', None)
        true_states = kwargs.get('true_states', None)

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
        for lbl in u_lbls:
            x = np.nan * np.ones((x_dim, len(s_lst)))
            for tt, lbls in enumerate(l_lst):
                if lbl in lbls:
                    ii = lbls.index(lbl)
                    x[:, [tt]] = s_lst[tt][ii].copy()

            # plot
            r = rnd.random()
            b = rnd.random()
            g = rnd.random()
            color = (r, g, b)
            f_hndl.axes[0].scatter(x[plt_inds[0], :], x[plt_inds[1], :],
                                   color=color)
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
                f_hndl.axes[0].plot(x[plt_inds[0], :, ii],
                                    x[plt_inds[1], :, ii],
                                    color='k')

        f_hndl.axes[0].grid(True)

        return f_hndl

    def plot_card_dist(self, **kwargs):
        f_hndl = kwargs.get('f_hndl', None)

        if len(self._card_dist) == 0:
            raise RuntimeWarning("Empty Cardinality")
            return f_hndl

        if f_hndl is None:
            f_hndl = plt.figure()
            f_hndl.add_subplot(1, 1, 1)

        x_vals = np.arange(0, len(self._card_dist))
        f_hndl.axes[0].bar(x_vals, self._card_dist)
