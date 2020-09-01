import numpy as np
from numpy.linalg import cholesky, inv
import abc
from copy import deepcopy

from gncpy.filters import BayesFilter
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
    def _extract_states(self, **kwargs):
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
            self.label = ()
            self.probDensity = GaussianMixture()
            self.assoc_hist = []  # list of gaussian mixtures

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

        hyp0 = self._HypothesisHelper()
        hyp0.assoc_prob = 1
        hyp0.track_set = []
        self._hypotheses = [hyp0]  # list of _HypothesisHelper objects

        self._card_dist = []  # probability of having index # as cardinality
        super().__init__(**kwargs)

    @property
    def states(self):
        """ Read only property
        """
        return self._states

    @property
    def labels(self):
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
        (paths, hyp_cost) = k_shortest(log_cost, self.req_births)

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
        for track in self._track_tab:
            gm_tup = zip(track.probDensity.means,
                         track.probDenisty.covariances)
            c_in = np.zeros((self.filter.get_input_mat().shape[0], 1))
            gm = GaussianMixture()
            gm.weights = track.probDensity.weights.copy()
            for ii, (m, P) in enumerate(gm_tup):
                self.filter.cov = P
                n_mean = self.filter.predict(cur_state=m, cur_input=c_in)
                gm.covariances.append(self.filter.cov.copy())
                gm.means.append(n_mean)

            entry = self._TabEntry()
            entry.probDensity = gm
            entry.assoc_hist = deepcopy(track.assoc_hist)
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
                (paths, hyp_cost) = k_shortest(log_cost, k)

                for (p, c) in zip(paths, hyp_cost):
                    new_hyp = self._HypothesisHelper()
                    new_hyp.assoc_prob = hyp.num_tracks \
                        * np.log(self.prob_death) + np.log(hyp.assoc_prob) \
                        - np.asscalar(c)
                    new_hyp.track_set = hyp.track_set[p]
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
        self._extract_states(**kwargs)

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

        num_meas = len(meas)

        # missed detection tracks
        num_pred = len(self._track_tab)
        up_tab = []
        for ii in range(0, (num_meas + 1) * num_pred):
            up_tab.append(self._TabEntry())

        for ii, track in enumerate(self._track_tab):
            up_tab[ii] = deepcopy(track)
            up_tab[ii].assoc_hist.append(None)

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

                # update association history with current state
                up_tab[s_to_ii].assoc_hist = ent.assoc_hist \
                    + [up_tab[s_to_ii].probDensity]
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
        self._extract_states(corr_updt=True, **kwargs)

    def _extract_states(self, **kwargs):
        corr_updt = kwargs.get('corr_updt', False)
        card = np.argmax(self._card_dist)
        new_states = []
        new_labels = []

        if card > 0:
            tracks_per_hyp = np.array([x.num_tracks for x in self._hypotheses])
            weight_per_hyp = np.array([x.assoc_prob for x in self._hypotheses])

            idx_cmp = np.argmax(weight_per_hyp * (tracks_per_hyp == card))
            track_gms = []
            track_labs = []
            for ptr in self._hypotheses[idx_cmp].track_set:
                track_gms.append(self._track_tab[ptr].assoc_hist[-1])
                track_labs.append(self._track_tab[ptr].label)

            # make sure no duplicate labels
            used = []
            for lab in track_labs:
                if lab in used:
                    msg = 'Probably should not have duplicate labels.'
                    msg = msg + ' Debug this'
                    raise RuntimeError(msg)
                else:
                    used.append(lab)

            for (gm, lab) in zip(track_gms, track_labs):
                idx = np.argmax(gm.weights)
                new_states.append(gm.means(idx))
                new_labels.append(lab)

        if corr_updt:
            self._states[-1] = new_states
            self._labels[-1] = new_labels
        else:
            self._states.append(new_states)
            self._labels.append(new_labels)

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

        new_tab = [self._track_tab[ii] for ii in nnz_inds]
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
