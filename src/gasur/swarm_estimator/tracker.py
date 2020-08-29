import numpy as np
import abc

from gncpy.filters import BayesFilter
from gncpy.math import log_sum_exp
from gasur.utilities.distributions import GaussianMixture
from gasur.utilities.graphs import k_shortest


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
        def __init___(self):
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

        self._track_tab = []  # list of _TabEntry objects

        hyp0 = self._HypothesisHelper()
        hyp0.assoc_prob = 1
        hyp0.track_set = []
        self._hypotheses = [hyp0]  # list of _HypothesisHelper objects

        self._card_dist = []  # probability of having index # as cardinality
        super().__init__(**kwargs)

    def predict(self, **kwargs):
        # Find cost for each birth track, and setup lookup table
        time_step = kwargs['time_step']
        log_cost = []
        birth_tab = []
        for ii, (gm, p) in enumerate(self.birth_terms):
            cost = p / (1 - p)
            log_cost.append(-np.log(cost))
            entry = self._TabEntry()
            entry.probDensity = gm
            entry.label = (time_step, ii)
            birth_tab.append(entry)

        # get K best hypothesis, and their index in the lookup table
        (paths, hyp_cost) = k_shortest(log_cost, self.req_births)

        # calculate association probabilities for birth hypothesis
        tot_cost = 0
        for c in hyp_cost:
            tot_cost = tot_cost + np.exp(-c)
        birth_hyps = []
        for (p, c) in zip(paths, hyp_cost):
            hyp = self._HypothesisHelper()
            # NOTE: this may suffer from underflow and can be improved
            hyp.assoc_prob = np.exp(-c) / tot_cost
            hyp.track_set = p
            birth_hyps.append(hyp)

        # Init and propagate surviving track table
        surv_tab = []
        for track in self._track_tab:
            gm_tup = zip(track.probDensity.means,
                         track.probDenisty.covariances)
            c_in = np.zeros((self.filter.get_input_mat().shape[0], 1))
            gm = GaussianMixture()
            gm.weights = track.probDensity.weights
            for ii, (m, P) in enumerate(gm_tup):
                self.filter.cov = P
                n_mean = self.filter.predict(cur_state=m, cur_input=c_in)
                gm.covariances.append(self.filter.cov)
                gm.means.append(n_mean)

            entry = self._TabEntry()
            entry.probDensity = gm
            entry.assoc_hist = track.assoc_hist
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
                        * np.log(self.prob_death) + np.log(hyp.assoc_prob) - c
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

    def correct(self, **kwargs):
        meas = kwargs['meas']
        num_meas = len(meas)
        up_tab = []
        for track in self._track_tab:
            up_tab.append(track)
            up_tab[-1].assoc_hist.append(GaussianMixture())

        assert 0

    def extract_states(self, **kwargs):
        assert 0

    def calc_card_dist(self, hyp_lst):
        if len(hyp_lst) == 0:
            return 0

        card_dist = []
        for ii in range(0, max(map(lambda x: x.num_tracks, hyp_lst))):
            card = 0
            for hyp in hyp_lst:
                if hyp.num_tracks == ii:
                    card = card + hyp.assoc_prob
            card_dist.append(card)
        return card_dist

    def clean_predictions(self):
        hash_lst = map(lambda x: hash(x.track_set.sort()), self._hypotheses)
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
