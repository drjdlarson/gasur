import pytest
import numpy as np
import numpy.testing as test
import numpy.random as rnd
import scipy.stats as stats
from copy import deepcopy
import scipy.linalg as la
import numpy.linalg as nla

import gasur.swarm_estimator.tracker as tracker
import gasur.utilities.distributions as mixtures
import gncpy.distributions as distributions
from gncpy.math import rk4
import gncpy.filters as filters


class TestGeneralizedLabeledMultiBernoulli:
    def test_multiple_timesteps(self, glmb_tup):
        glmb = glmb_tup[0]
        dt = glmb_tup[1]

        lst0 = [[0.633157293, 1773.703654],
                [1.18789096, 54.7751864],
                [0.535539478, 834.6096047],
                [0.184379534, 280.7738772],
                [-0.948442144, 1601.489137],
                [1.471087126, 626.8483563],
                [0.604199317, 1752.778305],
                [1.239693395, 170.0884227],
                [-1.448102107, 339.6608391],
                [1.187969711, 196.6936677],
                [-0.247847706, 1915.77906],
                [0.104191816, 1383.754228],
                [-0.579574738, 1373.001855],
                [1.051257553, 36.57655469],
                [0.785851542, 1977.722178],
                [0.779635397, 560.8879841],
                [0.908797813, 206.4520132],
                [-0.163697315, 1817.191006],
                [-0.648380275, 575.5506772]]

        lst1 = [[0.566508296, 1776.656535],
                [0.280561619, 1399.51672],
                [-1.249303237, 828.1119756],
                [0.610726107, 828.3585391],
                [-1.413862907, 1071.792812],
                [0.514576054, 1029.778224],
                [1.396735619, 1173.110081],
                [1.267324494, 274.9494083],
                [-1.133246777, 1614.782577],
                [-0.321457697, 330.7083942],
                [1.343057816, 695.5317195],
                [0.787949461, 1451.995971],
                [1.2041916, 1247.344414],
                [0.788358907, 697.796684],
                [-0.722792845, 1791.772436],
                [-0.22590819, 1929.680094],
                [0.513466609, 1243.39144]]

        lst2 = [[0.609257256, 1752.20395],
                [0.3680216, 653.2898035],
                [0.085005535, 1771.884199],
                [-0.448400273, 1817.070302],
                [0.387547234, 31.64248569],
                [1.349116859, 1381.793835],
                [1.562385813, 344.6810167],
                [-1.139971663, 1865.190926],
                [0.61832249, 132.0003454],
                [0.802560849, 1507.752377],
                [1.328970773, 1423.049517],
                [-1.180387586, 39.76026768],
                [-1.488452083, 56.61297604],
                [-0.797301446, 1720.055897],
                [0.121991386, 1105.643957],
                [1.074521739, 248.3466302],
                [-0.693714932, 1171.518543]]

        lst = [lst0, lst1, lst2]
        for k in range(0, 3):
            meas = []
            for z in lst[k]:
                meas.append(np.array(z).reshape((2, 1)))

            glmb.predict(time_step=k, dt=dt)
            glmb.correct(meas=meas)
            glmb.extract_states(dt=dt)

        # check number of time steps
        assert len(glmb.states) == 3
        assert len(glmb.labels) == 3

        # check for correct labeling
        assert glmb.labels[2][0] == (0, 3)

        # check number of agents
        assert len(glmb.states[1]) == 1
        assert len(glmb.labels[1]) == 1
        assert len(glmb.states[2]) == 1
        assert len(glmb.labels[2]) == 1

        act_state = glmb.states[1][0]
        exp_state = np.array([988.471986936194, -5.85688289282618,
                              1476.43207756921, 5.00728692275889,
                              0.000613071164420375]).reshape((5, 1))

        test.assert_array_almost_equal(act_state, exp_state, decimal=0)

        act_state = glmb.states[2][0]
        exp_state = np.array([995.355927676135, -4.11260562494401,
                              1447.77931434226, -11.2011621565549,
                              -0.00380720670474486]).reshape((5, 1))

        test.assert_array_almost_equal(act_state, exp_state, decimal=1)


def test_STMGeneralizedLabeledMultiBernoulli():
    rng = rnd.default_rng(1)
    state_mat = np.array([[1, 0, 1, 0],
                          [0, 1, 0, 1],
                          [0, 0, 1, 0],
                          [0, 0, 0, 1]])
    pos_bnds = np.array([[-2000, 2000],
                         [-2000, 2000]])
    meas_mat = np.array([[1, 0, 0, 0],
                         [0, 1, 0, 0]])

    proc_noise_dof = 3
    meas_noise_dof = 3

    factor = (proc_noise_dof - 2) / proc_noise_dof
    proc_noise_scale = factor * 0.5**2 * np.eye(4)
    sqrt_proc_scale = nla.cholesky(proc_noise_scale)

    factor = (meas_noise_dof - 2) / meas_noise_dof
    meas_noise_scale = factor * 0.1**2 * np.eye(2)
    sqrt_meas_scale = nla.cholesky(meas_noise_scale)

    filt = filters.StudentsTFilter()
    filt.set_state_mat(mat=state_mat)
    filt.set_proc_noise(mat=proc_noise_scale)

    filt.set_meas_mat(mat=meas_mat)
    filt.meas_noise = meas_noise_scale

    filt.meas_noise_dof = meas_noise_dof
    filt.proc_noise_dof = proc_noise_dof

    glmb = tracker.STMGeneralizedLabeledMultiBernoulli()
    glmb.filter = filt
    glmb.prob_detection = 0.9999
    glmb.prob_survive = 0.99

    dof = 3
    mu = [np.array([-1500., 250., 0, 0]).reshape((4, 1))]
    scale = [dof / (dof - 2) * np.diag(np.array([15, 15, 5, 5]))**2]
    stm0 = mixtures.StudentsTMixture(means=mu, scalings=scale,
                                     weights=[1], dof=dof)
    mu = [np.array([-250., 1000., 0, 0.]).reshape((4, 1))]
    stm1 = mixtures.StudentsTMixture(means=mu, scalings=scale.copy(),
                                     weights=[1], dof=dof)
    mu = [np.array([250., 750., 0, 0]).reshape((4, 1))]
    stm2 = mixtures.StudentsTMixture(means=mu, scalings=scale.copy(),
                                     weights=[1], dof=dof)
    mu = [np.array([1000.,  1500., 0, 0]).reshape((4, 1))]
    stm3 = mixtures.StudentsTMixture(means=mu, scalings=scale.copy(),
                                     weights=[1], dof=dof)

    glmb.birth_terms = [(stm0, 0.02), (stm1, 0.02), (stm2, 0.02), (stm3, 0.02)]
    b_probs = [x[1] for x in glmb.birth_terms]
    b_probs.append(1 - sum(b_probs))
    glmb.req_births = 5
    glmb.req_surv = 3000
    glmb.req_upd = 3000
    glmb.gating_on = False
    glmb.inv_chi2_gate = 32.2361913029694
    glmb.clutter_rate = 0.00005
    glmb.clutter_den = 1 / np.prod(pos_bnds[:, 1] - pos_bnds[:, 0])
    glmb.save_covs = True

    true_states = []
    true_plt_states = []
    for tt in range(0, 15):

        for ii in range(0, len(true_states)):
            x = true_states[ii]
            t = rng.standard_t(proc_noise_dof, size=x.size).reshape(x.shape)
            proc_noise = sqrt_proc_scale @ t
            x = state_mat @ x + proc_noise
            true_states[ii] = x

        meas = []
        if tt == 0:
            m = np.array([-1500 + 5,
                          250 - 10, 3, 5]).reshape((4, 1))
            true_states.append(m)
        elif tt == 3:
            m = np.array([1000 - 20, 1500 + 18., -2, -6]).reshape((4, 1))
            true_states.append(m)
        elif tt == 6:
            m = np.array([-250 - 10, 1000 + 28, -8, 6]).reshape((4, 1))
            true_states.append(m)
        elif tt == 10:
            m = np.array([250 + 10, 750 - 16., 7, -3]).reshape((4, 1))
            true_states.append(m)

        true_plt_states.append(deepcopy(true_states))

        for x in true_states:
            m = meas_mat @ x
            t = rng.standard_t(meas_noise_dof, size=m.size).reshape(m.shape)
            meas_noise = sqrt_meas_scale @ t
            meas.append(m + meas_noise)

        num_clutt = rng.poisson(glmb.clutter_rate)
        for ii in range(0, num_clutt):
            m = pos_bnds[:, [0]] + (pos_bnds[:, [1]] - pos_bnds[:, [0]]) \
                * rng.random((pos_bnds.shape[0], 1))
            meas.append(m)

        glmb.predict(time_step=tt)
        glmb.correct(meas=meas)
        glmb.prune()
        glmb.cap()
        glmb.extract_states()

    glmb.plot_states_labels([0, 1], true_states=true_plt_states,
                            meas_inds=[0, 1], sig_bnd=None)
    # print("Cardinality:")
    # print(glmb.cardinality)
    # print("labels")
    # print(glmb.labels)
    assert glmb.cardinality == 4, "Cardinality does not match"


class TestSMCGeneralizedLabeledMultiBernoulli:
    def __init__(self, **kwargs):
        self.rng = rnd.default_rng(1)

        self.max_time = 3
        self.dt = 1.0

        # proc noise
        self.sig_w = 2.5
        self.sig_u = np.pi / 180

        # meas noise
        self.std_turn = 0.25 * (np.pi / 180)
        self.std_pos = 5.0

        self.prob_detection = 0.98
        self.prob_survive = 0.99

        self.G = np.array([[self.sig_w * self.dt**2 / 2, 0, 0],
                          [self.sig_w * self.dt, 0, 0],
                          [0, self.sig_w * self.dt**2 / 2, 0],
                          [0, self.sig_w * self.dt, 0],
                          [0, 0, self.sig_u]])
        self.Q = np.eye(3)
        self.proc_std = nla.cholesky(self.Q)

    def compute_prob_detection(self, part_lst, **kwargs):
        if len(part_lst) == 0:
            return np.array([])
        else:
            inv_std = np.diag(np.array([1. / 2000., 1. / 2000.]))

            e_sq = np.sum(np.hstack([(inv_std
                                      @ x[[0, 2], 0].reshape((2, 1)))**2
                                     for x in part_lst]), axis=0)
            return self.prob_detection * np.exp(-e_sq / 2.)

    def compute_prob_survive(self, part_lst, **kwargs):
        if len(part_lst) == 0:
            return np.array([])
        else:
            return self.prob_survive * np.ones(len(part_lst))

    def meas_mod(self, state, **kwargs):
        x_pos = state[0, 0]
        y_pos = state[2, 0]
        z1 = np.arctan2(y_pos, x_pos)
        z2 = np.sqrt(x_pos**2 + y_pos**2)
        return np.array([[z1], [z2]])

    def meas_likelihood(self, meas, est, **kwargs):
        cov = np.array([[self.std_turn**2, 0],
                        [0, self.std_pos**2]])
        return stats.multivariate_normal.pdf(meas.copy().reshape(meas.size),
                                             mean=est.copy().reshape(est.size),
                                             cov=cov)

    # \dot{x} = f(x)
    def cont_dyn(self, x, **kwargs):
        # returns x_dot
        def f0(x, **kwargs):
            return x[1]

        # returns x_dot_dot
        def f1(x, **kwargs):
            return -x[4] * x[3]

        # returns y_dot
        def f2(x, **kwargs):
            return x[3]

        # returns y_dot_dot
        def f3(x, **kwargs):
            return x[4] * x[1]

        # returns omega_dot
        def f4(x, **kwargs):
            return 0

        out = np.zeros(x.shape)
        for ii, f in enumerate([f0, f1, f2, f3, f4]):
            out[ii] = f(x, **kwargs)
        return out

    # x_{k + 1} = f(x_{k})
    def dyn_fnc(self, x, noise_on=False, **kwargs):
        ctrl = np.zeros((2, 1))
        ns = rk4(self.cont_dyn, x.copy(), self.dt, cur_input=ctrl)
        if noise_on:
            dim = self.proc_std.shape[0]
            samp = self.proc_std @ self.rng.standard_normal((dim, 1))
            ns += self.G @ samp
        return ns

    def init_glmb(self, glmb, birth_terms):
        glmb.compute_prob_detection = self.compute_prob_detection
        glmb.compute_prob_survive = self.compute_prob_survive
        glmb.prob_detection = self.prob_detection
        glmb.prob_survive = self.prob_survive
        glmb.birth_terms = birth_terms
        glmb.req_births = 5
        glmb.req_surv = 5000
        glmb.req_upd = 5000
        glmb.gating_on = False
        glmb.clutter_rate = 0.0000001  # 10
        glmb.clutter_den = 1 / (np.pi * 2000)

    def prop_states(self, k, true_states, noise_on=True):
        new_states = []
        for s in true_states:
            ns = self.dyn_fnc(s.copy(), noise_on=noise_on)
            new_states.append(ns)

        wturn = 2 * np.pi / 180
        if k == 0:
            s = np.array([1000 + 3.8676, -10, 1500 - 11.7457, -10, wturn / 8])
            new_states.append(s.reshape((5, 1)))

        return new_states

    def gen_meas(self, true_states, clutter_rate):
        meas = []
        for s in true_states:
            # if self.rng.random() <= self.compute_prob_detection([s]):
            m = self.meas_mod(s)
            m += np.array([[self.std_turn], [self.std_pos]]) \
                * self.rng.standard_normal(size=m.shape)
            meas.append(m)

        num_clutt = self.rng.poisson(clutter_rate)
        for ii in range(0, num_clutt):
            samp = self.rng.standard_normal(size=(2, 1))
            m = np.array([[np.pi], [2000]]) * samp
            meas.append(m)

        return meas

    def proposal_sampling_fnc(self, x, **kwargs):
        cov = self.Q  # proc cov
        mean = np.zeros(cov.shape[0])
        samp = self.rng.multivariate_normal(mean, cov)
        samp = samp.reshape((samp.size, 1))
        return x + self.G @ samp

    def proposal_fnc(self, x_hat, cond, **kwargs):
        cov = self.Q  # proc cov
        x_norm = (self.G.T @ (x_hat - cond)).flatten()
        mean = np.zeros(x_norm.size)
        return stats.multivariate_normal.pdf(x_norm, mean=mean, cov=cov)

    def test_basic_PF(self):
        num_parts = 2000

        means = [np.array([-1500., 0., 250., 0., 0.]).reshape((5, 1)),
                 np.array([-250., 0., 1000., 0., 0.]).reshape((5, 1)),
                 np.array([250., 0., 750., 0., 0.]).reshape((5, 1)),
                 np.array([1000., 0., 1500., 0., 0.]).reshape((5, 1))]
        cov = np.diag(np.array([50, 50, 50, 50, 6 * (np.pi / 180)]))**2
        b_probs = [0.02, 0.02, 0.03, 0.03]
        birth_terms = []
        for (m, p) in zip(means, b_probs):
            distrib = distributions.ParticleDistribution()
            for ii in range(0, num_parts):
                part = distributions.Particle()
                part.point = self.rng.multivariate_normal(m.flatten(),
                                                          cov).reshape(m.shape)
                w = 1 / num_parts
                distrib.add_particle(part, w)

            birth_terms.append((distrib, p))

        filt = filters.ParticleFilter()
        filt.set_meas_model(self.meas_mod)
        filt.dyn_fnc = self.dyn_fnc
        filt.meas_noise = np.diag([self.std_turn**2, self.std_pos**2])
        filt.set_proc_noise(mat=self.G @ self.Q @ self.G.T)
        filt.meas_likelihood_fnc = self.meas_likelihood
        filt.proposal_sampling_fnc = self.proposal_sampling_fnc
        filt.proposal_fnc = self.proposal_fnc

        glmb = tracker.SMCGeneralizedLabeledMultiBernoulli()
        glmb.filter = filt
        self.init_glmb(glmb, birth_terms)

        true_states = []
        total_true = []
        for k in range(0, self.max_time):
            true_states = self.prop_states(k, true_states, noise_on=True)
            total_true.append(deepcopy(true_states))

            # generate measurements
            meas = self.gen_meas(true_states, glmb.clutter_rate)

            # run filter
            glmb.predict(time_step=k)
            glmb.correct(meas=meas)
            glmb.prune()
            glmb.cap()
            glmb.extract_states()

        # f_hndl = glmb.plot_states_labels([0, 2], true_states=total_true,
        #                                  sig_bnd=None)

        # meas_plt_x = []
        # meas_plt_y = []
        # for tt in glmb._meas_tab:
        #     for m in tt:
        #         theta = m[0]
        #         r = m[1]
        #         x = r * np.cos(theta)
        #         y = r * np.sin(theta)
        #         meas_plt_x.append(x)
        #         meas_plt_y.append(y)

        # color = (128/255, 128/255, 128/255)
        # f_hndl.axes[0].scatter(meas_plt_x, meas_plt_y, zorder=-1, alpha=0.35,
        #                        color=color, marker='^')

        # print("Cardinality:")
        # print(glmb.cardinality)
        # print("labels")
        # print(glmb.labels)
        assert glmb.cardinality == 1, "Cardinality does not match"

    def test_UPF(self):
        num_parts = 500
        alpha = 1
        kappa = 0

        means = [np.array([-1500., 0., 250., 0., 0.]).reshape((5, 1)),
                 np.array([-250., 0., 1000., 0., 0.]).reshape((5, 1)),
                 np.array([250., 0., 750., 0., 0.]).reshape((5, 1)),
                 np.array([1000., 0., 1500., 0., 0.]).reshape((5, 1))]
        cov = np.diag(np.array([25, 10, 25, 10, 6 * (np.pi / 180)]))**2
        b_probs = [0.02, 0.02, 0.02, 0.03]
        birth_terms = []
        sigPointRef = distributions.SigmaPoints(alpha=alpha, kappa=kappa,
                                                n=means[0].size)
        sigPointRef.init_weights()
        for (m, p) in zip(means, b_probs):
            distrib = distributions.ParticleDistribution()
            spread = 2 * np.sqrt(np.diag(cov)).reshape(m.shape)
            l_bnd = m - spread / 2
            for ii in range(0, num_parts):
                part = distributions.Particle()
                part.point = l_bnd + spread * self.rng.random(m.shape)
                # part.point = self.rng.multivariate_normal(m.flatten(),
                #                                           cov).reshape(m.shape)
                part.uncertainty = cov.copy()
                part.sigmaPoints = deepcopy(sigPointRef)
                part.sigmaPoints.update_points(m.copy(), cov)
                w = 1 / num_parts
                distrib.add_particle(part, w)

            birth_terms.append((distrib, p))

        filt = filters.UnscentedParticleFilter()
        filt.set_meas_model(self.meas_mod)
        filt.dyn_fnc = self.dyn_fnc
        filt.meas_noise = np.diag([self.std_turn**2, self.std_pos**2])
        filt.set_proc_noise(mat=self.G @ self.Q @ self.G.T)
        filt.meas_likelihood_fnc = self.meas_likelihood
        filt.proposal_sampling_fnc = self.proposal_sampling_fnc
        filt.proposal_fnc = self.proposal_fnc

        glmb = tracker.SMCGeneralizedLabeledMultiBernoulli()
        glmb.filter = filt
        self.init_glmb(glmb, birth_terms)

        true_states = []
        total_true = []
        for k in range(0, self.max_time):
            true_states = self.prop_states(k, true_states, noise_on=True)
            total_true.append(deepcopy(true_states))

            # generate measurements
            meas = self.gen_meas(true_states, glmb.clutter_rate)

            # run filter
            glmb.predict(time_step=k)
            glmb.correct(meas=meas)
            glmb.prune()
            glmb.cap()
            glmb.extract_states()

        print("Cardinality:")
        print(glmb.cardinality)
        print("labels")
        print(glmb.labels)

        # f_hndl = glmb.plot_states_labels([0, 2], true_states=total_true,
        #                                  sig_bnd=None)

        # meas_plt_x = []
        # meas_plt_y = []
        # for tt in glmb._meas_tab:
        #     for m in tt:
        #         theta = m[0]
        #         r = m[1]
        #         x = r * np.cos(theta)
        #         y = r * np.sin(theta)
        #         meas_plt_x.append(x)
        #         meas_plt_y.append(y)

        # color = (128/255, 128/255, 128/255)
        # f_hndl.axes[0].scatter(meas_plt_x, meas_plt_y, zorder=-1, alpha=0.35,
        #                        color=color, marker='^')

        # glmb.plot_card_dist()
        assert glmb.cardinality == 1, "Cardinality does not match"

    def test_UPFMCMC(self):
        num_parts = 300
        alpha = 1
        kappa = 0

        means = [np.array([-1500., 0., 250., 0., 0.]).reshape((5, 1)),
                 np.array([-250., 0., 1000., 0., 0.]).reshape((5, 1)),
                 np.array([250., 0., 750., 0., 0.]).reshape((5, 1)),
                 np.array([1000., 0., 1500., 0., 0.]).reshape((5, 1))]
        cov = np.diag(np.array([25, 10, 25, 10, 6 * (np.pi / 180)]))**2
        b_probs = [0.02, 0.02, 0.02, 0.03]
        birth_terms = []
        sigPointRef = distributions.SigmaPoints(alpha=alpha, kappa=kappa,
                                                n=means[0].size)
        sigPointRef.init_weights()
        for (m, p) in zip(means, b_probs):
            distrib = distributions.ParticleDistribution()
            spread = 2 * np.sqrt(np.diag(cov)).reshape(m.shape)
            l_bnd = m - spread / 2
            for ii in range(0, num_parts):
                part = distributions.Particle()
                part.point = l_bnd + spread * self.rng.random(m.shape)
                part.uncertainty = cov.copy()
                part.sigmaPoints = deepcopy(sigPointRef)
                part.sigmaPoints.update_points(m.copy(), cov)
                w = 1 / num_parts
                distrib.add_particle(part, w)

            birth_terms.append((distrib, p))

        filt = filters.UnscentedParticleFilter()
        filt.use_MCMC = True
        filt.set_meas_model(self.meas_mod)
        filt.dyn_fnc = self.dyn_fnc
        filt.meas_noise = np.diag([self.std_turn**2, self.std_pos**2])
        filt.set_proc_noise(mat=self.G @ self.Q @ self.G.T)
        filt.meas_likelihood_fnc = self.meas_likelihood
        filt.proposal_sampling_fnc = self.proposal_sampling_fnc
        filt.proposal_fnc = self.proposal_fnc

        glmb = tracker.SMCGeneralizedLabeledMultiBernoulli()
        glmb.filter = filt
        self.init_glmb(glmb, birth_terms)

        true_states = []
        total_true = []
        for k in range(0, self.max_time):
            print(k)
            true_states = self.prop_states(k, true_states, noise_on=True)
            total_true.append(deepcopy(true_states))

            # generate measurements
            meas = self.gen_meas(true_states, glmb.clutter_rate)

            # run filter
            glmb.predict(time_step=k)
            glmb.correct(meas=meas)
            glmb.prune()
            glmb.cap()
            glmb.extract_states()

        assert glmb.cardinality == 1, "Cardinality does not match"
