import pytest
import numpy as np
import numpy.testing as test
import numpy.random as rnd
import scipy.stats as stats
from copy import deepcopy
import scipy.linalg as la

import gasur.swarm_estimator.tracker as tracker
import gasur.utilities.distributions as distributions
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

    proc_noise_scale = 0.01 * np.eye(4)
    meas_noise_scale = 0.5**2 * np.eye(2)

    filt = filters.StudentsTFilter()
    filt.set_state_mat(mat=state_mat)
    filt.set_proc_noise(mat=proc_noise_scale)

    filt.set_meas_mat(mat=meas_mat)
    filt.meas_noise = meas_noise_scale

    filt.meas_noise_dof = meas_noise_dof
    filt.proc_noise_dof = proc_noise_dof

    glmb = tracker.STMGeneralizedLabeledMultiBernoulli()
    glmb.filter = filt
    glmb.prob_detection = 0.98
    glmb.prob_survive = 0.99

    dof = 3
    mu = [np.array([-1500., 250., 0, 0]).reshape((4, 1))]
    scale = [dof / (dof - 2) * np.diag(np.array([50, 50, 10, 10]))**2]
    # scale = [np.diag(np.array([50, 50, 10, 10]))**2]
    stm0 = distributions.StudentsTMixture(means=mu, scalings=scale,
                                          weights=[1], dof=dof)
    mu = [np.array([-250., 1000., 0, 0.]).reshape((4, 1))]
    stm1 = distributions.StudentsTMixture(means=mu, scalings=scale,
                                          weights=[1], dof=dof)
    mu = [np.array([250., 750., 0, 0]).reshape((4, 1))]
    stm2 = distributions.StudentsTMixture(means=mu, scalings=scale,
                                          weights=[1], dof=dof)
    mu = [np.array([1000.,  1500., 0, 0]).reshape((4, 1))]
    stm3 = distributions.StudentsTMixture(means=mu, scalings=scale,
                                          weights=[1], dof=dof)

    glmb.birth_terms = [(stm0, 0.02), (stm1, 0.02), (stm2, 0.03), (stm3, 0.03)]
    glmb.req_births = 5
    glmb.req_surv = 3000
    glmb.req_upd = 3000
    glmb.gating_on = False
    glmb.inv_chi2_gate = 32.2361913029694
    glmb.clutter_rate = 0.0005
    glmb.clutter_den = 1 / np.prod(pos_bnds[:, 1] - pos_bnds[:, 0])
    glmb.save_covs = True

    true_states = []
    for tt in range(0, 15):
        glmb.predict(time_step=tt)

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

        for x in true_states:
            m = meas_mat @ x
            s_norm = rng.multivariate_normal(np.zeros(m.size),
                                             meas_noise_scale)
            s_norm = s_norm.reshape(m.shape)
            s_chi = rng.chisquare(meas_noise_dof)
            meas.append(m + s_norm * np.sqrt(meas_noise_dof / s_chi))

        for ii in range(0, len(true_states)):
            x = true_states[ii]
            s_norm = rng.multivariate_normal(np.zeros(x.size),
                                             proc_noise_scale)
            s_norm = s_norm.reshape(x.shape)
            s_chi = rng.chisquare(proc_noise_dof)
            x = state_mat @ x + (s_norm * np.sqrt(proc_noise_dof / s_chi))
            true_states[ii] = x

        num_clutt = rng.poisson(glmb.clutter_rate)
        for ii in range(0, num_clutt):
            m = pos_bnds[:, [0]] + (pos_bnds[:, [1]] - pos_bnds[:, [0]]) \
                * rng.random((pos_bnds.shape[0], 1))
            meas.append(m)

        glmb.correct(meas=meas)
        glmb.prune()
        glmb.cap()

    glmb.extract_states()
    assert glmb.cardinality == 4, "Cardinality does not match"


def test_SMCGeneralizedLabeledMultiBernoulli():
    rng = rnd.default_rng(1)

    max_time = 10
    dt = 1.0
    num_parts = 1000
    sig_w = 5.0
    sig_u = np.pi / 180
    std_turn = 2 * (np.pi / 180)
    std_pos = 10.0
    prob_detection = 0.98
    prob_survive = 0.99

    G = np.array([[dt**2 / 2, 0, 0],
                  [dt, 0, 0],
                  [0, dt**2 / 2, 0],
                  [0, dt, 0],
                  [0, 0, 1]])
    Q = la.block_diag(sig_w**2 * np.eye(2), np.array([[sig_u**2]]))

    def compute_prob_detection(part_lst, **kwargs):
        if len(part_lst) == 0:
            return np.array([])
        else:
            inv_std = np.diag(np.array([1. / 2000., 1. / 2000.]))

            e_sq = np.sum(np.hstack([(inv_std
                                      @ x[[0, 2], 0].reshape((2, 1)))**2
                                     for x in part_lst]), axis=0)
            return prob_detection * np.exp(-e_sq / 2.)

    def compute_prob_survive(part_lst, **kwargs):
        if len(part_lst) == 0:
            return np.array([])
        else:
            return prob_survive * np.ones(len(part_lst))

    def meas_mod(state, **kwargs):
        z1 = np.arctan2(state[2, 0], state[0, 0])
        z2 = np.sqrt(np.sum(state.flatten()**2))
        return np.array([[z1], [z2]])

    def meas_likelihood(meas, est, **kwargs):
        cov = np.array([[std_turn**2, 0],
                        [0, std_pos**2]])
        return stats.multivariate_normal.pdf(meas.copy().reshape(meas.size),
                                             mean=est.copy().reshape(est.size),
                                             cov=cov)

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

    # \dot{x} = f(x)
    def cont_dyn(x, **kwargs):
        out = np.zeros(x.shape)
        for ii, f in enumerate([f0, f1, f2, f3, f4]):
            out[ii] = f(x, **kwargs)
        return out

    # x_{k + 1} = f(x_{k})
    def dyn_fnc(x, noise_on=True, **kwargs):
        ctrl = np.zeros((2, 1))
        ns = rk4(cont_dyn, x.copy(), dt, cur_input=ctrl)
        if noise_on:
            ns += G @ Q @ G.T @ rng.standard_normal(ns.shape)
        return ns

    means = [np.array([-1500., 0., 250., 0., 0.]).reshape((5, 1)),
             np.array([-250., 0., 1000., 0., 0.]).reshape((5, 1)),
             np.array([250., 0., 750., 0., 0.]).reshape((5, 1)),
             np.array([1000., 0., 1500., 0., 0.]).reshape((5, 1))]
    cov = np.diag(np.array([50, 50, 50, 50, 6 * (np.pi / 180)]))**2
    b_probs = [0.02, 0.02, 0.03, 0.03]
    birth_terms = []
    for (m, p) in zip(means, b_probs):
        parts = [rng.multivariate_normal(m.flatten(),
                                         cov).reshape(m.shape)
                 for ii in range(0, num_parts)]
        weights = [1 / num_parts] * num_parts
        distrib = distributions.ParticleDistribution(particles=parts,
                                                     weights=weights)
        birth_terms.append((distrib, p))

    filt = filters.ParticleFilter()
    filt.set_meas_model(meas_mod)
    filt.dyn_fnc = dyn_fnc
    filt.meas_noise = np.diag([std_turn**2, std_pos**2])
    filt.set_proc_noise(mat=G @ Q @ G.T)
    filt.meas_likelihood_fnc = meas_likelihood

    glmb = tracker.SMCGeneralizedLabeledMultiBernoulli()
    glmb.filter = filt
    glmb.compute_prob_detection = compute_prob_detection
    glmb.compute_prob_survive = compute_prob_survive
    glmb.prob_detection = prob_detection
    glmb.prob_survive = prob_survive
    glmb.birth_terms = birth_terms
    glmb.req_births = 5
    glmb.req_surv = 5000
    glmb.req_upd = 5000
    glmb.gating_on = False
    glmb.clutter_rate = 0.0000001  # 10
    glmb.clutter_den = 1 / (np.pi * 2000)

    def prop_states(k, true_states, dt, noise_on=True):
        new_states = []
        for s in true_states:
            ns = dyn_fnc(s.copy(), noise_on=noise_on)
            new_states.append(ns)

        wturn = 2 * np.pi / 180
        if k == 0:
            s = np.array([1000 + 3.8676, -10, 1500 - 11.7457, -10, wturn / 8])
            new_states.append(s.reshape((5, 1)))

        return new_states

    def gen_meas(true_states):
        meas = []
        for s in true_states:
            if rng.random() <= compute_prob_detection([s]):
                m = meas_mod(s)
                m += np.array([[std_turn], [std_pos]]) \
                    * rng.standard_normal(size=m.shape)
                meas.append(m)

        num_clutt = rng.poisson(glmb.clutter_rate)
        for ii in range(0, num_clutt):
            m = np.array([[np.pi], [2000]]) * rng.standard_normal(size=(2, 1))
            meas.append(m)

        return meas

    true_states = []
    total_true = []
    for k in range(0, max_time):
        print(k)
        true_states = prop_states(k, true_states, dt, noise_on=True)
        total_true.append(deepcopy(true_states))

        # generate measurements
        meas = gen_meas(true_states)

        # run filter
        glmb.predict(time_step=k, dt=dt)
        glmb.correct(meas=meas)
        glmb.prune()
        glmb.cap()

    glmb.extract_states(dt=dt)
    assert glmb.cardinality == 1, "Cardinality does not match"

    # glmb.plot_states_labels([0, 2], true_states=total_true)
