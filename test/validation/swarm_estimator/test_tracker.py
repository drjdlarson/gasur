import sys
import pytest
import numpy as np
import numpy.random as rnd
import matplotlib.pyplot as plt
from copy import deepcopy

import scipy.stats as stats

import gncpy.filters as gfilts
import gncpy.dynamics as gdyn
import gncpy.distributions as gdistrib
import gasur.swarm_estimator.tracker as tracker
from gasur.utilities.distributions import GaussianMixture, StudentsTMixture


global_seed = 69
debug_plots = False

_meas_mat = np.array([[1, 0, 0, 0],
                      [0, 1, 0, 0]])


def _state_mat_fun(t, dt, useless):
    # print('got useless arg: {}'.format(useless))
    return np.array([[1., 0, dt, 0],
                     [0., 1., 0, dt],
                     [0, 0, 1., 0],
                     [0, 0, 0, 1]])


def _meas_mat_fun(t, useless):
    # print('got useless arg: {}'.format(useless))
    return _meas_mat


def _meas_mat_fun_nonlin(t, x, *args):
    return _meas_mat @ x


def _setup_double_int_kf(dt):

    m_noise = 0.02
    p_noise = 0.2

    filt = gfilts.KalmanFilter()
    filt.set_state_model(state_mat_fun=_state_mat_fun)
    filt.set_measurement_model(meas_fun=_meas_mat_fun)
    filt.proc_noise = gdyn.DoubleIntegrator().get_dis_process_noise_mat(dt, np.array([[p_noise**2]]))
    filt.meas_noise = m_noise**2 * np.eye(2)

    return filt


def _setup_double_int_stf(dt):
    m_noise = 0.02
    p_noise = 0.2

    filt = gfilts.StudentsTFilter()
    filt.set_state_model(state_mat_fun=_state_mat_fun)
    filt.set_measurement_model(meas_fun=_meas_mat_fun)
    filt.proc_noise = gdyn.DoubleIntegrator().get_dis_process_noise_mat(dt, np.array([[p_noise**2]]))
    filt.meas_noise = m_noise**2 * np.eye(2)

    filt.meas_noise_dof = 3
    filt.proc_noise_dof = 3
    # Note filt.dof is determined by the birth terms

    return filt


def _setup_double_int_pf(dt, rng):
    m_noise = 0.02
    p_noise = 0.2

    doubleInt = gdyn.DoubleIntegrator()
    proc_noise = doubleInt.get_dis_process_noise_mat(dt, np.array([[p_noise**2]]))

    def meas_likelihood(meas, est, *args):
        return stats.multivariate_normal.pdf(meas.flatten(), mean=est.flatten(),
                                             cov=m_noise**2 * np.eye(2))

    # def proposal_sampling_fnc(x, rng):
    #     noise = p_noise * np.array([0, 0, 1, 1]) * rng.standard_normal(4)
    #     return x + noise.reshape((4, 1))

    def proposal_sampling_fnc(x, rng):  # noqa
        val = rng.multivariate_normal(x.flatten(), proc_noise).reshape(x.shape)
        return val

    def transition_prob_fnc(x_hat, mean, *args):
        return stats.multivariate_normal.pdf(x_hat.flatten(), mean.flatten(),
                                             proc_noise, True)

    def proposal_fnc(x_hat, mean, y, *args):
        return 1
        # return stats.multivariate_normal.pdf(x_hat.flatten(), mean.flatten(),
        #                                      proc_noise, True)

    filt = gfilts.ParticleFilter(rng=rng)
    filt.set_state_model(dyn_obj=doubleInt)
    filt.set_measurement_model(meas_fun=_meas_mat_fun_nonlin)

    filt.proc_noise = proc_noise.copy()
    filt.meas_noise = m_noise**2 * np.eye(2)

    filt.meas_likelihood_fnc = meas_likelihood
    filt.proposal_sampling_fnc = proposal_sampling_fnc
    filt.proposal_fnc = proposal_fnc
    # filt.transition_prob_fnc = transition_prob_fnc

    return filt


def _setup_double_int_upf(dt, rng, use_MCMC):
    m_noise = 0.02
    p_noise = 0.2

    doubleInt = gdyn.DoubleIntegrator()

    filt = gfilts.UnscentedParticleFilter(use_MCMC=use_MCMC, rng=rng)
    filt.use_cholesky_inverse = False

    filt.set_state_model(dyn_obj=doubleInt)
    filt.set_measurement_model(meas_mat=_meas_mat.copy())

    proc_noise = doubleInt.get_dis_process_noise_mat(dt, np.array([[p_noise**2]]))
    meas_noise = m_noise**2 * np.eye(2)
    filt.proc_noise = proc_noise.copy()
    filt.meas_noise = meas_noise.copy()

    return filt


def __gsm_import_dist_factory():
    def import_dist_fnc(parts, rng):
        new_parts = np.nan * np.ones(parts.particles.shape)

        disc = 0.99
        a = (3 * disc - 1) / (2 * disc)
        h = np.sqrt(1 - a**2)
        last_means = np.mean(parts.particles, axis=0)
        means = a * parts.particles[:, 0:2] + (1 - a) * last_means[0:2]

        # df, sig
        for ind in range(means.shape[1]):
            std = np.sqrt(h**2 * np.cov(parts.particles[:, ind]))

            for ii, m in enumerate(means):
                new_parts[ii, ind] = stats.norm.rvs(loc=m[ind], scale=std,
                                                    random_state=rng)

        for ii in range(new_parts.shape[0]):
            new_parts[ii, 2] = stats.invgamma.rvs(np.abs(new_parts[ii, 0]) / 2,
                                                  scale=1 / (2 / np.abs(new_parts[ii, 0])),
                                                  random_state=rng)

        return new_parts

    return import_dist_fnc


def _setup_qkf(dt, use_sqkf, m_vars):
    state_mat = np.vstack((np.hstack((np.eye(2), dt * np.eye(2), dt**2 / 2 * np.eye(2))),
                          np.hstack((np.zeros((2, 2)), np.eye(2), dt * np.eye(2))),
                          np.hstack((np.zeros((2, 2)), np.zeros((2, 2)), np.eye(2)))))
    proc_cov = np.diag((4, 4, 4, 4, 0.01, 0.01))

    def meas_fun(t, x, *args):
        return np.array([[np.sqrt(x[0, 0]**2 + x[1, 0]**2)],
                         [np.arctan2(x[1, 0], x[0, 0])]])

    # define base GSM parameters
    if use_sqkf:
        filt = gfilts.SquareRootQKF()
    else:
        filt = gfilts.QuadratureKalmanFilter()

    filt.set_state_model(state_mat=state_mat)
    filt.set_measurement_model(meas_fun=meas_fun)
    filt.proc_noise = proc_cov
    filt.meas_noise = np.diag(m_vars)

    filt.points_per_axis = 3

    return filt, state_mat, meas_fun


def _setup_ukf(dt, m_vars):
    state_mat = np.vstack((np.hstack((np.eye(2), dt * np.eye(2), dt**2 / 2 * np.eye(2))),
                          np.hstack((np.zeros((2, 2)), np.eye(2), dt * np.eye(2))),
                          np.hstack((np.zeros((2, 2)), np.zeros((2, 2)), np.eye(2)))))
    proc_cov = np.diag((4, 4, 4, 4, 0.01, 0.01))

    def meas_fun(t, x, *args):
        return np.vstack((meas_fun_range(t, x),
                          meas_fun_bearing(t, x)))

    def meas_fun_range(t, x, *args):
        return np.array([np.sqrt(x[0, 0]**2 + x[1, 0]**2)])

    def meas_fun_bearing(t, x, *args):
        return np.array([np.arctan2(x[1, 0], x[0, 0])])

    # define filter parameters
    filt = gfilts.UnscentedKalmanFilter()

    filt.set_state_model(state_mat=state_mat)
    filt.set_measurement_model(meas_fun_lst=[meas_fun_range, meas_fun_bearing])
    filt.proc_noise = proc_cov
    filt.meas_noise = np.diag(m_vars)

    # define UKF specific parameters
    filt.alpha = 0.5
    filt.kappa = 1
    filt.beta = 1.5

    return filt, state_mat, meas_fun


def __gsm_import_w_factory(inov_cov):
    def import_w_fnc(meas, parts):
        stds = np.sqrt(parts[:, 2] * parts[:, 1]**2 + inov_cov)
        return np.array([stats.norm.pdf(meas.item(), scale=scale)
                         for scale in stds])

    return import_w_fnc


def _setup_qkf_gsm(dt, rng, use_sqkf, m_dfs, m_vars):
    state_mat = np.vstack((np.hstack((np.eye(2), dt * np.eye(2), dt**2 / 2 * np.eye(2))),
                          np.hstack((np.zeros((2, 2)), np.eye(2), dt * np.eye(2))),
                          np.hstack((np.zeros((2, 2)), np.zeros((2, 2)), np.eye(2)))))
    proc_cov = np.diag((4, 4, 4, 4, 0.01, 0.01))

    def meas_fun(t, x, *args):
        return np.array([[np.sqrt(x[0, 0]**2 + x[1, 0]**2)],
                         [np.arctan2(x[1, 0], x[0, 0])]])

    # define base GSM parameters
    if use_sqkf:
        filt = gfilts.SQKFGaussianScaleMixtureFilter()
    else:
        filt = gfilts.QKFGaussianScaleMixtureFilter()

    filt.set_state_model(state_mat=state_mat)
    filt.proc_noise = proc_cov
    filt.set_measurement_model(meas_fun=meas_fun)

    # define measurement noise filters
    num_parts = 500
    bootstrap_lst = [None] * 2

    # manually setup each bootstrap filter (stripped down PF)
    for ind in range(len(bootstrap_lst)):
        mf = gfilts.BootstrapFilter()
        mf.importance_dist_fnc = __gsm_import_dist_factory()
        mf.particleDistribution = gdistrib.SimpleParticleDistribution()
        df_particles = stats.uniform.rvs(loc=1, scale=4, size=num_parts,
                                         random_state=rng)
        sig_particles = stats.uniform.rvs(loc=0, scale=5 * np.sqrt(m_vars[ind]),
                                          size=num_parts, random_state=rng)
        z_particles = np.nan * np.ones(num_parts)
        for ii, v in enumerate(df_particles):
            z_particles[ii] = stats.invgamma.rvs(v / 2, scale=1 / (2 / v),
                                                 random_state=rng)
        mf.particleDistribution.particles = np.stack((df_particles, sig_particles,
                                                      z_particles), axis=1)

        mf.particleDistribution.num_parts_per_ind = np.ones(num_parts)
        mf.particleDistribution.weights = 1 / num_parts * np.ones(num_parts)
        mf.rng = rng
        bootstrap_lst[ind] = mf

    importance_weight_factory_lst = [__gsm_import_w_factory] * len(bootstrap_lst)
    filt.set_meas_noise_model(bootstrap_lst=bootstrap_lst,
                              importance_weight_factory_lst=importance_weight_factory_lst)

    # define QKF specific parameters for core filter
    filt.points_per_axis = 3

    return filt, state_mat, meas_fun


def _setup_ukf_gsm(dt, rng, m_dfs, m_vars):
    state_mat = np.vstack((np.hstack((np.eye(2), dt * np.eye(2), dt**2 / 2 * np.eye(2))),
                          np.hstack((np.zeros((2, 2)), np.eye(2), dt * np.eye(2))),
                          np.hstack((np.zeros((2, 2)), np.zeros((2, 2)), np.eye(2)))))
    proc_cov = np.diag((4, 4, 4, 4, 0.01, 0.01))

    def meas_fun(t, x, *args):
        return np.vstack((meas_fun_range(t, x),
                          meas_fun_bearing(t, x)))

    def meas_fun_range(t, x, *args):
        return np.array([np.sqrt(x[0, 0]**2 + x[1, 0]**2)])

    def meas_fun_bearing(t, x, *args):
        return np.array([np.arctan2(x[1, 0], x[0, 0])])

    # define base GSM parameters
    filt = gfilts.UKFGaussianScaleMixtureFilter()

    filt.set_state_model(state_mat=state_mat)
    filt.proc_noise = proc_cov
    filt.set_measurement_model(meas_fun_lst=[meas_fun_range, meas_fun_bearing])

    # define measurement noise filters
    num_parts = 500
    bootstrap_lst = [None] * 2

    # manually setup each bootstrap filter (stripped down PF)
    for ind in range(len(bootstrap_lst)):
        mf = gfilts.BootstrapFilter()
        mf.importance_dist_fnc = __gsm_import_dist_factory()
        mf.particleDistribution = gdistrib.SimpleParticleDistribution()
        df_particles = stats.uniform.rvs(loc=1, scale=4, size=num_parts,
                                         random_state=rng)
        sig_particles = stats.uniform.rvs(loc=0, scale=5 * np.sqrt(m_vars[ind]),
                                          size=num_parts, random_state=rng)
        z_particles = np.nan * np.ones(num_parts)
        for ii, v in enumerate(df_particles):
            z_particles[ii] = stats.invgamma.rvs(v / 2, scale=1 / (2 / v),
                                                 random_state=rng)
        mf.particleDistribution.particles = np.stack((df_particles, sig_particles,
                                                      z_particles), axis=1)

        mf.particleDistribution.num_parts_per_ind = np.ones(num_parts)
        mf.particleDistribution.weights = 1 / num_parts * np.ones(num_parts)
        mf.rng = rng
        bootstrap_lst[ind] = mf

    importance_weight_factory_lst = [__gsm_import_w_factory] * len(bootstrap_lst)
    filt.set_meas_noise_model(bootstrap_lst=bootstrap_lst,
                              importance_weight_factory_lst=importance_weight_factory_lst)

    # define UKF specific parameters for core filter
    filt.alpha = 0.5
    filt.kappa = 1
    filt.beta = 1.5

    return filt, state_mat, meas_fun


def _setup_phd_double_int_birth():
    mu = [np.array([10., 0., 0., 0.]).reshape((4, 1))]
    cov = [np.diag(np.array([1, 1, 1, 1]))**2]
    gm0 = GaussianMixture(means=mu, covariances=cov, weights=[1])

    return [gm0, ]


def _setup_gm_glmb_double_int_birth():
    mu = [np.array([10., 0., 0., 1.]).reshape((4, 1))]
    cov = [np.diag(np.array([1, 1, 1, 1]))**2]
    gm0 = GaussianMixture(means=mu, covariances=cov, weights=[1])

    # return [(gm0, 0.03), ]
    return [(gm0, 0.003), ]


def _setup_stm_glmb_double_int_birth():
    mu = [np.array([10., 0., 0., 1.]).reshape((4, 1))]
    scale = [np.diag(np.array([1, 1, 1, 1]))**2]
    stm0 = StudentsTMixture(means=mu, scalings=scale, weights=[1])

    return [(stm0, 0.003), ]


def _setup_smc_glmb_double_int_birth(num_parts, rng):
    means = [np.array([10., 0., 0., 2.]).reshape((4, 1))]
    cov = np.diag(np.array([1, 1, 1, 1]))**2
    b_probs = [0.003, ]

    birth_terms = []
    for (m, p) in zip(means, b_probs):
        distrib = gdistrib.ParticleDistribution()
        spread = 2 * np.sqrt(np.diag(cov)).reshape(m.shape)
        l_bnd = m - spread / 2
        for ii in range(0, num_parts):
            part = gdistrib.Particle()
            part.point = l_bnd + spread * rng.random(m.shape)
            w = 1 / num_parts
            distrib.add_particle(part, w)

        birth_terms.append((distrib, p))

    return birth_terms


def _setup_usmc_glmb_double_int_birth(num_parts, rng):
    # means = [np.array([10., 0., 0., 2.]).reshape((4, 1))]
    means = [np.array([20, 80, 3, -3]).reshape((4, 1))]
    # cov = np.diag(np.array([1, 1, 1, 1]))**2
    cov = np.diag([3**2, 5**2, 2**2, 1])
    b_probs = [0.005, ]
    alpha = 10**-3
    kappa = 0

    birth_terms = []
    for (m, p) in zip(means, b_probs):
        distrib = gdistrib.ParticleDistribution()
        spread = 2 * np.sqrt(np.diag(cov)).reshape(m.shape)
        l_bnd = m - spread / 2
        for ii in range(0, num_parts):
            part = gdistrib.Particle()
            part.point = l_bnd + spread * rng.random(m.shape)
            part.uncertainty = cov.copy()
            part.sigmaPoints = gdistrib.SigmaPoints(alpha=alpha, kappa=kappa)
            part.sigmaPoints.update_points(part.point, part.uncertainty)
            part.sigmaPoints.init_weights()
            distrib.add_particle(part, 1 / num_parts)

        birth_terms.append((distrib, p))

    return birth_terms


def _setup_gsm_birth():
    # note: GSM filter assumes noise is conditionally Gaussian so use GM with 1 term for birth
    means = [np.array([2000, 2000, 20, 20, 0, 0]).reshape((6, 1))]
    cov = [np.diag((5 * 10**4, 5 * 10**4, 8, 8, 0.02, 0.02))]
    gm0 = GaussianMixture(means=means, covariances=cov, weights=[1])

    return [(gm0, 0.05), ]


def _gen_meas(tt, true_agents, proc_noise, meas_noise, rng):
    meas_in = []
    for x in true_agents:
        xp = rng.multivariate_normal(x.flatten(), proc_noise).reshape(x.shape)
        meas = _meas_mat @ xp
        m = rng.multivariate_normal(meas.flatten(), meas_noise).reshape(meas.shape)
        meas_in.append(m.copy())

    return meas_in


def _gen_meas_stf(tt, true_agents, proc_noise, p_df, meas_noise, m_df, rng):
    meas_in = []
    for x in true_agents:
        xp = stats.multivariate_t.rvs(df=p_df, loc=x.flatten(),
                                      shape=proc_noise,
                                      random_state=rng).reshape(x.shape)
        meas = _meas_mat @ xp
        m = stats.multivariate_t.rvs(df=m_df, loc=meas.flatten(),
                                      shape=meas_noise,
                                      random_state=rng).reshape(meas.shape)
        meas_in.append(m.copy())

    return meas_in


def _gen_meas_qkf(tt, true_agents, proc_noise, meas_fun, m_vars, rng):
    meas_out = []
    for x in true_agents:
        xp = rng.multivariate_normal(x.flatten(), proc_noise).reshape(x.shape)
        meas = meas_fun(tt, xp)
        for ii, var in enumerate(m_vars):
            meas[ii, 0] += stats.norm.rvs(scale=np.sqrt(var), random_state=rng)
        meas_out.append(meas.copy())

    return meas_out


def _gen_meas_gsm(tt, true_agents, proc_noise, meas_fun, m_dfs, m_vars, rng):
    meas_out = []
    for x in true_agents:
        xp = rng.multivariate_normal(x.flatten(), proc_noise).reshape(x.shape)
        meas = meas_fun(tt, xp)
        for ii, (df, var) in enumerate(zip(m_dfs, m_vars)):
            meas[ii, 0] += stats.t.rvs(df, scale=np.sqrt(var), random_state=rng)
        meas_out.append(meas.copy())

    return meas_out


def _prop_true(true_agents, tt, dt):
    out = []
    for ii, x in enumerate(true_agents):
        out.append(_state_mat_fun(tt, dt, 'useless') @ x)

    return out


def _update_true_agents(true_agents, tt, dt, b_model, rng):
    out = _prop_true(true_agents, tt, dt)

    if any(np.abs(tt - np.array([0, 1, 1.5])) < 1e-8):
        x = b_model[0].means[0] + (rng.standard_normal(4) * np.ones(4)).reshape((4, 1))
        out.append(x.copy())

    return out


def _update_true_agents_prob(true_agents, tt, dt, b_model, rng):
    out = _prop_true(true_agents, tt, dt)

    p = rng.uniform()
    for gm, w in b_model:
        if p <= w:
            print('birth at {:.2f}'.format(tt))
            x = gm.means[0] + (1 * rng.standard_normal(4)).reshape((4, 1))
            out.append(x.copy())

    return out


def _update_true_agents_prob_smc(true_agents, tt, dt, b_model, rng):
    out = []
    doubleInt = gdyn.DoubleIntegrator()
    for ii, x in enumerate(true_agents):
        out.append(doubleInt.get_state_mat(tt, dt) @ x)

    if any(np.abs(tt - np.array([0.5])) < 1e-8):
        for distrib, w in b_model:
            print('birth at {:.2f}'.format(tt))
            inds = np.arange(0, len(distrib.particles))
            ii = rnd.choice(inds, p=distrib.weights)
            out.append(distrib.particles[ii].copy())

    return out


def _update_true_agents_prob_usmc(true_agents, tt, dt, b_model, rng):
    out = []
    doubleInt = gdyn.DoubleIntegrator()
    for ii, x in enumerate(true_agents):
        out.append(doubleInt.get_state_mat(tt, dt) @ x)

    if any(np.abs(tt - np.array([0.5])) < 1e-8):
        for distrib, w in b_model:
            print('birth at {:.2f}'.format(tt))
            out.append(distrib.mean.copy())

    return out


def _update_true_agents_gsm(true_agents, tt, b_model, rng, state_mat):
    out = []
    for existing in true_agents:
        out.append(state_mat @ existing)

    if any(np.abs(tt - np.array([5])) < 1e-8):
        print('birth at {:.2f}'.format(tt))
        gm = b_model[0][0]
        out.append(gm.means[0].copy().reshape(gm.means[0].shape))

    return out


def test_PHD():  # noqa
    print('Test PHD')

    rng = rnd.default_rng(global_seed)

    dt = 0.01
    t0, t1 = 0, 10 + dt

    filt = _setup_double_int_kf(dt)
    state_mat_args = (dt, 'test arg')
    meas_fun_args = ('useless arg', )

    b_model = _setup_phd_double_int_birth()

    RFS_base_args = {'prob_detection': 0.99, 'prob_survive': 0.98,
                     'in_filter': filt, 'birth_terms': b_model,
                     'clutter_den': 1**-7, 'clutter_rate': 1**-7}
    phd = tracker.ProbabilityHypothesisDensity(**RFS_base_args)
    phd.gating_on = False

    time = np.arange(t0, t1, dt)
    true_agents = []
    global_true = []
    for kk, tt in enumerate(time):

        true_agents = _update_true_agents(true_agents, tt, dt, b_model, rng)
        global_true.append(deepcopy(true_agents))

        filt_args = {'state_mat_args': state_mat_args}
        phd.predict(tt, filt_args=filt_args)

        meas_in = _gen_meas(tt, true_agents, filt.proc_noise, filt.meas_noise, rng)

        filt_args = {'meas_fun_args': meas_fun_args}
        phd.correct(tt, meas_in, meas_mat_args={}, est_meas_args={},
                    filt_args=filt_args)

        phd.cleanup()

    phd.calculate_ospa(global_true, 2, 1)

    if debug_plots:
        phd.plot_states([0, 1])
        phd.plot_ospa_history(time=time, time_units='s')

    assert len(true_agents) == phd.cardinality, 'Wrong cardinality'


def test_CPHD():  # noqa
    print('Test CPHD')

    rng = rnd.default_rng(global_seed)

    dt = 0.01
    t0, t1 = 0, 10 + dt

    filt = _setup_double_int_kf(dt)
    state_mat_args = (dt, 'test arg')
    meas_fun_args = ('useless arg', )

    b_model = _setup_phd_double_int_birth()

    RFS_base_args = {'prob_detection': 0.99, 'prob_survive': 0.98,
                     'in_filter': filt, 'birth_terms': b_model,
                     'clutter_den': 1**-7, 'clutter_rate': 1**-7}
    phd = tracker.CardinalizedPHD(**RFS_base_args)
    phd.gating_on = False

    time = np.arange(t0, t1, dt)
    true_agents = []
    global_true = []
    for kk, tt in enumerate(time):

        true_agents = _update_true_agents(true_agents, tt, dt, b_model, rng)
        global_true.append(deepcopy(true_agents))

        filt_args = {'state_mat_args': state_mat_args}
        phd.predict(tt, filt_args=filt_args)

        meas_in = _gen_meas(tt, true_agents, filt.proc_noise, filt.meas_noise, rng)

        filt_args = {'meas_fun_args': meas_fun_args}
        phd.correct(tt, meas_in, meas_mat_args={}, est_meas_args={},
                    filt_args=filt_args)

        phd.cleanup()

    phd.calculate_ospa(global_true, 2, 1)

    if debug_plots:
        phd.plot_card_time_hist(time_vec=time)
        phd.plot_states([0, 1])
        phd.plot_ospa_history(time=time, time_units='s')

    assert len(true_agents) == phd.cardinality, 'Wrong cardinality'


def test_GLMB():  # noqa
    print('Test GM-GLMB')

    rng = rnd.default_rng(global_seed)

    dt = 0.01
    t0, t1 = 0, 6 + dt

    filt = _setup_double_int_kf(dt)
    state_mat_args = (dt, 'test arg')
    meas_fun_args = ('useless arg', )

    b_model = _setup_gm_glmb_double_int_birth()

    RFS_base_args = {'prob_detection': 0.99, 'prob_survive': 0.98,
                     'in_filter': filt, 'birth_terms': b_model,
                     'clutter_den': 1**-7, 'clutter_rate': 1**-7}
    GLMB_args = {'req_births': len(b_model) + 1, 'req_surv': 1000,
                 'req_upd': 800, 'prune_threshold': 10**-5, 'max_hyps': 1000}
    glmb = tracker.GeneralizedLabeledMultiBernoulli(**GLMB_args, **RFS_base_args)

    time = np.arange(t0, t1, dt)
    true_agents = []
    global_true = []
    print('\tStarting sim')
    for kk, tt in enumerate(time):
        if np.mod(kk, 100) == 0:
            print('\t\t{:.2f}'.format(tt))
            sys.stdout.flush()

        true_agents = _update_true_agents_prob(true_agents, tt, dt, b_model, rng)
        global_true.append(deepcopy(true_agents))

        pred_args = {'state_mat_args': state_mat_args}
        glmb.predict(tt, filt_args=pred_args)

        meas_in = _gen_meas(tt, true_agents, filt.proc_noise, filt.meas_noise, rng)

        cor_args = {'meas_fun_args': meas_fun_args}
        glmb.correct(tt, meas_in, filt_args=cor_args)

        extract_kwargs = {'update': True, 'calc_states': False}
        glmb.cleanup(extract_kwargs=extract_kwargs)

    extract_kwargs = {'update': False, 'calc_states': True}
    glmb.extract_states(**extract_kwargs)

    glmb.calculate_ospa(global_true, 2, 1)

    if debug_plots:
        glmb.plot_states_labels([0, 1], true_states=global_true,
                                meas_inds=[0, 1])
        glmb.plot_card_dist()
        glmb.plot_card_history(time_units='s', time=time)
        glmb.plot_ospa_history()

    print('\tExpecting {} agents'.format(len(true_agents)))

    assert len(true_agents) == glmb.cardinality, 'Wrong cardinality'


def test_STM_GLMB():  # noqa
    print('Test STM-GLMB')

    rng = rnd.default_rng(global_seed)

    dt = 0.01
    t0, t1 = 0, 6 + dt

    filt = _setup_double_int_stf(dt)
    state_mat_args = (dt, 'test arg')
    meas_fun_args = ('useless arg', )

    b_model = _setup_stm_glmb_double_int_birth()

    RFS_base_args = {'prob_detection': 0.99, 'prob_survive': 0.99,
                     'in_filter': filt, 'birth_terms': b_model,
                     'clutter_den': 1**-7, 'clutter_rate': 1**-7}
    GLMB_args = {'req_births': len(b_model) + 1, 'req_surv': 1000,
                 'req_upd': 800, 'prune_threshold': 10**-5, 'max_hyps': 1000}
    glmb = tracker.STMGeneralizedLabeledMultiBernoulli(**GLMB_args, **RFS_base_args)

    time = np.arange(t0, t1, dt)
    true_agents = []
    global_true = []
    print('\tStarting sim')
    for kk, tt in enumerate(time):
        if np.mod(kk, 100) == 0:
            print('\t\t{:.2f}'.format(tt))
            sys.stdout.flush()

        true_agents = _update_true_agents_prob(true_agents, tt, dt, b_model, rng)
        global_true.append(deepcopy(true_agents))

        pred_args = {'state_mat_args': state_mat_args}
        glmb.predict(tt, filt_args=pred_args)

        meas_in = _gen_meas_stf(tt, true_agents, filt.proc_noise,
                                filt.proc_noise_dof, filt.meas_noise,
                                filt.meas_noise_dof, rng)

        cor_args = {'meas_fun_args': meas_fun_args}
        glmb.correct(tt, meas_in, filt_args=cor_args)

        extract_kwargs = {'update': True, 'calc_states': False}
        glmb.cleanup(extract_kwargs=extract_kwargs)

    extract_kwargs = {'update': False, 'calc_states': True}
    glmb.extract_states(**extract_kwargs)

    if debug_plots:
        glmb.plot_states_labels([0, 1], true_states=global_true,
                                meas_inds=[0, 1])
        glmb.plot_card_dist()
    print('\tExpecting {} agents'.format(len(true_agents)))

    assert len(true_agents) == glmb.cardinality, 'Wrong cardinality'


@pytest.mark.slow
def test_SMC_GLMB():  # noqa
    print('Test SMC-GLMB')

    rng = rnd.default_rng(global_seed)
    filt_rng = rnd.default_rng(global_seed)

    dt = 0.01
    t0, t1 = 0, 1 + dt
    num_parts = 1000
    prob_detection = 0.99
    prob_survive = 0.98

    filt = _setup_double_int_pf(dt, filt_rng)
    meas_fun_args = ()
    dyn_fun_params = (dt, )

    b_model = _setup_smc_glmb_double_int_birth(num_parts, rng)

    def compute_prob_detection(part_lst, prob_det):
        if len(part_lst) == 0:
            return np.array([])
        else:
            return prob_det * np.ones(len(part_lst))

    def compute_prob_survive(part_lst, prob_survive):
        if len(part_lst) == 0:
            return np.array([])
        else:
            return prob_survive * np.ones(len(part_lst))

    RFS_base_args = {'prob_detection': prob_detection,
                     'prob_survive': prob_survive, 'in_filter': filt,
                     'birth_terms': b_model, 'clutter_den': 1**-7,
                     'clutter_rate': 1**-7}
    GLMB_args = {'req_births': len(b_model) + 1, 'req_surv': 1000,
                 'req_upd': 800, 'prune_threshold': 10**-5, 'max_hyps': 1000}
    SMC_args = {'compute_prob_detection': compute_prob_detection,
                'compute_prob_survive': compute_prob_survive}
    glmb = tracker.SMCGeneralizedLabeledMultiBernoulli(**SMC_args,
                                                       **GLMB_args,
                                                       **RFS_base_args)

    time = np.arange(t0, t1, dt)
    true_agents = []
    global_true = []
    print('\tStarting sim')
    for kk, tt in enumerate(time):
        if np.mod(kk, 100) == 0:
            print('\t\t{:.2f}'.format(tt))
            sys.stdout.flush()

        true_agents = _update_true_agents_prob_smc(true_agents, tt, dt, b_model, rng)
        global_true.append(deepcopy(true_agents))

        pred_args = {'dyn_fun_params': dyn_fun_params}
        prob_surv_args = (prob_survive, )
        glmb.predict(tt, prob_surv_args=prob_surv_args, filt_args=pred_args)

        meas_in = _gen_meas(tt, true_agents, filt.proc_noise, filt.meas_noise, rng)

        cor_args = {'meas_fun_args': meas_fun_args}
        prob_det_args = (prob_detection, )
        glmb.correct(tt, meas_in, prob_det_args=prob_det_args,
                     filt_args=cor_args)

        extract_kwargs = {'update': True, 'calc_states': False}
        glmb.cleanup(extract_kwargs=extract_kwargs)

    extract_kwargs = {'update': False, 'calc_states': True}
    glmb.extract_states(**extract_kwargs)

    if debug_plots:
        glmb.plot_states_labels([0, 1], true_states=global_true,
                                meas_inds=[0, 1])
        glmb.plot_card_dist()
        glmb.plot_card_history(time_units='s', time=time)
    print('\tExpecting {} agents'.format(len(true_agents)))

    assert len(true_agents) == glmb.cardinality, 'Wrong cardinality'


def test_USMC_GLMB():  # noqa
    print('Test USMC-GLMB')

    rng = rnd.default_rng(global_seed)
    filt_rng = rnd.default_rng(global_seed)

    dt = 0.01
    t0, t1 = 0, 1 + dt
    num_parts = 75
    prob_detection = 0.99
    prob_survive = 0.98
    use_MCMC = False

    filt = _setup_double_int_upf(dt, filt_rng, use_MCMC)
    meas_fun_args = ()
    dyn_fun_params = (dt, )

    b_model = _setup_usmc_glmb_double_int_birth(num_parts, rng)

    def compute_prob_detection(part_lst, prob_det):
        if len(part_lst) == 0:
            return np.array([])
        else:
            return prob_det * np.ones(len(part_lst))

    def compute_prob_survive(part_lst, prob_survive):
        if len(part_lst) == 0:
            return np.array([])
        else:
            return prob_survive * np.ones(len(part_lst))

    RFS_base_args = {'prob_detection': prob_detection,
                     'prob_survive': prob_survive, 'in_filter': filt,
                     'birth_terms': b_model, 'clutter_den': 1**-7,
                     'clutter_rate': 1**-7}
    GLMB_args = {'req_births': len(b_model) + 1, 'req_surv': 1000,
                 'req_upd': 800, 'prune_threshold': 10**-5, 'max_hyps': 1000}
    SMC_args = {'compute_prob_detection': compute_prob_detection,
                'compute_prob_survive': compute_prob_survive}
    glmb = tracker.SMCGeneralizedLabeledMultiBernoulli(**SMC_args,
                                                       **GLMB_args,
                                                       **RFS_base_args)

    time = np.arange(t0, t1, dt)
    true_agents = []
    global_true = []
    print('\tStarting sim')
    for kk, tt in enumerate(time):
        if np.mod(kk, 100) == 0:
            print('\t\t{:.2f}'.format(tt))
            sys.stdout.flush()

        true_agents = _update_true_agents_prob_usmc(true_agents, tt, dt,
                                                    b_model, rng)
        global_true.append(deepcopy(true_agents))

        prob_surv_args = (prob_survive, )
        ukf_kwargs_pred = {'state_mat_args': dyn_fun_params}
        filt_args_pred = {'ukf_kwargs': ukf_kwargs_pred}
        glmb.predict(tt, prob_surv_args=prob_surv_args, filt_args=filt_args_pred)

        meas_in = _gen_meas(tt, true_agents, filt.proc_noise, filt.meas_noise, rng)

        prob_det_args = (prob_detection, )
        ukf_kwargs_cor = {'meas_fun_args': meas_fun_args}
        filt_args_cor = {'ukf_kwargs': ukf_kwargs_cor}
        glmb.correct(tt, meas_in, prob_det_args=prob_det_args,
                     filt_args=filt_args_cor)

        extract_kwargs = {'update': True, 'calc_states': False}
        glmb.cleanup(extract_kwargs=extract_kwargs)

    extract_kwargs = {'update': False, 'calc_states': True}
    glmb.extract_states(**extract_kwargs)

    if debug_plots:
        glmb.plot_states_labels([0, 1], true_states=global_true,
                                meas_inds=[0, 1])
        glmb.plot_card_dist()
        glmb.plot_card_history(time_units='s', time=time)
    print('\tExpecting {} agents'.format(len(true_agents)))
    print('\tmax cardinality {}'.format(np.max([len(s_set) for s_set in glmb.states])))

    assert len(true_agents) == glmb.cardinality, 'Wrong cardinality'


def test_MCMC_USMC_GLMB():  # noqa
    print('Test MCMC USMC-GLMB')

    rng = rnd.default_rng(global_seed)
    filt_rng = rnd.default_rng(global_seed)

    dt = 0.01
    t0, t1 = 0, 1 + dt
    num_parts = 30
    prob_detection = 0.99
    prob_survive = 0.98
    use_MCMC = True

    filt = _setup_double_int_upf(dt, filt_rng, use_MCMC)
    meas_fun_args = ()
    dyn_fun_params = (dt, )

    b_model = _setup_usmc_glmb_double_int_birth(num_parts, rng)

    def compute_prob_detection(part_lst, prob_det):
        if len(part_lst) == 0:
            return np.array([])
        else:
            return prob_det * np.ones(len(part_lst))

    def compute_prob_survive(part_lst, prob_survive):
        if len(part_lst) == 0:
            return np.array([])
        else:
            return prob_survive * np.ones(len(part_lst))

    RFS_base_args = {'prob_detection': prob_detection,
                     'prob_survive': prob_survive, 'in_filter': filt,
                     'birth_terms': b_model, 'clutter_den': 1**-7,
                     'clutter_rate': 1**-7}
    GLMB_args = {'req_births': len(b_model) + 1, 'req_surv': 1000,
                 'req_upd': 800, 'prune_threshold': 10**-5, 'max_hyps': 1000}
    SMC_args = {'compute_prob_detection': compute_prob_detection,
                'compute_prob_survive': compute_prob_survive}
    glmb = tracker.SMCGeneralizedLabeledMultiBernoulli(**SMC_args,
                                                       **GLMB_args,
                                                       **RFS_base_args)

    time = np.arange(t0, t1, dt)
    true_agents = []
    global_true = []
    print('\tStarting sim')
    for kk, tt in enumerate(time):
        if np.mod(kk, 100) == 0:
            print('\t\t{:.2f}'.format(tt))
            sys.stdout.flush()

        true_agents = _update_true_agents_prob_usmc(true_agents, tt, dt,
                                                    b_model, rng)
        global_true.append(deepcopy(true_agents))

        prob_surv_args = (prob_survive, )
        ukf_kwargs_pred = {'state_mat_args': dyn_fun_params}
        filt_args_pred = {'ukf_kwargs': ukf_kwargs_pred}
        glmb.predict(tt, prob_surv_args=prob_surv_args, filt_args=filt_args_pred)

        meas_in = _gen_meas(tt, true_agents, filt.proc_noise, filt.meas_noise, rng)

        prob_det_args = (prob_detection, )
        ukf_kwargs_cor = {'meas_fun_args': meas_fun_args}
        filt_args_cor = {'ukf_kwargs': ukf_kwargs_cor}
        glmb.correct(tt, meas_in, prob_det_args=prob_det_args,
                     filt_args=filt_args_cor)

        extract_kwargs = {'update': True, 'calc_states': False}
        glmb.cleanup(extract_kwargs=extract_kwargs)

    extract_kwargs = {'update': False, 'calc_states': True}
    glmb.extract_states(**extract_kwargs)

    if debug_plots:
        glmb.plot_states_labels([0, 1], true_states=global_true,
                                meas_inds=[0, 1])
        glmb.plot_card_dist()
        glmb.plot_card_history(time_units='s', time=time)
    print('\tExpecting {} agents'.format(len(true_agents)))
    print('max cardinality {}'.format(np.max([len(s_set) for s_set in glmb.states])))

    assert len(true_agents) == glmb.cardinality, 'Wrong cardinality'


def test_JGLMB():  # noqa
    print('Test GM-JGLMB')

    rng = rnd.default_rng(global_seed)

    dt = 0.01
    # t0, t1 = 0, 6 + dt
    t0, t1 = 0, 6 + dt

    filt = _setup_double_int_kf(dt)
    state_mat_args = (dt, 'test arg')
    meas_fun_args = ('useless arg', )

    b_model = _setup_gm_glmb_double_int_birth()

    RFS_base_args = {'prob_detection': 0.99, 'prob_survive': 0.98,
                     'in_filter': filt, 'birth_terms': b_model,
                     'clutter_den': 1**-3, 'clutter_rate': 1**-3}
    JGLMB_args = {'req_births': len(b_model) + 1, 'req_surv': 1000,
                 'req_upd': 800, 'prune_threshold': 10**-5, 'max_hyps': 1000}
    jglmb = tracker.JointGeneralizedLabeledMultiBernoulli(**JGLMB_args,
                                                          **RFS_base_args)

    time = np.arange(t0, t1, dt)
    true_agents = []
    global_true = []
    print('\tStarting sim')
    for kk, tt in enumerate(time):
        if np.mod(kk, 100) == 0:
            print('\t\t{:.2f}'.format(tt))
            sys.stdout.flush()

        true_agents = _update_true_agents_prob(true_agents, tt, dt, b_model, rng)
        global_true.append(deepcopy(true_agents))

        pred_args = {'state_mat_args': state_mat_args}
        jglmb.predict(tt, filt_args=pred_args)

        meas_in = _gen_meas(tt, true_agents, filt.proc_noise, filt.meas_noise, rng)

        cor_args = {'meas_fun_args': meas_fun_args}
        jglmb.correct(tt, meas_in, filt_args=cor_args)

        extract_kwargs = {'update': True, 'calc_states': False}
        jglmb.cleanup(extract_kwargs=extract_kwargs)

    extract_kwargs = {'update': False, 'calc_states': True}
    jglmb.extract_states(**extract_kwargs)

    jglmb.calculate_ospa(global_true, 2, 1)

    if debug_plots:
        jglmb.plot_states_labels([0, 1], true_states=global_true,
                                meas_inds=[0, 1])
        jglmb.plot_card_dist()
        jglmb.plot_card_history(time_units='s', time=time)
        jglmb.plot_ospa_history()

    print('\tExpecting {} agents'.format(len(true_agents)))

    assert len(true_agents) == jglmb.cardinality, 'Wrong cardinality'


def test_STM_JGLMB():  # noqa
    print('Test STM-JGLMB')

    rng = rnd.default_rng(global_seed)

    dt = 0.01
    t0, t1 = 0, 6 + dt

    filt = _setup_double_int_stf(dt)
    state_mat_args = (dt, 'test arg')
    meas_fun_args = ('useless arg', )

    b_model = _setup_stm_glmb_double_int_birth()

    RFS_base_args = {'prob_detection': 0.99, 'prob_survive': 0.98,
                     'in_filter': filt, 'birth_terms': b_model,
                     'clutter_den': 1**-3, 'clutter_rate': 1**-3}
    JGLMB_args = {'req_births': len(b_model) + 1, 'req_surv': 1000,
                 'req_upd': 800, 'prune_threshold': 10**-5, 'max_hyps': 1000}
    jglmb = tracker.STMJointGeneralizedLabeledMultiBernoulli(**JGLMB_args,
                                                             **RFS_base_args)

    time = np.arange(t0, t1, dt)
    true_agents = []
    global_true = []
    print('\tStarting sim')
    for kk, tt in enumerate(time):
        if np.mod(kk, 100) == 0:
            print('\t\t{:.2f}'.format(tt))
            sys.stdout.flush()

        true_agents = _update_true_agents_prob(true_agents, tt, dt, b_model, rng)
        global_true.append(deepcopy(true_agents))

        pred_args = {'state_mat_args': state_mat_args}
        jglmb.predict(tt, filt_args=pred_args)

        meas_in = _gen_meas_stf(tt, true_agents, filt.proc_noise,
                                filt.proc_noise_dof, filt.meas_noise,
                                filt.meas_noise_dof, rng)

        cor_args = {'meas_fun_args': meas_fun_args}
        jglmb.correct(tt, meas_in, filt_args=cor_args)

        extract_kwargs = {'update': True, 'calc_states': False}
        jglmb.cleanup(extract_kwargs=extract_kwargs)

    extract_kwargs = {'update': False, 'calc_states': True}
    jglmb.extract_states(**extract_kwargs)

    if debug_plots:
        jglmb.plot_states_labels([0, 1], true_states=global_true,
                                meas_inds=[0, 1])
        jglmb.plot_card_dist()
    print('\tExpecting {} agents'.format(len(true_agents)))

    assert len(true_agents) == jglmb.cardinality, 'Wrong cardinality'


@pytest.mark.slow
def test_SMC_JGLMB():  # noqa
    print('Test SMC-JGLMB')

    rng = rnd.default_rng(global_seed)
    filt_rng = rnd.default_rng(global_seed)

    dt = 0.01
    t0, t1 = 0, 1 + dt
    num_parts = 1000
    prob_detection = 0.99
    prob_survive = 0.98

    filt = _setup_double_int_pf(dt, filt_rng)
    meas_fun_args = ()
    dyn_fun_params = (dt, )

    b_model = _setup_smc_glmb_double_int_birth(num_parts, rng)

    def compute_prob_detection(part_lst, prob_det):
        if len(part_lst) == 0:
            return np.array([])
        else:
            return prob_det * np.ones(len(part_lst))

    def compute_prob_survive(part_lst, prob_survive):
        if len(part_lst) == 0:
            return np.array([])
        else:
            return prob_survive * np.ones(len(part_lst))

    RFS_base_args = {'prob_detection': prob_detection,
                     'prob_survive': prob_survive, 'in_filter': filt,
                     'birth_terms': b_model, 'clutter_den': 1**-7,
                     'clutter_rate': 1**-7}
    JGLMB_args = {'req_births': len(b_model) + 1, 'req_surv': 1000,
                  'req_upd': 800, 'prune_threshold': 10**-5, 'max_hyps': 1000}
    SMC_args = {'compute_prob_detection': compute_prob_detection,
                'compute_prob_survive': compute_prob_survive}
    jglmb = tracker.SMCJointGeneralizedLabeledMultiBernoulli(**SMC_args,
                                                             **JGLMB_args,
                                                             **RFS_base_args)

    time = np.arange(t0, t1, dt)
    true_agents = []
    global_true = []
    print('\tStarting sim')
    for kk, tt in enumerate(time):
        if np.mod(kk, 100) == 0:
            print('\t\t{:.2f}'.format(tt))
            sys.stdout.flush()

        true_agents = _update_true_agents_prob_smc(true_agents, tt, dt, b_model, rng)
        global_true.append(deepcopy(true_agents))

        pred_args = {'dyn_fun_params': dyn_fun_params}
        prob_surv_args = (prob_survive, )
        jglmb.predict(tt, prob_surv_args=prob_surv_args, filt_args=pred_args)

        meas_in = _gen_meas(tt, true_agents, filt.proc_noise, filt.meas_noise, rng)

        cor_args = {'meas_fun_args': meas_fun_args}
        prob_det_args = (prob_detection, )
        jglmb.correct(tt, meas_in, prob_det_args=prob_det_args,
                     filt_args=cor_args)

        extract_kwargs = {'update': True, 'calc_states': False}
        jglmb.cleanup(extract_kwargs=extract_kwargs)

    extract_kwargs = {'update': False, 'calc_states': True}
    jglmb.extract_states(**extract_kwargs)

    if debug_plots:
        jglmb.plot_states_labels([0, 1], true_states=global_true,
                                meas_inds=[0, 1])
        jglmb.plot_card_dist()
        jglmb.plot_card_history(time_units='s', time=time)
    print('\tExpecting {} agents'.format(len(true_agents)))

    assert len(true_agents) == jglmb.cardinality, 'Wrong cardinality'


def test_USMC_JGLMB():  # noqa
    print('Test USMC-JGLMB')

    rng = rnd.default_rng(global_seed)
    filt_rng = rnd.default_rng(global_seed)

    dt = 0.01
    t0, t1 = 0, 1 + dt
    num_parts = 75
    prob_detection = 0.99
    prob_survive = 0.98
    use_MCMC = False

    filt = _setup_double_int_upf(dt, filt_rng, use_MCMC)
    meas_fun_args = ()
    dyn_fun_params = (dt, )

    b_model = _setup_usmc_glmb_double_int_birth(num_parts, rng)

    def compute_prob_detection(part_lst, prob_det):
        if len(part_lst) == 0:
            return np.array([])
        else:
            return prob_det * np.ones(len(part_lst))

    def compute_prob_survive(part_lst, prob_survive):
        if len(part_lst) == 0:
            return np.array([])
        else:
            return prob_survive * np.ones(len(part_lst))

    RFS_base_args = {'prob_detection': prob_detection,
                     'prob_survive': prob_survive, 'in_filter': filt,
                     'birth_terms': b_model, 'clutter_den': 1**-7,
                     'clutter_rate': 1**-7}
    JGLMB_args = {'req_births': len(b_model) + 1, 'req_surv': 1000,
                 'req_upd': 800, 'prune_threshold': 10**-5, 'max_hyps': 1000}
    SMC_args = {'compute_prob_detection': compute_prob_detection,
                'compute_prob_survive': compute_prob_survive}
    jglmb = tracker.SMCJointGeneralizedLabeledMultiBernoulli(**SMC_args,
                                                             **JGLMB_args,
                                                             **RFS_base_args)

    time = np.arange(t0, t1, dt)
    true_agents = []
    global_true = []
    print('\tStarting sim')
    for kk, tt in enumerate(time):
        if np.mod(kk, 100) == 0:
            print('\t\t{:.2f}'.format(tt))
            sys.stdout.flush()

        true_agents = _update_true_agents_prob_usmc(true_agents, tt, dt,
                                                    b_model, rng)
        global_true.append(deepcopy(true_agents))

        prob_surv_args = (prob_survive, )
        ukf_kwargs_pred = {'state_mat_args': dyn_fun_params}
        filt_args_pred = {'ukf_kwargs': ukf_kwargs_pred}
        jglmb.predict(tt, prob_surv_args=prob_surv_args, filt_args=filt_args_pred)

        meas_in = _gen_meas(tt, true_agents, filt.proc_noise, filt.meas_noise, rng)

        prob_det_args = (prob_detection, )
        ukf_kwargs_cor = {'meas_fun_args': meas_fun_args}
        filt_args_cor = {'ukf_kwargs': ukf_kwargs_cor}
        jglmb.correct(tt, meas_in, prob_det_args=prob_det_args,
                      filt_args=filt_args_cor)

        extract_kwargs = {'update': True, 'calc_states': False}
        jglmb.cleanup(extract_kwargs=extract_kwargs)

    extract_kwargs = {'update': False, 'calc_states': True}
    jglmb.extract_states(**extract_kwargs)

    if debug_plots:
        jglmb.plot_states_labels([0, 1], true_states=global_true,
                                 meas_inds=[0, 1])
        jglmb.plot_card_dist()
        jglmb.plot_card_history(time_units='s', time=time)
    print('\tExpecting {} agents'.format(len(true_agents)))
    print('\tmax cardinality {}'.format(np.max([len(s_set)
                                                for s_set in jglmb.states])))

    assert len(true_agents) == jglmb.cardinality, 'Wrong cardinality'


def test_MCMC_USMC_JGLMB():  # noqa
    print('Test MCMC USMC-JGLMB')

    rng = rnd.default_rng(global_seed)
    filt_rng = rnd.default_rng(global_seed)

    dt = 0.01
    t0, t1 = 0, 1 + dt
    num_parts = 30
    prob_detection = 0.99
    prob_survive = 0.98
    use_MCMC = True

    filt = _setup_double_int_upf(dt, filt_rng, use_MCMC)
    meas_fun_args = ()
    dyn_fun_params = (dt, )

    b_model = _setup_usmc_glmb_double_int_birth(num_parts, rng)

    def compute_prob_detection(part_lst, prob_det):
        if len(part_lst) == 0:
            return np.array([])
        else:
            return prob_det * np.ones(len(part_lst))

    def compute_prob_survive(part_lst, prob_survive):
        if len(part_lst) == 0:
            return np.array([])
        else:
            return prob_survive * np.ones(len(part_lst))

    RFS_base_args = {'prob_detection': prob_detection,
                     'prob_survive': prob_survive, 'in_filter': filt,
                     'birth_terms': b_model, 'clutter_den': 1**-7,
                     'clutter_rate': 1**-7}
    JGLMB_args = {'req_births': len(b_model) + 1, 'req_surv': 1000,
                  'req_upd': 800, 'prune_threshold': 10**-5, 'max_hyps': 1000}
    SMC_args = {'compute_prob_detection': compute_prob_detection,
                'compute_prob_survive': compute_prob_survive}
    jglmb = tracker.SMCJointGeneralizedLabeledMultiBernoulli(**SMC_args,
                                                             **JGLMB_args,
                                                             **RFS_base_args)

    time = np.arange(t0, t1, dt)
    true_agents = []
    global_true = []
    print('\tStarting sim')
    for kk, tt in enumerate(time):
        if np.mod(kk, 100) == 0:
            print('\t\t{:.2f}'.format(tt))
            sys.stdout.flush()

        true_agents = _update_true_agents_prob_usmc(true_agents, tt, dt,
                                                    b_model, rng)
        global_true.append(deepcopy(true_agents))

        prob_surv_args = (prob_survive, )
        ukf_kwargs_pred = {'state_mat_args': dyn_fun_params}
        filt_args_pred = {'ukf_kwargs': ukf_kwargs_pred}
        jglmb.predict(tt, prob_surv_args=prob_surv_args, filt_args=filt_args_pred)

        meas_in = _gen_meas(tt, true_agents, filt.proc_noise, filt.meas_noise, rng)

        prob_det_args = (prob_detection, )
        ukf_kwargs_cor = {'meas_fun_args': meas_fun_args}
        filt_args_cor = {'ukf_kwargs': ukf_kwargs_cor}
        jglmb.correct(tt, meas_in, prob_det_args=prob_det_args,
                     filt_args=filt_args_cor)

        extract_kwargs = {'update': True, 'calc_states': False}
        jglmb.cleanup(extract_kwargs=extract_kwargs)

    extract_kwargs = {'update': False, 'calc_states': True}
    jglmb.extract_states(**extract_kwargs)

    if debug_plots:
        jglmb.plot_states_labels([0, 1], true_states=global_true,
                                meas_inds=[0, 1])
        jglmb.plot_card_dist()
        jglmb.plot_card_history(time_units='s', time=time)
    print('\tExpecting {} agents'.format(len(true_agents)))
    print('max cardinality {}'.format(np.max([len(s_set)
                                              for s_set in jglmb.states])))

    assert len(true_agents) == jglmb.cardinality, 'Wrong cardinality'


def test_QKF_JGLMB():  # noqa
    print('Test QKF-JGLMB')

    rng = rnd.default_rng(global_seed)

    dt = 1  # s
    t0, t1 = 0, 20 + dt
    prob_detection = 0.99
    prob_survive = 0.98
    use_sqkf = False
    print_interval = 10  # s

    # measurement noise parameters
    m_vars = (100, (0.15 * np.pi / 180)**2)

    filt, state_mat, meas_fun = _setup_qkf(dt, use_sqkf, m_vars)
    b_model = _setup_gsm_birth()

    RFS_base_args = {'prob_detection': prob_detection,
                     'prob_survive': prob_survive, 'in_filter': filt,
                     'birth_terms': b_model, 'clutter_den': 1**-7,
                     'clutter_rate': 1**-7}
    JGLMB_args = {'req_births': len(b_model) + 1, 'req_surv': 1000,
                  'req_upd': 800, 'prune_threshold': 10**-5, 'max_hyps': 1000}
    jglmb = tracker.JointGeneralizedLabeledMultiBernoulli(**JGLMB_args,
                                                          **RFS_base_args)

    time = np.arange(t0, t1, dt)
    true_agents = []
    global_true = []
    print('\tStarting sim')
    for kk, tt in enumerate(time):
        if np.mod(kk, int(print_interval / dt)) == 0:
            print('\t\t{:.2f}'.format(tt))
            sys.stdout.flush()

        true_agents = _update_true_agents_gsm(true_agents, tt, b_model, rng,
                                              state_mat)
        global_true.append(deepcopy(true_agents))

        filt_args_pred = {}
        jglmb.predict(tt, filt_args=filt_args_pred)

        meas_in = _gen_meas_qkf(tt, true_agents, filt.proc_noise, meas_fun,
                                m_vars, rng)

        filt_args_cor = {}
        jglmb.correct(tt, meas_in, filt_args=filt_args_cor)

        extract_kwargs = {'update': True, 'calc_states': False}
        jglmb.cleanup(extract_kwargs=extract_kwargs)

    extract_kwargs = {'update': False, 'calc_states': True}
    jglmb.extract_states(**extract_kwargs)

    if debug_plots:
        jglmb.plot_states_labels([0, 1], true_states=global_true,
                                meas_inds=[])
        jglmb.plot_card_dist()
        jglmb.plot_card_history(time_units='s', time=time)

    print('\tExpecting {} agents'.format(len(true_agents)))
    print('\tmax cardinality {}'.format(np.max([len(s_set)
                                                for s_set in jglmb.states])))

    assert len(true_agents) == jglmb.cardinality, 'Wrong cardinality'


def test_SQKF_JGLMB():  # noqa
    print('Test SQKF-JGLMB')

    rng = rnd.default_rng(global_seed)

    dt = 1  # s
    t0, t1 = 0, 20 + dt
    prob_detection = 0.99
    prob_survive = 0.98
    use_sqkf = True
    print_interval = 10  # s

    # measurement noise parameters
    m_vars = (100, (0.15 * np.pi / 180)**2)

    filt, state_mat, meas_fun = _setup_qkf(dt, use_sqkf, m_vars)
    b_model = _setup_gsm_birth()

    RFS_base_args = {'prob_detection': prob_detection,
                     'prob_survive': prob_survive, 'in_filter': filt,
                     'birth_terms': b_model, 'clutter_den': 1**-7,
                     'clutter_rate': 1**-7}
    JGLMB_args = {'req_births': len(b_model) + 1, 'req_surv': 1000,
                  'req_upd': 800, 'prune_threshold': 10**-5, 'max_hyps': 1000}
    jglmb = tracker.GeneralizedLabeledMultiBernoulli(**JGLMB_args,
                                                     **RFS_base_args)

    time = np.arange(t0, t1, dt)
    true_agents = []
    global_true = []
    print('\tStarting sim')
    for kk, tt in enumerate(time):
        if np.mod(kk, int(print_interval / dt)) == 0:
            print('\t\t{:.2f}'.format(tt))
            sys.stdout.flush()

        true_agents = _update_true_agents_gsm(true_agents, tt, b_model, rng,
                                              state_mat)
        global_true.append(deepcopy(true_agents))

        filt_args_pred = {}
        jglmb.predict(tt, filt_args=filt_args_pred)

        meas_in = _gen_meas_qkf(tt, true_agents, filt.proc_noise, meas_fun,
                                m_vars, rng)

        filt_args_cor = {}
        jglmb.correct(tt, meas_in, filt_args=filt_args_cor)

        extract_kwargs = {'update': True, 'calc_states': False}
        jglmb.cleanup(extract_kwargs=extract_kwargs)

    extract_kwargs = {'update': False, 'calc_states': True}
    jglmb.extract_states(**extract_kwargs)

    if debug_plots:
        jglmb.plot_states_labels([0, 1], true_states=global_true,
                                meas_inds=[])
        jglmb.plot_card_dist()
        jglmb.plot_card_history(time_units='s', time=time)

    print('\tExpecting {} agents'.format(len(true_agents)))
    print('max cardinality {}'.format(np.max([len(s_set)
                                              for s_set in jglmb.states])))

    assert len(true_agents) == jglmb.cardinality, 'Wrong cardinality'


def test_UKF_JGLMB():  # noqa
    print('Test UKF-JGLMB')

    rng = rnd.default_rng(global_seed)

    dt = 1  # s
    t0, t1 = 0, 20 + dt
    prob_detection = 0.99
    prob_survive = 0.98
    print_interval = 10  # s

    # measurement noise parameters
    m_vars = (100, (0.15 * np.pi / 180)**2)

    filt, state_mat, meas_fun = _setup_ukf(dt, m_vars)
    b_model = _setup_gsm_birth()

    RFS_base_args = {'prob_detection': prob_detection,
                     'prob_survive': prob_survive, 'in_filter': filt,
                     'birth_terms': b_model, 'clutter_den': 1**-7,
                     'clutter_rate': 1**-7}
    JGLMB_args = {'req_births': len(b_model) + 1, 'req_surv': 1000,
                  'req_upd': 800, 'prune_threshold': 10**-5, 'max_hyps': 1000}
    jglmb = tracker.JointGeneralizedLabeledMultiBernoulli(**JGLMB_args,
                                                          **RFS_base_args)

    time = np.arange(t0, t1, dt)
    true_agents = []
    global_true = []
    print('\tStarting sim')
    for kk, tt in enumerate(time):
        if np.mod(kk, int(print_interval / dt)) == 0:
            print('\t\t{:.2f}'.format(tt))
            sys.stdout.flush()

        true_agents = _update_true_agents_gsm(true_agents, tt, b_model, rng,
                                              state_mat)
        global_true.append(deepcopy(true_agents))

        filt_args_pred = {}
        jglmb.predict(tt, filt_args=filt_args_pred)

        meas_in = _gen_meas_qkf(tt, true_agents, filt.proc_noise, meas_fun,
                                m_vars, rng)

        filt_args_cor = {}
        jglmb.correct(tt, meas_in, filt_args=filt_args_cor)

        extract_kwargs = {'update': True, 'calc_states': False}
        jglmb.cleanup(extract_kwargs=extract_kwargs)

    extract_kwargs = {'update': False, 'calc_states': True}
    jglmb.extract_states(**extract_kwargs)

    if debug_plots:
        jglmb.plot_states_labels([0, 1], true_states=global_true,
                                meas_inds=[])
        jglmb.plot_card_dist()
        jglmb.plot_card_history(time_units='s', time=time)

    print('\tExpecting {} agents'.format(len(true_agents)))
    print('\tmax cardinality {}'.format(np.max([len(s_set)
                                                for s_set in jglmb.states])))

    assert len(true_agents) == jglmb.cardinality, 'Wrong cardinality'


def test_QKF_GSM_JGLMB():  # noqa
    print('Test QKF GSM-JGLMB')

    rng = rnd.default_rng(global_seed)
    filt_rng = rnd.default_rng(global_seed)

    dt = 1  # s
    t0, t1 = 0, 20 + dt
    prob_detection = 0.99
    prob_survive = 0.98
    use_sqkf = False
    print_interval = 10  # s

    # measurement noise parameters
    m_dfs = (2, 2)
    m_vars = (25, (0.015 * np.pi / 180)**2)

    filt, state_mat, meas_fun = _setup_qkf_gsm(dt, filt_rng, use_sqkf, m_dfs, m_vars)
    b_model = _setup_gsm_birth()

    RFS_base_args = {'prob_detection': prob_detection,
                     'prob_survive': prob_survive, 'in_filter': filt,
                     'birth_terms': b_model, 'clutter_den': 1**-7,
                     'clutter_rate': 1**-7}
    JGLMB_args = {'req_births': len(b_model) + 1, 'req_surv': 1000,
                 'req_upd': 800, 'prune_threshold': 10**-5, 'max_hyps': 1000}
    jglmb = tracker.GSMJointGeneralizedLabeledMultiBernoulli(**JGLMB_args,
                                                             **RFS_base_args)

    time = np.arange(t0, t1, dt)
    true_agents = []
    global_true = []
    print('\tStarting sim')
    for kk, tt in enumerate(time):
        if np.mod(kk, int(print_interval / dt)) == 0:
            print('\t\t{:.2f}'.format(tt))
            sys.stdout.flush()

        true_agents = _update_true_agents_gsm(true_agents, tt, b_model, rng,
                                              state_mat)
        global_true.append(deepcopy(true_agents))

        filt_args_pred = {}
        jglmb.predict(tt, filt_args=filt_args_pred)

        meas_in = _gen_meas_gsm(tt, true_agents, filt.proc_noise, meas_fun,
                                m_dfs, m_vars, rng)

        filt_args_cor = {}
        jglmb.correct(tt, meas_in, filt_args=filt_args_cor)

        extract_kwargs = {'update': True, 'calc_states': False}
        jglmb.cleanup(extract_kwargs=extract_kwargs)

    extract_kwargs = {'update': False, 'calc_states': True}
    jglmb.extract_states(**extract_kwargs)

    if debug_plots:
        jglmb.plot_states_labels([0, 1], true_states=global_true,
                                meas_inds=[])
        jglmb.plot_card_dist()
        jglmb.plot_card_history(time_units='s', time=time)

    print('\tExpecting {} agents'.format(len(true_agents)))
    print('\tmax cardinality {}'.format(np.max([len(s_set)
                                                for s_set in jglmb.states])))

    assert len(true_agents) == jglmb.cardinality, 'Wrong cardinality'


def test_SQKF_GSM_JGLMB():  # noqa
    print('Test SQKF GSM-JGLMB')

    rng = rnd.default_rng(global_seed)
    filt_rng = rnd.default_rng(global_seed)

    dt = 1  # s
    t0, t1 = 0, 20 + dt
    prob_detection = 0.99
    prob_survive = 0.98
    use_sqkf = True
    print_interval = 10  # s

    # measurement noise parameters
    m_dfs = (2, 2)
    m_vars = (25, (0.015 * np.pi / 180)**2)

    filt, state_mat, meas_fun = _setup_qkf_gsm(dt, filt_rng, use_sqkf, m_dfs, m_vars)
    b_model = _setup_gsm_birth()

    RFS_base_args = {'prob_detection': prob_detection,
                     'prob_survive': prob_survive, 'in_filter': filt,
                     'birth_terms': b_model, 'clutter_den': 1**-7,
                     'clutter_rate': 1**-7}
    JGLMB_args = {'req_births': len(b_model) + 1, 'req_surv': 1000,
                 'req_upd': 800, 'prune_threshold': 10**-5, 'max_hyps': 1000}
    jglmb = tracker.GSMJointGeneralizedLabeledMultiBernoulli(**JGLMB_args,
                                                             **RFS_base_args)

    time = np.arange(t0, t1, dt)
    true_agents = []
    global_true = []
    print('\tStarting sim')
    for kk, tt in enumerate(time):
        if np.mod(kk, int(print_interval / dt)) == 0:
            print('\t\t{:.2f}'.format(tt))
            sys.stdout.flush()

        true_agents = _update_true_agents_gsm(true_agents, tt, b_model, rng,
                                              state_mat)
        global_true.append(deepcopy(true_agents))

        filt_args_pred = {}
        jglmb.predict(tt, filt_args=filt_args_pred)

        meas_in = _gen_meas_gsm(tt, true_agents, filt.proc_noise, meas_fun,
                                m_dfs, m_vars, rng)

        filt_args_cor = {}
        jglmb.correct(tt, meas_in, filt_args=filt_args_cor)

        extract_kwargs = {'update': True, 'calc_states': False}
        jglmb.cleanup(extract_kwargs=extract_kwargs)

    extract_kwargs = {'update': False, 'calc_states': True}
    jglmb.extract_states(**extract_kwargs)

    if debug_plots:
        jglmb.plot_states_labels([0, 1], true_states=global_true,
                                meas_inds=[])
        jglmb.plot_card_dist()
        jglmb.plot_card_history(time_units='s', time=time)

    print('\tExpecting {} agents'.format(len(true_agents)))
    print('\tmax cardinality {}'.format(np.max([len(s_set)
                                                for s_set in jglmb.states])))

    assert len(true_agents) == jglmb.cardinality, 'Wrong cardinality'


def test_UKF_GSM_JGLMB():  # noqa
    print('Test UKF GSM-JGLMB')

    rng = rnd.default_rng(global_seed)
    filt_rng = rnd.default_rng(global_seed)

    dt = 1  # s
    t0, t1 = 0, 20 + dt
    prob_detection = 0.99
    prob_survive = 0.98
    print_interval = 10  # s

    # measurement noise parameters
    m_dfs = (2, 2)
    m_vars = (25, (0.015 * np.pi / 180)**2)

    filt, state_mat, meas_fun = _setup_ukf_gsm(dt, filt_rng, m_dfs, m_vars)
    b_model = _setup_gsm_birth()

    RFS_base_args = {'prob_detection': prob_detection,
                     'prob_survive': prob_survive, 'in_filter': filt,
                     'birth_terms': b_model, 'clutter_den': 1**-7,
                     'clutter_rate': 1**-7}
    JGLMB_args = {'req_births': len(b_model) + 1, 'req_surv': 1000,
                  'req_upd': 800, 'prune_threshold': 10**-5, 'max_hyps': 1000}
    jglmb = tracker.GSMJointGeneralizedLabeledMultiBernoulli(**JGLMB_args,
                                                             **RFS_base_args)

    time = np.arange(t0, t1, dt)
    true_agents = []
    global_true = []
    print('\tStarting sim')
    for kk, tt in enumerate(time):
        if np.mod(kk, int(print_interval / dt)) == 0:
            print('\t\t{:.2f}'.format(tt))
            sys.stdout.flush()

        true_agents = _update_true_agents_gsm(true_agents, tt, b_model, rng,
                                              state_mat)
        global_true.append(deepcopy(true_agents))

        filt_args_pred = {}
        jglmb.predict(tt, filt_args=filt_args_pred)

        meas_in = _gen_meas_gsm(tt, true_agents, filt.proc_noise, meas_fun,
                                m_dfs, m_vars, rng)

        filt_args_cor = {}
        jglmb.correct(tt, meas_in, filt_args=filt_args_cor)

        extract_kwargs = {'update': True, 'calc_states': False}
        jglmb.cleanup(extract_kwargs=extract_kwargs)

    extract_kwargs = {'update': False, 'calc_states': True}
    jglmb.extract_states(**extract_kwargs)

    if debug_plots:
        jglmb.plot_states_labels([0, 1], true_states=global_true,
                                 meas_inds=[])
        jglmb.plot_card_dist()
        jglmb.plot_card_history(time_units='s', time=time)

    print('\tExpecting {} agents'.format(len(true_agents)))
    print('\tmax cardinality {}'.format(np.max([len(s_set)
                                                for s_set in jglmb.states])))

    assert len(true_agents) == jglmb.cardinality, 'Wrong cardinality'


def test_QKF_GLMB():  # noqa
    print('Test QKF-GLMB')

    rng = rnd.default_rng(global_seed)

    dt = 1  # s
    t0, t1 = 0, 20 + dt
    prob_detection = 0.99
    prob_survive = 0.98
    use_sqkf = False
    print_interval = 10  # s

    # measurement noise parameters
    m_vars = (100, (0.15 * np.pi / 180)**2)

    filt, state_mat, meas_fun = _setup_qkf(dt, use_sqkf, m_vars)
    b_model = _setup_gsm_birth()

    RFS_base_args = {'prob_detection': prob_detection,
                     'prob_survive': prob_survive, 'in_filter': filt,
                     'birth_terms': b_model, 'clutter_den': 1**-7,
                     'clutter_rate': 1**-7}
    GLMB_args = {'req_births': len(b_model) + 1, 'req_surv': 1000,
                 'req_upd': 800, 'prune_threshold': 10**-5, 'max_hyps': 1000}
    glmb = tracker.GeneralizedLabeledMultiBernoulli(**GLMB_args,
                                                    **RFS_base_args)

    time = np.arange(t0, t1, dt)
    true_agents = []
    global_true = []
    print('\tStarting sim')
    for kk, tt in enumerate(time):
        if np.mod(kk, int(print_interval / dt)) == 0:
            print('\t\t{:.2f}'.format(tt))
            sys.stdout.flush()

        true_agents = _update_true_agents_gsm(true_agents, tt, b_model, rng,
                                              state_mat)
        global_true.append(deepcopy(true_agents))

        filt_args_pred = {}
        glmb.predict(tt, filt_args=filt_args_pred)

        meas_in = _gen_meas_qkf(tt, true_agents, filt.proc_noise, meas_fun,
                                m_vars, rng)

        filt_args_cor = {}
        glmb.correct(tt, meas_in, filt_args=filt_args_cor)

        extract_kwargs = {'update': True, 'calc_states': False}
        glmb.cleanup(extract_kwargs=extract_kwargs)

    extract_kwargs = {'update': False, 'calc_states': True}
    glmb.extract_states(**extract_kwargs)

    if debug_plots:
        glmb.plot_states_labels([0, 1], true_states=global_true,
                                meas_inds=[])
        glmb.plot_card_dist()
        glmb.plot_card_history(time_units='s', time=time)

    print('\tExpecting {} agents'.format(len(true_agents)))
    print('\tmax cardinality {}'.format(np.max([len(s_set) for s_set in glmb.states])))

    assert len(true_agents) == glmb.cardinality, 'Wrong cardinality'


def test_SQKF_GLMB():  # noqa
    print('Test SQKF-GLMB')

    rng = rnd.default_rng(global_seed)

    dt = 1  # s
    t0, t1 = 0, 20 + dt
    prob_detection = 0.99
    prob_survive = 0.98
    use_sqkf = True
    print_interval = 10  # s

    # measurement noise parameters
    m_vars = (100, (0.15 * np.pi / 180)**2)

    filt, state_mat, meas_fun = _setup_qkf(dt, use_sqkf, m_vars)
    b_model = _setup_gsm_birth()

    RFS_base_args = {'prob_detection': prob_detection,
                     'prob_survive': prob_survive, 'in_filter': filt,
                     'birth_terms': b_model, 'clutter_den': 1**-7,
                     'clutter_rate': 1**-7}
    GLMB_args = {'req_births': len(b_model) + 1, 'req_surv': 1000,
                 'req_upd': 800, 'prune_threshold': 10**-5, 'max_hyps': 1000}
    glmb = tracker.GeneralizedLabeledMultiBernoulli(**GLMB_args,
                                                    **RFS_base_args)

    time = np.arange(t0, t1, dt)
    true_agents = []
    global_true = []
    print('\tStarting sim')
    for kk, tt in enumerate(time):
        if np.mod(kk, int(print_interval / dt)) == 0:
            print('\t\t{:.2f}'.format(tt))
            sys.stdout.flush()

        true_agents = _update_true_agents_gsm(true_agents, tt, b_model, rng,
                                              state_mat)
        global_true.append(deepcopy(true_agents))

        filt_args_pred = {}
        glmb.predict(tt, filt_args=filt_args_pred)

        meas_in = _gen_meas_qkf(tt, true_agents, filt.proc_noise, meas_fun,
                                m_vars, rng)

        filt_args_cor = {}
        glmb.correct(tt, meas_in, filt_args=filt_args_cor)

        extract_kwargs = {'update': True, 'calc_states': False}
        glmb.cleanup(extract_kwargs=extract_kwargs)

    extract_kwargs = {'update': False, 'calc_states': True}
    glmb.extract_states(**extract_kwargs)

    if debug_plots:
        glmb.plot_states_labels([0, 1], true_states=global_true,
                                meas_inds=[])
        glmb.plot_card_dist()
        glmb.plot_card_history(time_units='s', time=time)

    print('\tExpecting {} agents'.format(len(true_agents)))
    print('max cardinality {}'.format(np.max([len(s_set) for s_set in glmb.states])))

    assert len(true_agents) == glmb.cardinality, 'Wrong cardinality'


def test_UKF_GLMB():  # noqa
    print('Test UKF-GLMB')

    rng = rnd.default_rng(global_seed)

    dt = 1  # s
    t0, t1 = 0, 20 + dt
    prob_detection = 0.99
    prob_survive = 0.98
    print_interval = 10  # s

    # measurement noise parameters
    m_vars = (100, (0.15 * np.pi / 180)**2)

    filt, state_mat, meas_fun = _setup_ukf(dt, m_vars)
    b_model = _setup_gsm_birth()

    RFS_base_args = {'prob_detection': prob_detection,
                     'prob_survive': prob_survive, 'in_filter': filt,
                     'birth_terms': b_model, 'clutter_den': 1**-7,
                     'clutter_rate': 1**-7}
    GLMB_args = {'req_births': len(b_model) + 1, 'req_surv': 1000,
                 'req_upd': 800, 'prune_threshold': 10**-5, 'max_hyps': 1000}
    glmb = tracker.GeneralizedLabeledMultiBernoulli(**GLMB_args,
                                                    **RFS_base_args)

    time = np.arange(t0, t1, dt)
    true_agents = []
    global_true = []
    print('\tStarting sim')
    for kk, tt in enumerate(time):
        if np.mod(kk, int(print_interval / dt)) == 0:
            print('\t\t{:.2f}'.format(tt))
            sys.stdout.flush()

        true_agents = _update_true_agents_gsm(true_agents, tt, b_model, rng,
                                              state_mat)
        global_true.append(deepcopy(true_agents))

        filt_args_pred = {}
        glmb.predict(tt, filt_args=filt_args_pred)

        meas_in = _gen_meas_qkf(tt, true_agents, filt.proc_noise, meas_fun,
                                m_vars, rng)

        filt_args_cor = {}
        glmb.correct(tt, meas_in, filt_args=filt_args_cor)

        extract_kwargs = {'update': True, 'calc_states': False}
        glmb.cleanup(extract_kwargs=extract_kwargs)

    extract_kwargs = {'update': False, 'calc_states': True}
    glmb.extract_states(**extract_kwargs)

    if debug_plots:
        glmb.plot_states_labels([0, 1], true_states=global_true,
                                meas_inds=[])
        glmb.plot_card_dist()
        glmb.plot_card_history(time_units='s', time=time)

    print('\tExpecting {} agents'.format(len(true_agents)))
    print('\tmax cardinality {}'.format(np.max([len(s_set) for s_set in glmb.states])))

    assert len(true_agents) == glmb.cardinality, 'Wrong cardinality'


def test_QKF_GSM_GLMB():  # noqa
    print('Test QKF GSM-GLMB')

    rng = rnd.default_rng(global_seed)
    filt_rng = rnd.default_rng(global_seed)

    dt = 1  # s
    t0, t1 = 0, 20 + dt
    prob_detection = 0.99
    prob_survive = 0.98
    use_sqkf = False
    print_interval = 10  # s

    # measurement noise parameters
    m_dfs = (2, 2)
    m_vars = (25, (0.015 * np.pi / 180)**2)

    filt, state_mat, meas_fun = _setup_qkf_gsm(dt, filt_rng, use_sqkf, m_dfs, m_vars)
    b_model = _setup_gsm_birth()

    RFS_base_args = {'prob_detection': prob_detection,
                     'prob_survive': prob_survive, 'in_filter': filt,
                     'birth_terms': b_model, 'clutter_den': 1**-7,
                     'clutter_rate': 1**-7}
    GLMB_args = {'req_births': len(b_model) + 1, 'req_surv': 1000,
                 'req_upd': 800, 'prune_threshold': 10**-5, 'max_hyps': 1000}
    glmb = tracker.GSMGeneralizedLabeledMultiBernoulli(**GLMB_args,
                                                       **RFS_base_args)

    time = np.arange(t0, t1, dt)
    true_agents = []
    global_true = []
    print('\tStarting sim')
    for kk, tt in enumerate(time):
        if np.mod(kk, int(print_interval / dt)) == 0:
            print('\t\t{:.2f}'.format(tt))
            sys.stdout.flush()

        true_agents = _update_true_agents_gsm(true_agents, tt, b_model, rng,
                                              state_mat)
        global_true.append(deepcopy(true_agents))

        filt_args_pred = {}
        glmb.predict(tt, filt_args=filt_args_pred)

        meas_in = _gen_meas_gsm(tt, true_agents, filt.proc_noise, meas_fun,
                                m_dfs, m_vars, rng)

        filt_args_cor = {}
        glmb.correct(tt, meas_in, filt_args=filt_args_cor)

        extract_kwargs = {'update': True, 'calc_states': False}
        glmb.cleanup(extract_kwargs=extract_kwargs)

    extract_kwargs = {'update': False, 'calc_states': True}
    glmb.extract_states(**extract_kwargs)

    if debug_plots:
        glmb.plot_states_labels([0, 1], true_states=global_true,
                                meas_inds=[])
        glmb.plot_card_dist()
        glmb.plot_card_history(time_units='s', time=time)

    print('\tExpecting {} agents'.format(len(true_agents)))
    print('\tmax cardinality {}'.format(np.max([len(s_set) for s_set in glmb.states])))

    assert len(true_agents) == glmb.cardinality, 'Wrong cardinality'


def test_SQKF_GSM_GLMB():  # noqa
    print('Test SQKF GSM-GLMB')

    rng = rnd.default_rng(global_seed)
    filt_rng = rnd.default_rng(global_seed)

    dt = 1  # s
    t0, t1 = 0, 20 + dt
    prob_detection = 0.99
    prob_survive = 0.98
    use_sqkf = True
    print_interval = 10  # s

    # measurement noise parameters
    m_dfs = (2, 2)
    m_vars = (25, (0.015 * np.pi / 180)**2)

    filt, state_mat, meas_fun = _setup_qkf_gsm(dt, filt_rng, use_sqkf, m_dfs, m_vars)
    b_model = _setup_gsm_birth()

    RFS_base_args = {'prob_detection': prob_detection,
                     'prob_survive': prob_survive, 'in_filter': filt,
                     'birth_terms': b_model, 'clutter_den': 1**-7,
                     'clutter_rate': 1**-7}
    GLMB_args = {'req_births': len(b_model) + 1, 'req_surv': 1000,
                 'req_upd': 800, 'prune_threshold': 10**-5, 'max_hyps': 1000}
    glmb = tracker.GSMGeneralizedLabeledMultiBernoulli(**GLMB_args,
                                                       **RFS_base_args)

    time = np.arange(t0, t1, dt)
    true_agents = []
    global_true = []
    print('\tStarting sim')
    for kk, tt in enumerate(time):
        if np.mod(kk, int(print_interval / dt)) == 0:
            print('\t\t{:.2f}'.format(tt))
            sys.stdout.flush()

        true_agents = _update_true_agents_gsm(true_agents, tt, b_model, rng,
                                              state_mat)
        global_true.append(deepcopy(true_agents))

        filt_args_pred = {}
        glmb.predict(tt, filt_args=filt_args_pred)

        meas_in = _gen_meas_gsm(tt, true_agents, filt.proc_noise, meas_fun,
                                m_dfs, m_vars, rng)

        filt_args_cor = {}
        glmb.correct(tt, meas_in, filt_args=filt_args_cor)

        extract_kwargs = {'update': True, 'calc_states': False}
        glmb.cleanup(extract_kwargs=extract_kwargs)

    extract_kwargs = {'update': False, 'calc_states': True}
    glmb.extract_states(**extract_kwargs)

    if debug_plots:
        glmb.plot_states_labels([0, 1], true_states=global_true,
                                meas_inds=[])
        glmb.plot_card_dist()
        glmb.plot_card_history(time_units='s', time=time)

    print('\tExpecting {} agents'.format(len(true_agents)))
    print('\tmax cardinality {}'.format(np.max([len(s_set) for s_set in glmb.states])))

    assert len(true_agents) == glmb.cardinality, 'Wrong cardinality'


def test_UKF_GSM_GLMB():  # noqa
    print('Test UKF GSM-GLMB')

    rng = rnd.default_rng(global_seed)
    filt_rng = rnd.default_rng(global_seed)

    dt = 1  # s
    t0, t1 = 0, 20 + dt
    prob_detection = 0.99
    prob_survive = 0.98
    print_interval = 10  # s

    # measurement noise parameters
    m_dfs = (2, 2)
    m_vars = (25, (0.015 * np.pi / 180)**2)

    filt, state_mat, meas_fun = _setup_ukf_gsm(dt, filt_rng, m_dfs, m_vars)
    b_model = _setup_gsm_birth()

    RFS_base_args = {'prob_detection': prob_detection,
                     'prob_survive': prob_survive, 'in_filter': filt,
                     'birth_terms': b_model, 'clutter_den': 1**-7,
                     'clutter_rate': 1**-7}
    GLMB_args = {'req_births': len(b_model) + 1, 'req_surv': 1000,
                 'req_upd': 800, 'prune_threshold': 10**-5, 'max_hyps': 1000}
    glmb = tracker.GSMGeneralizedLabeledMultiBernoulli(**GLMB_args,
                                                       **RFS_base_args)

    time = np.arange(t0, t1, dt)
    true_agents = []
    global_true = []
    print('\tStarting sim')
    for kk, tt in enumerate(time):
        if np.mod(kk, int(print_interval / dt)) == 0:
            print('\t\t{:.2f}'.format(tt))
            sys.stdout.flush()

        true_agents = _update_true_agents_gsm(true_agents, tt, b_model, rng,
                                              state_mat)
        global_true.append(deepcopy(true_agents))

        filt_args_pred = {}
        glmb.predict(tt, filt_args=filt_args_pred)

        meas_in = _gen_meas_gsm(tt, true_agents, filt.proc_noise, meas_fun,
                                m_dfs, m_vars, rng)

        filt_args_cor = {}
        glmb.correct(tt, meas_in, filt_args=filt_args_cor)

        extract_kwargs = {'update': True, 'calc_states': False}
        glmb.cleanup(extract_kwargs=extract_kwargs)

    extract_kwargs = {'update': False, 'calc_states': True}
    glmb.extract_states(**extract_kwargs)

    if debug_plots:
        glmb.plot_states_labels([0, 1], true_states=global_true,
                                meas_inds=[])
        glmb.plot_card_dist()
        glmb.plot_card_history(time_units='s', time=time)

    print('\tExpecting {} agents'.format(len(true_agents)))
    print('\tmax cardinality {}'.format(np.max([len(s_set) for s_set in glmb.states])))

    assert len(true_agents) == glmb.cardinality, 'Wrong cardinality'


# %% main
if __name__ == "__main__":
    from timeit import default_timer as timer
    plt.close('all')

    debug_plots = True

    start = timer()

    # test_PHD()
    # test_CPHD()

    # test_GLMB()
    # test_STM_GLMB()
    # test_SMC_GLMB()
    # test_USMC_GLMB()
    # test_MCMC_USMC_GLMB()

    # test_JGLMB()
    # test_STM_JGLMB()
    # test_SMC_JGLMB()
    test_USMC_JGLMB()
    # test_MCMC_USMC_JGLMB()
    # test_QKF_JGLMB()
    # test_SQKF_JGLMB()
    # test_UKF_JGLMB()
    # test_QKF_GSM_JGLMB()
    # test_SQKF_GSM_JGLMB()
    test_UKF_GSM_JGLMB()

    # test_QKF_GLMB()
    # test_SQKF_GLMB()
    # test_UKF_GLMB()

    # test_QKF_GSM_GLMB()
    # test_SQKF_GSM_GLMB()
    # test_UKF_GSM_GLMB()

    end = timer()
    print('{:.2f} s'.format(end - start))
    print('Close all plots to exit')
    plt.show()
