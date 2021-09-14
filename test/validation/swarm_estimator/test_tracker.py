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
    filt.dof = 3

    return filt


def _setup_double_int_pf(dt, rng):
    m_noise = 0.02
    p_noise = 0.2

    def meas_likelihood(meas, est, *args):
        # z = ((meas - est) / m_noise)
        # p1 = stats.norm.pdf(z[0].item())
        # p2 = stats.norm.pdf(z[1].item())
        return stats.multivariate_normal.pdf(meas.flatten(), mean=est.flatten(),
                                             cov=m_noise**2 * np.eye(2))

    def proposal_sampling_fnc(x, rng):
        noise = p_noise * np.array([0, 0, 1, 1]) * rng.standard_normal(4)
        return x + noise.reshape((4, 1))

    def proposal_fnc(x_hat, cond, *args):
        return 1

    doubleInt = gdyn.DoubleIntegrator()
    filt = gfilts.ParticleFilter()
    filt.set_state_model(dyn_obj=doubleInt)
    filt.set_measurement_model(meas_fun=_meas_mat_fun_nonlin)

    filt.proc_noise = doubleInt.get_dis_process_noise_mat(dt, np.array([[p_noise**2]]))
    filt.meas_noise = m_noise**2 * np.eye(2)

    filt.meas_likelihood_fnc = meas_likelihood
    filt.proposal_sampling_fnc = proposal_sampling_fnc
    filt.proposal_fnc = proposal_fnc

    return filt


def _setup_double_int_upf(dt, rng):
    m_noise = 0.02
    p_noise = 0.2

    doubleInt = gdyn.DoubleIntegrator()

    filt = gfilts.UnscentedParticleFilter()
    filt.set_state_model(dyn_obj=doubleInt)
    filt.set_measurement_model(meas_mat=_meas_mat)

    filt.proc_noise = doubleInt.get_dis_process_noise_mat(dt, np.array([[p_noise**2]]))
    filt.meas_noise = m_noise**2 * np.eye(2)

    def meas_likelihood(meas, est, *args):
        return stats.multivariate_normal.pdf(meas.flatten(), mean=est.flatten(),
                                             cov=m_noise**2 * np.eye(2))

    def proposal_sampling_fnc(x, rng):
        val = rng.multivariate_normal(x.flatten(), filt.proc_noise).reshape(x.shape)
        return val
        # noise = p_noise * np.array([0, 0, 1, 1]) * rng.standard_normal(4)
        # return x + noise.reshape((4, 1))

    def proposal_fnc(x_hat, cond, *args):
        return 1

    filt.meas_likelihood_fnc = meas_likelihood
    filt.proposal_sampling_fnc = proposal_sampling_fnc
    filt.proposal_fnc = proposal_fnc

    return filt


def _setup_phd_double_int_birth():
    mu = [np.array([10., 0., 0., 0.]).reshape((4, 1))]
    cov = [np.diag(np.array([1, 1, 1, 1]))**2]
    gm0 = GaussianMixture(means=mu, covariances=cov, weights=[1])

    return [gm0, ]


def _setup_gm_glmb_double_int_birth():
    mu = [np.array([10., 0., 0., 1.]).reshape((4, 1))]
    cov = [np.diag(np.array([1, 1, 1, 1]))**2]
    gm0 = GaussianMixture(means=mu, covariances=cov, weights=[1])

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
    means = [np.array([10., 0., 0., 2.]).reshape((4, 1))]
    cov = np.diag(np.array([1, 1, 1, 1]))**2
    b_probs = [0.005, ]
    alpha = 0.5
    kappa = 3

    birth_terms = []
    for (m, p) in zip(means, b_probs):
        distrib = gdistrib.ParticleDistribution()
        spread = 2 * np.sqrt(np.diag(cov)).reshape(m.shape)
        l_bnd = m - spread / 2
        for ii in range(0, num_parts):
            part = gdistrib.Particle()
            part.point = l_bnd + spread * rng.random(m.shape)
            part.uncertainty = cov.copy()
            part.sigmaPoints = gdistrib.SigmaPoints(alpha=alpha, kappa=kappa,
                                                    n=m.size)
            part.sigmaPoints.init_weights()
            part.sigmaPoints.update_points(part.point, part.uncertainty)
            distrib.add_particle(part, 1 / num_parts)

        birth_terms.append((distrib, p))

    return birth_terms


def _gen_meas(tt, true_agents, proc_noise, meas_noise, rng):
    meas_in = []
    for x in true_agents:
        xp = x + (np.sqrt(np.diag(proc_noise)) * rng.standard_normal(1)).reshape((4, 1))
        noise = (np.sqrt(np.diag(meas_noise)) * rng.standard_normal(1)).reshape((2, 1))
        m = _meas_mat_fun(tt, 'useless') @ xp + noise
        meas_in.append(m.copy())

    return meas_in


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

    p = rng.uniform()
    for distrib, w in b_model:
        if p <= w:
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
            inds = np.arange(0, len(distrib.particles))
            ii = rnd.choice(inds, p=distrib.weights)
            out.append(distrib.particles[ii].copy())

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
    for kk, tt in enumerate(time):

        true_agents = _update_true_agents(true_agents, tt, dt, b_model, rng)

        filt_args = {'state_mat_args': state_mat_args}
        phd.predict(tt, filt_args=filt_args)

        meas_in = _gen_meas(tt, true_agents, filt.proc_noise, filt.meas_noise, rng)

        filt_args = {'meas_fun_args': meas_fun_args}
        phd.correct(tt, meas_in, meas_mat_args={}, est_meas_args={},
                    filt_args=filt_args)

        phd.cleanup()
        # phd.prune()
        # phd.merge()
        # phd.cap()
        # phd.extract_states()

    if debug_plots:
        phd.plot_states([0, 1])


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
    for kk, tt in enumerate(time):

        true_agents = _update_true_agents(true_agents, tt, dt, b_model, rng)

        filt_args = {'state_mat_args': state_mat_args}
        phd.predict(tt, filt_args=filt_args)

        meas_in = _gen_meas(tt, true_agents, filt.proc_noise, filt.meas_noise, rng)

        filt_args = {'meas_fun_args': meas_fun_args}
        phd.correct(tt, meas_in, meas_mat_args={}, est_meas_args={},
                    filt_args=filt_args)

        phd.cleanup()
        # phd.prune()
        # phd.merge()
        # phd.cap()
        # phd.extract_states()

    if debug_plots:
        phd.plot_card_time_hist(time_vec=time)
        phd.plot_states([0, 1])


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

    extract_kwargs = {'pred_args': pred_args, 'cor_args': cor_args,
                      'update': False, 'calc_states': True}
    glmb.extract_states(**extract_kwargs)

    if debug_plots:
        glmb.plot_states_labels([0, 1], true_states=global_true,
                                meas_inds=[0, 1])
        glmb.plot_card_dist()
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

    RFS_base_args = {'prob_detection': 0.99, 'prob_survive': 0.98,
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

        meas_in = _gen_meas(tt, true_agents, filt.proc_noise, filt.meas_noise, rng)

        cor_args = {'meas_fun_args': meas_fun_args}
        glmb.correct(tt, meas_in, filt_args=cor_args)

        extract_kwargs = {'update': True, 'calc_states': False}
        glmb.cleanup(extract_kwargs=extract_kwargs)

    extract_kwargs = {'pred_args': pred_args, 'cor_args': cor_args,
                      'update': False, 'calc_states': True}
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

    dt = 0.01
    t0, t1 = 0, 2 + dt
    num_parts = 1000
    prob_detection = 0.99
    prob_survive = 0.98

    filt = _setup_double_int_pf(dt, rng)
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

        pred_args = {'dyn_fun_params': dyn_fun_params, 'sampling_args': (rng, )}
        prob_surv_args = (prob_survive, )
        glmb.predict(tt, prob_surv_args=prob_surv_args, filt_args=pred_args)

        meas_in = _gen_meas(tt, true_agents, filt.proc_noise, filt.meas_noise, rng)

        cor_args = {'meas_fun_args': meas_fun_args, 'rng': rng}
        prob_det_args = (prob_detection, )
        glmb.correct(tt, meas_in, prob_det_args=prob_det_args, rng=rng,
                     filt_args=cor_args)

        extract_kwargs = {'update': True, 'calc_states': False}
        glmb.cleanup(extract_kwargs=extract_kwargs)

    extract_kwargs = {'update': False, 'calc_states': True,
                      'prob_surv_args': prob_surv_args,
                      'prob_det_args': prob_det_args, 'rng': rng,
                      'pred_args': pred_args, 'cor_args': cor_args}
    glmb.extract_states(**extract_kwargs)

    if debug_plots:
        glmb.plot_states_labels([0, 1], true_states=global_true,
                                meas_inds=[0, 1])
        glmb.plot_card_dist()
    print('\tExpecting {} agents'.format(len(true_agents)))

    assert len(true_agents) == glmb.cardinality, 'Wrong cardinality'


@pytest.mark.slow
def test_USMC_GLMB():  # noqa
    print('Test USMC-GLMB')

    rng = rnd.default_rng(global_seed)

    dt = 0.01
    t0, t1 = 0, 2 + dt
    num_parts = 150
    prob_detection = 0.99
    prob_survive = 0.98

    filt = _setup_double_int_upf(dt, rng)
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
        filt_args_cor = {'ukf_kwargs': ukf_kwargs_cor, 'rng': rng,
                         'sampling_args': (rng, ),
                         'meas_fun_args': meas_fun_args}
        glmb.correct(tt, meas_in, prob_det_args=prob_det_args, rng=rng,
                     filt_args=filt_args_cor)

        extract_kwargs = {'update': True, 'calc_states': False}
        glmb.cleanup(extract_kwargs=extract_kwargs)

    extract_kwargs = {'update': False, 'calc_states': True,
                      'prob_surv_args': prob_surv_args,
                      'prob_det_args': prob_det_args, 'rng': rng,
                      'pred_args': filt_args_pred, 'cor_args': filt_args_cor}
    glmb.extract_states(**extract_kwargs)

    if debug_plots:
        glmb.plot_states_labels([0, 1], true_states=global_true,
                                meas_inds=[0, 1])
        glmb.plot_card_dist()
    print('\tExpecting {} agents'.format(len(true_agents)))

    assert len(true_agents) == glmb.cardinality, 'Wrong cardinality'


# %% main
if __name__ == "__main__":
    plt.close('all')

    debug_plots = True

    # test_PHD()
    # test_CPHD()

    # test_GLMB()
    # test_STM_GLMB()
    # test_SMC_GLMB()
    test_USMC_GLMB()


# ---------------------------------------------------------------------------
# old
# ---------------------------------------------------------------------------
# @pytest.mark.incremental
# class TestSMCGeneralizedLabeledMultiBernoulli:
#     def __init__(self, **kwargs):
#         self.rng = rnd.default_rng(1)

#         self.max_time = 3
#         self.dt = 1.0

#         # proc noise
#         self.sig_w = 2.5
#         self.sig_u = np.pi / 180

#         # meas noise
#         self.std_turn = 0.25 * (np.pi / 180)
#         self.std_pos = 5.0

#         self.prob_detection = 0.98
#         self.prob_survive = 0.99

#         self.G = np.array([[self.dt**2 / 2, 0, 0],
#                           [self.dt, 0, 0],
#                           [0, self.dt**2 / 2, 0],
#                           [0, self.dt, 0],
#                           [0, 0, 1]])
#         self.Q = np.diag([self.sig_w, self.sig_w, self.sig_u])**2

#     def compute_prob_detection(self, part_lst, **kwargs):
#         if len(part_lst) == 0:
#             return np.array([])
#         else:
#             inv_std = np.diag(np.array([1. / 2000., 1. / 2000.]))

#             e_sq = np.sum(np.hstack([(inv_std
#                                       @ x[[0, 2], 0].reshape((2, 1)))**2
#                                      for x in part_lst]), axis=0)
#             return self.prob_detection * np.exp(-e_sq / 2.)

#     def compute_prob_survive(self, part_lst, **kwargs):
#         if len(part_lst) == 0:
#             return np.array([])
#         else:
#             return self.prob_survive * np.ones(len(part_lst))

#     def meas_mod(self, state, **kwargs):
#         x_pos = state[0, 0]
#         y_pos = state[2, 0]
#         z1 = np.arctan2(y_pos, x_pos)
#         z2 = np.sqrt(x_pos**2 + y_pos**2)
#         return np.array([[z1], [z2]])

#     def meas_likelihood(self, meas, est, **kwargs):
#         cov = np.array([[self.std_turn**2, 0],
#                         [0, self.std_pos**2]])
#         return stats.multivariate_normal.pdf(meas.copy().reshape(meas.size),
#                                              mean=est.copy().reshape(est.size),
#                                              cov=cov)

#     # \dot{x} = f(x)
#     def cont_dyn(self, x, **kwargs):
#         # returns x_dot
#         def f0(x, **kwargs):
#             return x[1]

#         # returns x_dot_dot
#         def f1(x, **kwargs):
#             return -x[4] * x[3]

#         # returns y_dot
#         def f2(x, **kwargs):
#             return x[3]

#         # returns y_dot_dot
#         def f3(x, **kwargs):
#             return x[4] * x[1]

#         # returns omega_dot
#         def f4(x, **kwargs):
#             return 0

#         out = np.zeros(x.shape)
#         for ii, f in enumerate([f0, f1, f2, f3, f4]):
#             out[ii] = f(x, **kwargs)
#         return out

#     # x_{k + 1} = f(x_{k})
#     def dyn_fnc(self, x, noise_on=False, **kwargs):
#         ctrl = np.zeros((2, 1))
#         ns = rk4(self.cont_dyn, x.copy(), self.dt, cur_input=ctrl)
#         if noise_on:
#             dim = self.Q.shape[0]
#             samp = self.rng.multivariate_normal(np.zeros(dim),
#                                                 self.Q).reshape((dim, 1))
#             ns += self.G @ samp
#         return ns

#     def init_glmb(self, glmb, birth_terms):
#         glmb.compute_prob_detection = self.compute_prob_detection
#         glmb.compute_prob_survive = self.compute_prob_survive
#         glmb.prob_detection = self.prob_detection
#         glmb.prob_survive = self.prob_survive
#         glmb.birth_terms = birth_terms
#         glmb.req_births = 5
#         glmb.req_surv = 5000
#         glmb.req_upd = 5000
#         glmb.gating_on = False
#         glmb.clutter_rate = 0.0000001  # 10
#         glmb.clutter_den = 1 / (np.pi * 2000)

#     def prop_states(self, k, true_states, noise_on=True):
#         new_states = []
#         for s in true_states:
#             ns = self.dyn_fnc(s.copy(), noise_on=noise_on)
#             new_states.append(ns)

#         wturn = 2 * np.pi / 180
#         if k == 0:
#             s = np.array([1000 + 3.8676, -10, 1500 - 11.7457, -10, wturn / 8])
#             new_states.append(s.reshape((5, 1)))

#         return new_states

#     def gen_meas(self, true_states, clutter_rate):
#         meas = []
#         for s in true_states:
#             m = self.meas_mod(s)
#             cov = np.diag([self.std_turn, self.std_pos])**2
#             m = self.rng.multivariate_normal(m.flatten(), cov).reshape(m.shape)
#             meas.append(m)

#         num_clutt = self.rng.poisson(clutter_rate)
#         for ii in range(0, num_clutt):
#             samp = self.rng.standard_normal(size=(2, 1))
#             m = np.array([[np.pi], [2000]]) * samp
#             meas.append(m)

#         return meas

#     def proposal_sampling_fnc(self, x, **kwargs):
#         cov = self.Q  # proc cov
#         mean = np.zeros(cov.shape[0])
#         samp = self.rng.multivariate_normal(mean, cov)
#         samp = samp.reshape((samp.size, 1))
#         return x + self.G @ samp

#     def proposal_fnc(self, x_hat, cond, **kwargs):
#         cov = self.Q  # proc cov
#         x_norm = (self.G.T @ (x_hat - cond)).flatten()
#         mean = np.zeros(x_norm.size)
#         return stats.multivariate_normal.pdf(x_norm, mean=mean, cov=cov)

#     def test_basic_PF(self):
#         num_parts = 2000

#         means = [np.array([-1500., 0., 250., 0., 0.]).reshape((5, 1)),
#                  np.array([-250., 0., 1000., 0., 0.]).reshape((5, 1)),
#                  np.array([250., 0., 750., 0., 0.]).reshape((5, 1)),
#                  np.array([1000., 0., 1500., 0., 0.]).reshape((5, 1))]
#         cov = np.diag(np.array([50, 50, 50, 50, 6 * (np.pi / 180)]))**2
#         b_probs = [0.02, 0.02, 0.03, 0.03]
#         birth_terms = []
#         for (m, p) in zip(means, b_probs):
#             distrib = distributions.ParticleDistribution()
#             spread = 2 * np.sqrt(np.diag(cov)).reshape(m.shape)
#             l_bnd = m - spread / 2
#             for ii in range(0, num_parts):
#                 part = distributions.Particle()
#                 part.point = l_bnd + spread * self.rng.random(m.shape)
#                 w = 1 / num_parts
#                 distrib.add_particle(part, w)

#             birth_terms.append((distrib, p))

#         filt = filters.ParticleFilter()
#         filt.set_meas_model(self.meas_mod)
#         filt.dyn_fnc = self.dyn_fnc
#         filt.meas_noise = np.diag([self.std_turn**2, self.std_pos**2])
#         filt.set_proc_noise(mat=self.G @ self.Q @ self.G.T)
#         filt.meas_likelihood_fnc = self.meas_likelihood
#         filt.proposal_sampling_fnc = self.proposal_sampling_fnc
#         filt.proposal_fnc = self.proposal_fnc

#         glmb = tracker.SMCGeneralizedLabeledMultiBernoulli()
#         glmb.filter = filt
#         self.init_glmb(glmb, birth_terms)

#         true_states = []
#         total_true = []
#         for k in range(0, self.max_time):
#             true_states = self.prop_states(k, true_states, noise_on=True)
#             total_true.append(deepcopy(true_states))

#             # generate measurements
#             meas = self.gen_meas(true_states, glmb.clutter_rate)

#             # run filter
#             glmb.predict(time_step=k)
#             glmb.correct(meas=meas)
#             glmb.prune()
#             glmb.cap()
#             glmb.extract_states()

#         assert glmb.cardinality == 1, "Cardinality does not match"

#     def test_UPF(self):
#         num_parts = 500
#         alpha = 1
#         kappa = 0

#         means = [np.array([-1500., 0., 250., 0., 0.]).reshape((5, 1)),
#                  np.array([-250., 0., 1000., 0., 0.]).reshape((5, 1)),
#                  np.array([250., 0., 750., 0., 0.]).reshape((5, 1)),
#                  np.array([1000., 0., 1500., 0., 0.]).reshape((5, 1))]
#         cov = np.diag(np.array([25, 10, 25, 10, 6 * (np.pi / 180)]))**2
#         b_probs = [0.02, 0.02, 0.02, 0.03]
#         birth_terms = []
#         sigPointRef = distributions.SigmaPoints(alpha=alpha, kappa=kappa,
#                                                 n=means[0].size)
#         sigPointRef.init_weights()
#         for (m, p) in zip(means, b_probs):
#             distrib = distributions.ParticleDistribution()
#             spread = 2 * np.sqrt(np.diag(cov)).reshape(m.shape)
#             l_bnd = m - spread / 2
#             for ii in range(0, num_parts):
#                 part = distributions.Particle()
#                 part.point = l_bnd + spread * self.rng.random(m.shape)
#                 part.uncertainty = cov.copy()
#                 part.sigmaPoints = deepcopy(sigPointRef)
#                 part.sigmaPoints.update_points(m.copy(), cov)
#                 w = 1 / num_parts
#                 distrib.add_particle(part, w)

#             birth_terms.append((distrib, p))

#         filt = filters.UnscentedParticleFilter()
#         filt.set_meas_model(self.meas_mod)
#         filt.dyn_fnc = self.dyn_fnc
#         filt.meas_noise = np.diag([self.std_turn**2, self.std_pos**2])
#         filt.set_proc_noise(mat=self.G @ self.Q @ self.G.T)
#         filt.meas_likelihood_fnc = self.meas_likelihood
#         filt.proposal_sampling_fnc = self.proposal_sampling_fnc
#         filt.proposal_fnc = self.proposal_fnc

#         glmb = tracker.SMCGeneralizedLabeledMultiBernoulli()
#         glmb.filter = filt
#         self.init_glmb(glmb, birth_terms)

#         true_states = []
#         total_true = []
#         for k in range(0, self.max_time):
#             true_states = self.prop_states(k, true_states, noise_on=True)
#             total_true.append(deepcopy(true_states))

#             # generate measurements
#             meas = self.gen_meas(true_states, glmb.clutter_rate)

#             # run filter
#             glmb.predict(time_step=k)
#             glmb.correct(meas=meas)
#             glmb.prune()
#             glmb.cap()
#             glmb.extract_states()

#         assert glmb.cardinality == 1, "Cardinality does not match"

#     def test_UPFMCMC(self):
#         num_parts = 100
#         alpha = 1
#         kappa = 0

#         means = [np.array([-1500., 0., 250., 0., 0.]).reshape((5, 1)),
#                  np.array([-250., 0., 1000., 0., 0.]).reshape((5, 1)),
#                  np.array([250., 0., 750., 0., 0.]).reshape((5, 1)),
#                  np.array([1000., 0., 1500., 0., 0.]).reshape((5, 1))]
#         cov = np.diag(np.array([10, 5, 10, 5, 3 * (np.pi / 180)]))**2
#         b_probs = [0.02, 0.02, 0.02, 0.03]
#         birth_terms = []
#         sigPointRef = distributions.SigmaPoints(alpha=alpha, kappa=kappa,
#                                                 n=means[0].size)
#         sigPointRef.init_weights()
#         for (m, p) in zip(means, b_probs):
#             distrib = distributions.ParticleDistribution()
#             spread = 2 * np.sqrt(np.diag(cov)).reshape(m.shape)
#             l_bnd = m - spread / 2
#             for ii in range(0, num_parts):
#                 part = distributions.Particle()
#                 part.point = l_bnd + spread * self.rng.random(m.shape)
#                 part.uncertainty = cov.copy()
#                 part.sigmaPoints = deepcopy(sigPointRef)
#                 part.sigmaPoints.update_points(m.copy(), cov)
#                 w = 1 / num_parts
#                 distrib.add_particle(part, w)

#             birth_terms.append((distrib, p))

#         filt = filters.UnscentedParticleFilter()
#         filt.use_MCMC = True
#         filt.set_meas_model(self.meas_mod)
#         filt.dyn_fnc = self.dyn_fnc
#         filt.meas_noise = np.diag([self.std_turn**2, self.std_pos**2])
#         filt.set_proc_noise(mat=self.G @ self.Q @ self.G.T)
#         filt.meas_likelihood_fnc = self.meas_likelihood
#         filt.proposal_sampling_fnc = self.proposal_sampling_fnc
#         filt.proposal_fnc = self.proposal_fnc

#         glmb = tracker.SMCGeneralizedLabeledMultiBernoulli()
#         glmb.filter = filt
#         self.init_glmb(glmb, birth_terms)

#         true_states = []
#         total_true = []
#         for k in range(0, self.max_time):
#             true_states = self.prop_states(k, true_states, noise_on=True)
#             total_true.append(deepcopy(true_states))

#             # generate measurements
#             meas = self.gen_meas(true_states, glmb.clutter_rate)

#             # run filter
#             glmb.predict(time_step=k)
#             glmb.correct(meas=meas)
#             glmb.prune()
#             glmb.cap()
#             glmb.extract_states()

#         # print("Cardinality:")
#         # print(glmb.cardinality)
#         # print("labels")
#         # print(glmb.labels)

#         # f_hndl = glmb.plot_states_labels([0, 2], true_states=total_true,
#         #                                  sig_bnd=None)

#         # meas_plt_x = []
#         # meas_plt_y = []
#         # for tt in glmb._meas_tab:
#         #     for m in tt:
#         #         theta = m[0]
#         #         r = m[1]
#         #         x = r * np.cos(theta)
#         #         y = r * np.sin(theta)
#         #         meas_plt_x.append(x)
#         #         meas_plt_y.append(y)

#         # color = (128/255, 128/255, 128/255)
#         # f_hndl.axes[0].scatter(meas_plt_x, meas_plt_y, zorder=-1, alpha=0.35,
#         #                        color=color, marker='^')

#         # glmb.plot_card_dist()

#         assert glmb.cardinality == 1, "Cardinality does not match"
