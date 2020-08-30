import pytest
import numpy as np
import numpy.testing as test
from scipy.linalg import block_diag

import gncpy.filters as filters
import gasur.swarm_estimator.tracker as trckr
from gasur.utilities.distributions import GaussianMixture


@pytest.mark.incremental
class TestGeneralizedLabeledMultiBernoulli:
    def test_predict(self):
        dt = 1

        # setup filter for coordinated turn model
        filt = filters.ExtendedKalmanFilter()
        sig_w = 15
        sig_u = np.pi / 180
        G = np.array([[dt**2, 0, 0],
                      [dt, 0, 0],
                      [0, dt**2, 0],
                      [0, dt, 0],
                      [0, 0, 1]])
        Q = block_diag(sig_w**2 * np.eye(2), np.array([[sig_u**2]]))
        filt.set_proc_noise(mat=G @ Q @ G.T)

        # returns x_dot
        def f0(x, u, **kwargs):
            return x[1]

        # returns x_dot_dot
        def f1(x, u, **kwargs):
            return -x[4] * x[3]

        # returns y_dot
        def f2(x, u, **kwargs):
            return x[3]

        # returns y_dot_dot
        def f3(x, u, **kwargs):
            return x[4] * x[1]

        # returns omega_dot
        def f4(x, u, **kwargs):
            return 0

        filt.dyn_fncs = [f0, f1, f2, f3, f4]

        glmb = trckr.GeneralizedLabeledMultiBernoulli()
        glmb.filter = filt
        glmb.prob_detection = 0.98
        glmb.prob_survive = 0.99

        mu = [np.array([-1500, 0, 250, 0, 0]).reshape((5, 1))]
        cov = [np.diag(np.array([50, 50, 50, 50, 6 * (np.pi / 180)]))]
        gm0 = GaussianMixture(means=mu, covariances=cov, weights=[1])
        mu = [np.array([-250, 0, 1000, 0, 0]).reshape((5, 1))]
        gm1 = GaussianMixture(means=mu, covariances=cov, weights=[1])
        mu = [np.array([250, 0, 750, 0, 0]).reshape((5, 1))]
        gm2 = GaussianMixture(means=mu, covariances=cov, weights=[1])
        mu = [np.array([1000, 0, 1500, 0, 0]).reshape((5, 1))]
        gm3 = GaussianMixture(means=mu, covariances=cov, weights=[1])

        glmb.birth_terms = [(gm0, 0.02), (gm1, 0.02), (gm2, 0.03), (gm3, 0.03)]
        glmb.req_births = 5
        glmb.req_surv = 3000

        glmb.predict(time_step=0)

        # check that code ran with no errors, values are all private at this
        # point
        assert 1

    def test_correct(self):
        dt = 1

        # setup filter for coordinated turn model
        filt = filters.ExtendedKalmanFilter()
        filt.set_meas_mat(fnc=[])
        filt.meas_noise = []
        sig_w = 15
        sig_u = np.pi / 180
        G = np.array([[dt**2, 0, 0],
                      [dt, 0, 0],
                      [0, dt**2, 0],
                      [0, dt, 0],
                      [0, 0, 1]])
        Q = block_diag(sig_w**2 * np.eye(2), np.array([[sig_u**2]]))
        filt.set_proc_noise(mat=G @ Q @ G.T)

        # returns x_dot
        def f0(x, u, **kwargs):
            return x[1]

        # returns x_dot_dot
        def f1(x, u, **kwargs):
            return -x[4] * x[3]

        # returns y_dot
        def f2(x, u, **kwargs):
            return x[3]

        # returns y_dot_dot
        def f3(x, u, **kwargs):
            return x[4] * x[1]

        # returns omega_dot
        def f4(x, u, **kwargs):
            return 0

        filt.dyn_fncs = [f0, f1, f2, f3, f4]

        glmb = trckr.GeneralizedLabeledMultiBernoulli()
        glmb.filter = filt
        glmb.prob_detection = 0.98
        glmb.prob_survive = 0.99

        mu = [np.array([-1500, 0, 250, 0, 0]).reshape((5, 1))]
        cov = [np.diag(np.array([50, 50, 50, 50, 6 * (np.pi / 180)]))]
        gm0 = GaussianMixture(means=mu, covariances=cov, weights=[1])
        mu = [np.array([-250, 0, 1000, 0, 0]).reshape((5, 1))]
        gm1 = GaussianMixture(means=mu, covariances=cov, weights=[1])
        mu = [np.array([250, 0, 750, 0, 0]).reshape((5, 1))]
        gm2 = GaussianMixture(means=mu, covariances=cov, weights=[1])
        mu = [np.array([1000, 0, 1500, 0, 0]).reshape((5, 1))]
        gm3 = GaussianMixture(means=mu, covariances=cov, weights=[1])

        glmb.birth_terms = [(gm0, 0.02), (gm1, 0.02), (gm2, 0.03), (gm3, 0.03)]
        glmb.req_births = 5
        glmb.req_surv = 3000

        meas = np.array([[]])

        glmb.predict(time_step=0)

        glmb.correct(meas=meas)

        # check that code ran with no errors, values are all private at this
        # point
        assert 1
