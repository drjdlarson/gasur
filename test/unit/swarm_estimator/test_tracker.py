import numpy as np
import numpy.testing as test

import gncpy.filters as filters
import gasur.swarm_estimator.tracker as trckr
from gasur.utilities.distributions import GaussianMixture


class TestGeneralizedLabeledMultiBernoulli:
    def test_predict(self):
        dt = 1
        filt = filters.ExtendedKalmanFilter()
        filt.cov = []
        filt.meas_mat = []
        filt.meas_noise = []
        filt.proc_map = []
        filt.proc_cov = []

        # returns x_dot
        def f0(x, u, **kwargs):
            return x[2]

        # returns y_dot
        def f1(x, u, **kwargs):
            return x[3]

        # returns x_dot_dot
        def f2(x, u, **kwargs):
            return -x[4] * x[3]

        # returns y_dot_dot
        def f3(x, u, **kwargs):
            return x[4] * x[2]

        # returns omega_dot
        def f4(x, u, **kwargs):
            return 0

        filt.dyn_fncs = [f0, f1, f2, f3, f4]

        glmb = trckr.GeneralizedLabeledMultiBernoulli()
        glmb.filter = filt
        glmb.prob_detection = 0.98
        glmb.prob_survive = 0.99

        mu = [np.array([-1500, 0, 250, 0, 0]).reshape((3, 1))]
        cov = [np.diag(np.array([50, 50, 50, 50, 6 * (np.pi / 180)]))]
        gm0 = GaussianMixture(means=mu, covariances=cov, weights=[1])
        mu = [np.array([-250, 0, 1000, 0, 0])]
        gm1 = GaussianMixture(means=mu, covariances=cov, weights=[1])
        mu = [np.array([250, 0, 750, 0, 0])]
        gm2 = GaussianMixture(means=mu, covariances=cov, weights=[1])
        mu = [np.array([1000, 0, 1500, 0, 0])]
        gm3 = GaussianMixture(means=mu, covariances=cov, weights=[1])

        glmb.birth_terms = [(gm0, 0.02), (gm1, 0.02), (gm2, 0.03), (gm3, 0.03)]
        glmb.req_births = 5
        glmb.req_surv = 3000

        glmb.predict(time_step=0, dt=dt)

        assert 0
