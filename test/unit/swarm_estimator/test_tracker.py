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

        # check that code ran with no errors, should have a cardinality of 0
        assert len(glmb.states[0]) == 0

        # also means no labels
        assert len(glmb.labels[0]) == 0

    def test_correct(self):
        dt = 1

        # setup filter for coordinated turn model
        filt = filters.ExtendedKalmanFilter()

        def meas_fnc(state, **kwargs):
            mag = state[0]**2 + state[2]**2
            sqrt_mag = np.sqrt(mag)
            mat = np.vstack((np.hstack((state[2] / mag, 0, -state[0] / mag,
                                        0, 0)),
                            np.hstack((state[0] / sqrt_mag, 0,
                                       state[2] / sqrt_mag, 0, 0))))
            return mat

        filt.set_meas_mat(fnc=meas_fnc)
        filt.meas_noise = np.diag([(2 * np.pi / 180)**2, 10**2])
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
        glmb.req_upd = 3000
        glmb.clutter_rate = 15
        glmb.clutter_den = 1 / (np.pi * 2000)

        lst = [[0.633157293, 1773.703654],
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
        meas = []
        for z in lst:
            meas.append(np.array(z).reshape((2, 1)))

        glmb.predict(time_step=0)

        glmb.correct(meas=meas)

        # check only 1 time step
        assert len(glmb.states) == 0
        assert len(glmb.labels) == 0

        # check that code ran with no errors, should have a cardinality of 0
        assert len(glmb.states[0]) == 0

        # also means no labels
        assert len(glmb.labels[0]) == 0
