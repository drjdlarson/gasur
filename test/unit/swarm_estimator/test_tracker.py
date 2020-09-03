import pytest
import numpy as np


@pytest.mark.incremental
class TestGeneralizedLabeledMultiBernoulli:
    def test_predict(self, glmb_tup):
        glmb = glmb_tup[0]
        dt = glmb_tup[1]
        glmb.predict(time_step=0, dt=dt)

        # check that code runs
        assert 1

    def test_correct(self, glmb_tup):
        glmb = glmb_tup[0]
        dt = glmb_tup[1]

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

        glmb.predict(time_step=0, dt=dt)
        glmb.correct(meas=meas)

        # check that code ran
        assert 1

    def test_extract_states(self, glmb_tup):
        glmb = glmb_tup[0]
        dt = glmb_tup[1]

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

        glmb.predict(time_step=0, dt=dt)
        glmb.correct(meas=meas)
        glmb.extract_states(dt=dt)

        # check only 1 time step
        assert len(glmb.states) == 1
        assert len(glmb.labels) == 1

        # check that code ran with no errors, should have a cardinality of 0
        assert len(glmb.states[0]) == 0

        # also means no labels
        assert len(glmb.labels[0]) == 0
