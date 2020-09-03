import pytest
import numpy as np
import numpy.testing as test

import gasur.utilities.graphs as graphs


@pytest.mark.incremental
class TestKShortesst:
    def test_bfm_shortest_path1(self):
        eps = np.finfo(float).eps
        ncm = np.array([[0., 3.8918, 3.8918, 3.4761, 3.4761, eps],
                        [0., 0., 3.8918, 3.4761, 3.4761, eps],
                        [0., 0., 0., 3.4761, 3.4761, eps],
                        [0., 0., 0., 0., 3.4761, eps],
                        [0., 0., 0., 0., 0., eps],
                        [0., 0., 0., 0., 0., 0]])
        src = 0
        dst = 5

        (cost, path, pred) = graphs.bfm_shortest_path(ncm, src, dst)

        exp_cost = eps
        exp_path = [0, 5]

        test.assert_approx_equal(cost, exp_cost)
        test.assert_array_equal(np.array(path), np.array(exp_path))

    def test_bfm_shortest_path2(self):
        eps = np.finfo(float).eps
        ncm = np.array([[0., 3.89182030, 3.89182030, 3.47609869,
                         3.47609869e+00, np.inf],
                       [0., 0., 3.89182030, 3.47609869, 3.47609869, eps],
                       [0., 0., 0., 3.47609869, 3.47609869, eps],
                       [0., 0., 0., 0., 3.47609869, eps],
                       [0., 0., 0., 0., 0., eps],
                       [0., 0., 0., 0., 0., 0.]])
        src = 0
        dst = 5

        (cost, path, _) = graphs.bfm_shortest_path(ncm, src, dst)

        exp_cost = 3.47609868983527
        exp_path = [0, 3, 5]

        test.assert_approx_equal(cost, exp_cost)
        test.assert_array_equal(np.array(path), np.array(exp_path))

    def test_k_shortest1(self):
        eps = np.finfo(float).eps
        k = 5
        log_cost = np.array([3.89182029811063, 3.89182029811063,
                             3.47609868983527, 3.47609868983527])

        (paths, costs) = graphs.k_shortest(log_cost, k)

        exp_paths = [[], [3], [2], [1], [0]]
        exp_costs = [eps, 3.47609868983527, 3.47609868983527,
                     3.89182029811063, 3.89182029811063]

        assert len(paths) == len(exp_paths)
        for ii in range(0, len(paths)):
            test.assert_array_equal(np.array(paths[ii]),
                                    np.array(exp_paths[ii]))

        test.assert_array_almost_equal(np.array(costs).squeeze(),
                                       np.array(exp_costs))

    def test_k_shortest2(self):
        eps = np.finfo(float).eps
        k = 24
        log_cost = np.array([-4.595119850134589])

        (paths, costs) = graphs.k_shortest(log_cost, k)

        exp_paths = [[0], []]
        exp_costs = [-4.59511985013459, eps]

        assert len(paths) == len(exp_paths)
        for ii in range(0, len(paths)):
            test.assert_array_equal(np.array(paths[ii]),
                                    np.array(exp_paths[ii]))

        test.assert_array_almost_equal(np.array(costs).squeeze(),
                                       np.array(exp_costs))


def test_murty_m_best1():
    cm = np.array([[-6.59485513, 26.11425095, -6.72357331,  8.46214993]])
    m = 322

    (assigns, costs) = graphs.murty_m_best(cm, m)

    exp_assigns = np.array([3, 1, 0, 4, 2]).reshape((5, 1))
    exp_costs = np.array([-6.72357330218789, -6.59485513523219, 0,
                          8.46214993805601, 26.1142509467173])

    test.assert_array_equal(assigns, exp_assigns)
    test.assert_array_almost_equal(costs, exp_costs)


def test_murty_m_best2():
    cm = np.array([[-5.80280392, 26.11425095, 17.47669203, 26.11425095,
                    26.11425095, 1.48712218, 26.11425095, 26.11097866,
                    26.11425095]])
    m = 1

    (assigns, costs) = graphs.murty_m_best(cm, m)

    exp_assigns = np.array([1]).reshape((1, 1))
    exp_costs = np.array([-5.80280392])

    test.assert_array_equal(assigns, exp_assigns)
    test.assert_array_almost_equal(costs, exp_costs)


def test_murty_m_best3():
    cm = np.array([[-5.80280392000000, 26.1142509500000, 17.4766920300000,
                    26.1142509500000, 26.1142509500000, 1.48712218000000,
                    26.1142509500000, 26.1109786600000, 26.1142509500000],
                   [-6.80184202000000, 26.1142509500000, 26.1142504300000,
                    26.1142509500000, 26.1142509500000, 15.0267480200000,
                    26.1142509500000, 26.1142509500000, 26.1142509500000]])
    m = 1

    (assigns, costs) = graphs.murty_m_best(cm, m)

    exp_assigns = np.array([0, 1]).reshape((1, 2))
    exp_costs = np.array([-6.80184202])

    test.assert_array_equal(assigns, exp_assigns)
    test.assert_array_almost_equal(costs, exp_costs)