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

    def test_k_shortest(self):
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


def test_murty_m_best1():
    cm = np.array([[26.1142509467173, 26.1142509467173, 26.1142509467173,
                   26.1142509467173]])
    m = 256

    (assigns, costs) = graphs.murty_m_best(cm, m)

    exp_assigns = np.array([np.nan, 0, 1, 2, 3]).reshape((5, 1))
    exp_costs = np.array([0, 26.1142509467173, 26.1142509467173,
                          26.1142509467173, 26.1142509467173])

    test.assert_array_equal(assigns, exp_assigns)
    test.assert_array_equal(costs, exp_costs)


def test_murty_m_best2():
    cm = np.array([[26.1142509467173, 26.1142509467173, 26.1142509467173,
                    10.6392155935140, 26.1142509467173, 26.1142509467173],
                   [-6.82765666982308, 26.1142509467173, 26.1142509467173,
                    26.1142509467173, 26.1142509467173, 26.1142509456407]])
    m = 69

    (assigns, costs) = graphs.murty_m_best(cm, m)

    exp_assigns = np.array([[np.nan, 0],
                            [np.nan, np.nan],
                            [3, 0],
                            [3, np.nan],
                            [1, 0],
                            [2, 0],
                            [4, 0],
                            [5, 0],
                            [np.nan, 5],
                            [1, np.nan],
                            [2, np.nan],
                            [4, np.nan],
                            [0, np.nan],
                            [5, np.nan],
                            [np.nan, 1],
                            [np.nan, 2],
                            [np.nan, 3],
                            [np.nan, 4],
                            [3, 5],
                            [3, 1],
                            [3, 2],
                            [3, 4],
                            [1, 5],
                            [2, 5],
                            [4, 5],
                            [0, 5],
                            [5, 1],
                            [1, 2],
                            [2, 1],
                            [4, 1],
                            [0, 1],
                            [5, 2],
                            [1, 3],
                            [2, 3],
                            [4, 2],
                            [0, 2],
                            [5, 3],
                            [1, 4],
                            [2, 4],
                            [4, 3],
                            [0, 3],
                            [5, 4],
                            [0, 4]])
    exp_costs = np.array([-6.82765666982308, 0, 3.81155892369094,
                          10.6392155935140, 19.2865942768942, 19.2865942768942,
                          19.2865942768942, 19.2865942768942, 26.1142509456407,
                          26.1142509467173, 26.1142509467173, 26.1142509467173,
                          26.1142509467173, 26.1142509467173, 26.1142509467173,
                          26.1142509467173, 26.1142509467173, 26.1142509467173,
                          36.7534665391547, 36.7534665402313, 36.7534665402313,
                          36.7534665402313, 52.2285018923580, 52.2285018923580,
                          52.2285018923580, 52.2285018923580, 52.2285018934345,
                          52.2285018934345, 52.2285018934345, 52.2285018934345,
                          52.2285018934345, 52.2285018934345, 52.2285018934345,
                          52.2285018934345, 52.2285018934345, 52.2285018934345,
                          52.2285018934345, 52.2285018934345, 52.2285018934345,
                          52.2285018934345, 52.2285018934345, 52.2285018934345,
                          52.2285018934345])

    test.assert_array_equal(assigns, exp_assigns)
    test.assert_array_almost_equal(costs, exp_costs)


def test_murty_m_best3():
    cm = np.array([[-6.59485513, 26.11425095, -6.72357331,  8.46214993]])
    m = 322

    (assigns, costs) = graphs.murty_m_best(cm, m)

    exp_assigns = np.array([3, 1, 0, 4, 2]).reshape((5, 1))
    exp_costs = np.array([-6.72357330218789, -6.59485513523219, 0,
                          8.46214993805601, 26.1142509467173])

    test.assert_array_equal(assigns, exp_assigns)
    test.assert_array_almost_equal(costs, exp_costs)
