import pytest
import numpy as np
import numpy.testing as test

import gasur.utilities.graphs_subroutines as g_sub


def test_assign_opt1():
    P0 = np.array([26.1142509467173, 26.1142509467173, 26.1142509467173,
                   26.1142509467173, 0])
    (S0, C0) = g_sub.assign_opt(P0)

    exp_C0 = 0
    exp_S0 = np.array([4])

    test.assert_allclose(C0, exp_C0)
    test.assert_array_equal(S0, exp_S0)


def test_assign_opt2():
    P0 = np.array([[32.9419076165403, 32.9419076165403, 32.9419076165403,
                    17.4668722633371, 32.9419076165403, 32.9419076165403,
                    6.82765666982308, 6.82765666982308],
                   [0, 32.9419076165403, 32.9419076165403, 32.9419076165403,
                    32.9419076165403, 32.9419076154638, 6.82765666982308,
                    6.82765666982308]])
    (S0, C0) = g_sub.assign_opt(P0)

    exp_C0 = 6.82765666982308
    exp_S0 = np.array([6, 0])

    test.assert_allclose(C0, exp_C0)
    test.assert_array_equal(S0, exp_S0)


def test_assign_opt3():
    P0 = np.array([[np.inf, np.inf, np.inf, np.inf, np.inf, np.inf, np.inf,
                    np.inf],
                   [np.inf, 32.9419076165403, 32.9419076165403,
                    32.9419076165403, 32.9419076165403, 32.9419076154638,
                    np.inf, 6.82765666982308]])
    (S0, C0) = g_sub.assign_opt(P0)

    exp_C0 = 6.82765666982308
    exp_S0 = np.array([-1, 7])

    test.assert_allclose(C0, exp_C0)
    test.assert_array_equal(S0, exp_S0)


def test_assign_opt4():
    P0 = np.array([[32.2839240000000, 3.21131500000000, 32.2839240000000,
                    32.2839240000000, 32.2839240000000, 32.2839240000000,
                    32.2839240000000, 22.2960390000000, 32.2839240000000,
                    np.inf, np.inf, np.inf],
                   [32.2839240000000, 1.92542300000000, 32.2839240000000,
                    32.2839240000000, 32.2839240000000, 32.2839240000000,
                    32.2839240000000, 11.4484150000000, 32.2839240000000,
                    6.16967300000000, 6.16967300000000, 6.16967300000000],
                   [0, 32.2839240000000, 32.2504870000000, 32.2839240000000,
                    32.2839240000000, 32.2839240000000, 32.2839240000000,
                    32.2839240000000, 32.2839240000000, 6.16967300000000,
                    6.16967300000000, 6.16967300000000]])
    (S0, C0) = g_sub.assign_opt(P0)

    exp_C0 = 9.380988
    exp_S0 = np.array([1, 9, 0])

    test.assert_allclose(C0, exp_C0)
    test.assert_array_equal(S0, exp_S0)
