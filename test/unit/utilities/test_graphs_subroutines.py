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
    test.assert_equal(S0, exp_S0)


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
    test.assert_equal(S0, exp_S0)
