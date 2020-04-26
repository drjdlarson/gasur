# -*- coding: utf-8 -*-
"""
Created on Thu Apr 23 19:12:55 2020

@author: ryan4
"""
import numpy as np
import numpy.testing as test

import gasur.utilities.math as math


def test_get_state_jacobian(func_list, x_point, u_point):
    A = math.get_state_jacobian(x_point, u_point, func_list)
    A_exp = np.vstack((np.array([0.42737988, -21.46221643, 8.1000000]),
                       np.array([6.00000, -18.75000000, 8.1000000]),
                       np.array([0.88200, 5.400000, -0.989992497])))
    test.assert_allclose(A, A_exp)


def test_get_input_jacobian(func_list, x_point, u_point):
    B = math.get_input_jacobian(x_point, u_point, func_list)
    B_exp = np.vstack((np.array([1, 0]),
                       np.array([2, -0.5]),
                       np.array([0.877582562, 0])))
    test.assert_allclose(B, B_exp)
