# -*- coding: utf-8 -*-
"""
Created on Sun Apr 19 10:22:52 2020

@author: ryan4
"""
import numpy as np


def get_state_jacobian(x, u, fncs, **kwargs):
    step_size = kwargs.get('step_size', 0.00001)
    inv_step2 = 1 / (2 * step_size)
    n_states = x.size
    A = np.zeros((n_states, n_states))
    for row in range(0, n_states):
        for col in range(0, n_states):
            x_r = x_l = x
            x_r[col] += step_size
            x_l[col] -= step_size
            A[row, col] = inv_step2 * (fncs[row](x_r, u, **kwargs)
                                       - fncs[row](x_l, u, **kwargs))
    return A


def get_input_jacobian(x, u, fncs, **kwargs):
    step_size = kwargs.get('step_size', 0.00001)
    inv_step2 = 1 / (2 * step_size)
    n_inputs = u.size
    B = np.zeros((n_inputs, n_inputs))
    for row in range(0, n_inputs):
        for col in range(0, n_inputs):
            u_r = u_l = u
            u_r[col] += step_size
            u_l[col] -= step_size
            B[row, col] = inv_step2 * (fncs[row](x, u_r, **kwargs)
                                       - fncs[row](x, u_l, **kwargs))
    return B
