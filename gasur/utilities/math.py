# -*- coding: utf-8 -*-
import numpy as np


def get_state_space_derivatives(fnc, state, ctrl, dt, step_size=1*10**(-5)):
    state_len = state.size
    ctrl_len = ctrl.size

    state_derivatives = np.zeros((state_len, state_len))
    for col in range(0, state_len):
        step = np.zeros(state_len)
        step[col] = step_size

        forward_point = state + step
        f_forward = fnc(state, ctrl, dt)
        backward_point = state - step
        f_backward = fnc(state, ctrl, dt)

        state_derivatives[:, col] = 0.5 * (f_forward - f_backward) \
            / step_size

    ctrl_derivatives = np.zeros((ctrl_len, ctrl_len))
    for col in range(0, ctrl_len):
        step = np.zeros(ctrl_len)
        step[col] = step_size

        forward_point = ctrl + step
        backward_point = ctrl - step

        ctrl_derivatives[:, col] = 0.5 * (forward_point - backward_point) \
            / step_size

    return state_derivatives, ctrl_derivatives
