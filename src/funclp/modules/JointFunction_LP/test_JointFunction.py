#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# Date          : 2026-04-26
# Author        : Lancelot PINCET
# GitHub        : https://github.com/LancelotPincet
# Library       : funcLP
# Module        : JointFunction

"""
This file allows to test JointFunction

JointFunction : This class defines a way to apply joint models between various models.
"""



# %% Libraries
from corelp import debug
import pytest
import numpy as np
from funclp import Gaussian2D, JointChannel, JointFunction, LM, LSE, Spline2D
debug_folder = debug(__file__)




def _make_joint_problem(xp=np, cuda=False):
    # Always generate reference data on CPU first.
    v_np = np.linspace(-3, 3, 15, dtype=np.float32)
    x_np, y_np = np.meshgrid(v_np, v_np)

    mux_np = np.array([0.25, -0.3], dtype=np.float32)
    muy_np = np.array([-0.15, 0.2], dtype=np.float32)

    dx = np.float32(-0.5)
    dy = np.float32(0.35)

    sig = np.float32(0.9)
    amp = np.float32(8.)
    offset = np.float32(0.5)

    base_np = Gaussian2D(sigx=sig, sigy=sig, amp=1., offset=0.)(x_np, y_np)

    data0_np = Gaussian2D(
        mux=mux_np,
        muy=muy_np,
        sigx=sig,
        sigy=sig,
        amp=amp,
        offset=offset,
    )(x_np, y_np)

    data1_np = Spline2D(
        base_np,
        x_np,
        y_np,
        mux=mux_np + dx,
        muy=muy_np + dy,
        amp=amp,
        offset=offset,
    )(x_np, y_np)

    if xp is np:
        x, y = x_np, y_np
        mux, muy = mux_np, muy_np
        data0, data1 = data0_np, data1_np
    else:
        x, y = xp.asarray(x_np), xp.asarray(y_np)
        mux, muy = xp.asarray(mux_np), xp.asarray(muy_np)
        data0, data1 = xp.asarray(data0_np), xp.asarray(data1_np)

    function0 = Gaussian2D(
        mux=xp.zeros(2, dtype=xp.float32),
        muy=xp.zeros(2, dtype=xp.float32),
        sigx=xp.float32(sig),
        sigy=xp.float32(sig),
        amp=xp.float32(amp),
        offset=xp.float32(offset),
    )

    function1 = Spline2D(
        base_np,
        x_np,
        y_np,
        mux=xp.zeros(2, dtype=xp.float32),
        muy=xp.zeros(2, dtype=xp.float32),
        amp=xp.float32(amp),
        offset=xp.float32(offset),
    )

    affine1 = np.array([
        [1., 0., -dx],
        [0., 1., -dy],
        [0., 0., 1.],
    ], dtype=np.float32)

    function = JointFunction(
        [
            JointChannel(function0, prefix='gaus'),
            JointChannel(function1, prefix='spline', affine={('x', 'y'): affine1}),
        ],
        shared_parameters={
            'mux': ['mux', 'mux'],
            'muy': ['muy', 'muy'],
        },
        shared_variables={
            'x': ['x', 'x'],
            'y': ['y', 'y'],
        },
        cuda=cuda,
    )

    for key in function.parameters:
        if key not in ('mux', 'muy'):
            setattr(function, f'{key}_fit', False)

    return function, data0, data1, x, y, mux, muy


def test_joint_fit_gaussian_spline_cpu():
    function, data0, data1, x, y, mux, muy = _make_joint_problem(np, cuda=False)

    fit = LM(function, LSE(), max_iterations=25)
    fit(
        [data0, data1],
        [
            {'x': x, 'y': y},
            {'x': x, 'y': y},
        ],
    )

    assert np.allclose(function.mux, mux, atol=1e-3)
    assert np.allclose(function.muy, muy, atol=1e-3)
    assert np.all(fit.converged > 0)


def test_joint_fit_gaussian_spline_gpu():
    try:
        import cupy as cp
        from numba import cuda
    except Exception:
        return

    if not cuda.is_available():
        return

    function, data0, data1, x, y, mux, muy = _make_joint_problem(cp, cuda=True)

    fit = LM(function, LSE(), max_iterations=25)
    fit(
        [data0, data1],
        [
            {'x': x, 'y': y},
            {'x': x, 'y': y},
        ],
    )

    assert np.allclose(cp.asnumpy(function.mux), cp.asnumpy(mux), atol=1e-3)
    assert np.allclose(cp.asnumpy(function.muy), cp.asnumpy(muy), atol=1e-3)
    assert np.all(cp.asnumpy(fit.converged) > 0)



# %% Test function run
if __name__ == "__main__":
    from corelp import test
    test(__file__)