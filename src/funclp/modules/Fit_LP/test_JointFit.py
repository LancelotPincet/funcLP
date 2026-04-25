#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# Date          : 2026-04-25
# Author        : Lancelot PINCET
# GitHub        : https://github.com/LancelotPincet
# Library       : funcLP



# %% Libraries
from funclp import Gaussian2D, JointChannel, JointFunction, LM, LSE, Spline2D
import numpy as np



# %% Tests
def test_joint_fit_gaussian_spline():
    v = np.linspace(-3, 3, 15, dtype=np.float32)
    x, y = np.meshgrid(v, v)
    mux = np.array([0.25, -0.3], dtype=np.float32)
    muy = np.array([-0.15, 0.2], dtype=np.float32)
    dx, dy = -0.5, 0.35
    sig = np.float32(0.9)
    amp = np.float32(8.)
    offset = np.float32(0.5)

    base = Gaussian2D(sigx=sig, sigy=sig, amp=1., offset=0.)(x, y)
    data0 = Gaussian2D(mux=mux, muy=muy, sigx=sig, sigy=sig, amp=amp, offset=offset)(x, y)
    data1 = Spline2D(base, x, y, mux=mux + dx, muy=muy + dy, amp=amp, offset=offset)(x, y)

    function0 = Gaussian2D(mux=np.zeros(2, dtype=np.float32), muy=np.zeros(2, dtype=np.float32), sigx=sig, sigy=sig, amp=amp, offset=offset)
    function1 = Spline2D(base, x, y, mux=np.zeros(2, dtype=np.float32), muy=np.zeros(2, dtype=np.float32), amp=amp, offset=offset)
    function = JointFunction([JointChannel(function0), JointChannel(function1, offset=(dx, dy))], cuda=False)
    for key in function.parameters:
        if key not in ('mux', 'muy'):
            setattr(function, f'{key}_fit', False)

    fit = LM(function, LSE(), max_iterations=25)
    fit([data0, data1], [(x, y), (x, y)])

    assert np.allclose(function.mux, mux, atol=1e-3)
    assert np.allclose(function.muy, muy, atol=1e-3)
    assert np.all(fit.converged > 0)



# %% Test function run
if __name__ == "__main__":
    from corelp import test
    test(__file__)
