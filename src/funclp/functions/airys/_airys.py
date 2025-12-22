#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# Date          : 2026-01-01
# Author        : Lancelot PINCET
# GitHub        : https://github.com/LancelotPincet
# Library       : funcLP



# %% Libraries
import numpy as np
import numba as nb
import math




# %% tools

airy_min = 0.1322794682184133

def get_amp(y) :
    return np.nanmax(y) - np.nanmin(y)

def get_offset(y) :
    return (np.nanmin(y)+np.nanmax(y)*airy_min)/(1+airy_min)

def get_mean(y, x) :
    y0 = y - np.nanmin(y)
    num = np.nansum(x * y0)
    denom = np.nansum(y0)
    return num / denom

#Function tools
@nb.njit(nogil = True)
def j1(z):
    if z <= 8.:
        t = z / 8.
        fz = z * (-0.2666949632 * t**14 + 1.7629415168000002 * t**12 + -5.6392305344 * t**10 + 11.1861160576 * t**8 + -14.1749644604 * t**6 + 10.6608917307 * t**4 + -3.9997296130249995 * t**2 + 0.49999791505)
    else:
        t = 8. / z
        eta = z - 0.75 * np.pi
        fz = math.sqrt(2 / (np.pi * z)) * ((1.9776e-06 * t**6 + -3.48648e-05 * t**4 + 0.0018309904000000001 * t**2 + 1.00000000195) * math.cos(eta) - (-6.688e-07 * t**7 + 8.313600000000001e-06 * t**5 + -0.000200241 * t**3 + 0.04687499895 * t) * math.sin(eta))
    return np.float32(fz)


