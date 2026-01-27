#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# Date          : 2026-01-01
# Author        : Lancelot PINCET
# GitHub        : https://github.com/LancelotPincet
# Library       : funcLP



# %% Libraries
import numpy as np
import numba as nb



# %% tools

def get_amp(y) :
    return np.nanmax(y) - np.nanmin(y)

def get_offset(y) :
    return np.nanmin(y)

def get_mean(y, x) :
    y0 = y - np.nanmin(y)
    num = np.nansum(x * y0)
    denom = np.nansum(y0)
    return num / denom

@nb.njit
def get_r(y,x) :
    y,x = y.ravel(), x.ravel()
    y = y[np.argsort(x)]
    y -= y.min()
    y /= y.max()
    xmin = 0
    for pos in range(0,len(x),1) :
        if y[pos] > 0.05 :
            xmin = pos
            break
    xmax = len(x) - 1
    for pos in range(len(x)-1,-1,-1) :
        if y[pos] > 0.05 :
            xmax = pos
            break
    if xmax == xmin :
        return x.max() - x.min()
    else :
        return abs(x[xmax] - x[xmin])

