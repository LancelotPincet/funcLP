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

def get_amp(y) :
    return np.nanmax(y) - np.nanmin(y)

def get_offset(y) :
    return np.nanmin(y)

def get_mean(y, x) :
    y0 = y - np.nanmin(y)
    num = np.nansum(x * y0)
    denom = np.nansum(y0)
    return num / denom

def get_std(y, x):
    if np.size(x) == 1:
        return 1
    pixel = np.mean(np.diff(x))
    y = y - np.nanmin(y)
    A = np.nanmax(y)
    I = np.nansum(y) * pixel
    return I / A / np.sqrt(2 * np.pi)

@nb.njit(nogil = True)
def gausfunc(x, mean=np.float32(0), std=np.float32(1), amp=np.float32(1), offset=np.float32(0), pix=np.float32(-1), num_std=np.float32(-1)) :
    if num_std > 0 and x-mean > num_std * std :
        return 0.
    if pix < 0 :
        return amp * math.exp(-(x - mean)**2 / 2 / std**2) + offset
    else :
        xmin = (x-mean - pix/2) / math.sqrt(2) / std
        xmax = (x-mean + pix/2) / math.sqrt(2) / std
        return amp * math.sqrt(math.pi) / math.sqrt(2) * std / pix * (math.erf(xmax) - math.erf(xmin)) + offset

@nb.njit(nogil = True)
def correct_angle(theta, x, y, mux, muy) :
    if theta == 0 :
        return x, y, mux, muy
    theta = theta / 180 * math.pi
    x, y = (x - mux), (y - muy)
    mux, muy = 0, 0
    x, y = x * math.cos(theta) - y * math.sin(theta), x * math.sin(theta) + y * math.cos(theta)
    return x, y, mux, muy

