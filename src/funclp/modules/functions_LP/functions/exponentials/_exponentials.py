#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# Date          : 2026-01-01
# Author        : Lancelot PINCET
# GitHub        : https://github.com/LancelotPincet
# Library       : funcLP



# %% Libraries
import numpy as np



# %% tools

def get_amp(y) :
    return np.nanmax(y) - np.nanmin(y)

def get_offset(y) :
    return np.nanmin(y)

def get_tau(y, x) :
    return np.nanmean(-y / np.gradient(y, x))

