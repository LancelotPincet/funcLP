#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# Date          : 2026-01-01
# Author        : Lancelot PINCET
# GitHub        : https://github.com/LancelotPincet
# Library       : funcLP



# %% Libraries
from funclp import Distribution, ufunc
import numpy as np
import math


# %% Function

class Gamma(Distribution) :

    @ufunc(main=True, data=['raw_data', 'model_data'])
    def pdf(self, raw_data, model_data, /, k=np.float32(1.), eps=np.float32(1e-6), weights=np.float32(1.)):
        """Probability Density Function."""
        if raw_data < 0 :
            return 0
        x = raw_data
        theta = eps if model_data < eps else model_data
        return (x ** (k - 1) * math.exp(-x / theta) / (math.gamma(k) * theta ** k)) * weights

    @ufunc(data=['raw_data', 'model_data'])
    def loglikelihood_reduced(self, raw_data, model_data, /, k=np.float32(1.), eps=np.float32(1e-6), weights=np.float32(1.)):
        """Log-likelihood up to additive constants."""
        x = 0 if raw_data < 0 else raw_data
        theta = eps if model_data < eps else model_data
        return (- x / theta - k * math.log(theta)) * weights

    @ufunc(data=['raw_data', 'model_data'])
    def loglikelihood(self, raw_data, model_data, /, k=np.float32(1.), eps=np.float32(1e-6), weights=np.float32(1.)):
        """Exact log-likelihood (with constants)."""
        x = 0 if raw_data < 0 else raw_data
        theta = eps if model_data < eps else model_data
        return ((k - 1) * math.log(x) - x / theta - math.lgamma(k) - k * math.log(theta)) * weights

    @ufunc(data=['raw_data', 'model_data'])
    def dloglikelihood(self, raw_data, model_data, /, k=np.float32(1.), eps=np.float32(1e-6), weights=np.float32(1.)):
        """Derivative of log-likelihood w.r.t model parameter."""
        x = 0 if raw_data < 0 else raw_data
        theta = eps if model_data < eps else model_data
        return (x / (theta ** 2) - k / theta) * weights

    @ufunc(data=['raw_data', 'model_data'])
    def d2loglikelihood(self, raw_data, model_data, /, k=np.float32(1.), eps=np.float32(1e-6), weights=np.float32(1.)):
        """Second derivative of log-likelihood (observed curvature)."""
        x = 0 if raw_data < 0 else raw_data
        theta = eps if model_data < eps else model_data
        return (-2.0 * x / (theta ** 3) + k / (theta ** 2)) * weights

    @ufunc(data=['raw_data', 'model_data'])
    def fisher(self, raw_data, model_data, /, k=np.float32(1.), eps=np.float32(1e-6), weights=np.float32(1.)):
        """Expected curvature (Fisher information)."""
        theta = eps if model_data < eps else model_data
        return (k / (theta ** 2)) * weights

