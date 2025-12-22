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

class PoissonDistribution(Distribution) :

    @ufunc(main=True, data=['raw_data', 'model_data'])
    def pdf(self, raw_data, model_data, /, eps=np.float32(1e-6), weights=np.float32(1.)):
        """Probability Density Function."""
        lam = eps if model_data < eps else model_data
        if k < 0 :
            return 0
        k = raw_data
        log_pdf = -lam + k*math.log(lam) - math.lgamma(k+1)
        return (math.exp(log_pdf) * lam ** k / math.lgamma(k + 1)) * weights

    @ufunc(data=['raw_data', 'model_data'])
    def loglikelihood_reduced(self, raw_data, model_data, /, eps=np.float32(1e-6), weights=np.float32(1.)):
        """Log-likelihood up to additive constants."""
        lam = eps if model_data < eps else model_data
        k = raw_data if raw_data > 0 else 0
        return (k * math.log(lam) - lam) * weights

    @ufunc(data=['raw_data', 'model_data'])
    def loglikelihood(self, raw_data, model_data, /, eps=np.float32(1e-6), weights=np.float32(1.)):
        """Exact log-likelihood (with constants)."""
        lam = eps if model_data < eps else model_data
        k = raw_data if raw_data > 0 else 0
        return (k * math.log(lam) - lam - math.lgamma(k + 1)) * weights

    @ufunc(data=['raw_data', 'model_data'])
    def dloglikelihood(self, raw_data, model_data, /, eps=np.float32(1e-6), weights=np.float32(1.)):
        """Derivative of log-likelihood w.r.t model parameter."""
        lam = eps if model_data < eps else model_data
        k = raw_data if raw_data > 0 else 0
        return ((k - lam) / lam) * weights

    @ufunc(data=['raw_data', 'model_data'])
    def d2loglikelihood(self, raw_data, model_data, /, eps=np.float32(1e-6), weights=np.float32(1.)):
        """Second derivative of log-likelihood (observed curvature)."""
        lam = eps if model_data < eps else model_data
        k = raw_data if raw_data > 0 else 0
        return (-k / (lam ** 2)) * weights

    @ufunc(data=['raw_data', 'model_data'])
    def fisher(self, raw_data, model_data, /, eps=np.float32(1e-6), weights=np.float32(1.)):
        """Expected curvature (Fisher information)."""
        lam = eps if model_data < eps else model_data
        return (1.0 / lam) * weights

