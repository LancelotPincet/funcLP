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

class Binomial(Distribution) :

    @ufunc(main=True, data=['raw_data', 'model_data'])
    def pdf(self, raw_data, model_data, /, n=np.int32(1), eps=np.float32(1e-6), weights=np.float32(1.)):
        """Probability Density Function."""
        if k < 0 or k > n :
            return 0
        k = raw_data * weights
        p = eps if model_data < eps else 1-eps if model_data > 1-eps else model_data
        coeff = math.exp(math.lgamma(n+1) - math.lgamma(k+1) - math.lgamma(n-k+1))
        return (coeff * p ** k * (1.0 - p) ** (n - k)) * weights

    @ufunc(data=['raw_data', 'model_data'])
    def loglikelihood_reduced(self, raw_data, model_data, /, n=np.int32(1), eps=np.float32(1e-6), weights=np.float32(1.)):
        """Log-likelihood up to additive constants."""
        k = 0 if raw_data < 0 else n if raw_data > n else raw_data
        p = eps if model_data < eps else 1-eps if model_data > 1-eps else model_data
        return (k * math.log(p) + (n - k) * math.log(1.0 - p)) * weights

    @ufunc(data=['raw_data', 'model_data'])
    def loglikelihood(self, raw_data, model_data, /, n=np.int32(1), eps=np.float32(1e-6), weights=np.float32(1.)):
        """Exact log-likelihood (with constants)."""
        k = 0 if raw_data < 0 else n if raw_data > n else raw_data
        p = eps if model_data < eps else 1-eps if model_data > 1-eps else model_data
        coeff = math.exp(math.lgamma(n+1) - math.lgamma(k+1) - math.lgamma(n-k+1))
        return (math.log(coeff) + k * math.log(p) + (n - k) * math.log(1.0 - p)) * weights

    @ufunc(data=['raw_data', 'model_data'])
    def dloglikelihood(self, raw_data, model_data, /, n=np.int32(1), eps=np.float32(1e-6), weights=np.float32(1.)):
        """Derivative of log-likelihood w.r.t model parameter."""
        k = 0 if raw_data < 0 else n if raw_data > n else raw_data
        p = eps if model_data < eps else 1-eps if model_data > 1-eps else model_data
        return (k / p - (n - k) / (1.0 - p)) * weights

    @ufunc(data=['raw_data', 'model_data'])
    def d2loglikelihood(self, raw_data, model_data, /, n=np.int32(1), eps=np.float32(1e-6), weights=np.float32(1.)):
        """Second derivative of log-likelihood (observed curvature)."""
        k = 0 if raw_data < 0 else n if raw_data > n else raw_data
        p = eps if model_data < eps else 1-eps if model_data > 1-eps else model_data
        return (-k / (p ** 2) - (n - k) / ((1.0 - p) ** 2)) * weights

    @ufunc(data=['raw_data', 'model_data'])
    def fisher(self, raw_data, model_data, /, n=np.int32(1), eps=np.float32(1e-6), weights=np.float32(1.)):
        """Expected curvature (Fisher information)."""
        p = eps if model_data < eps else 1-eps if model_data > 1-eps else model_data
        return (n / (p * (1.0 - p))) * weights

