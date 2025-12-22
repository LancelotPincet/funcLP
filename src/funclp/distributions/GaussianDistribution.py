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

class GaussianDistribution(Distribution) :

    @ufunc(main=True, data=['raw_data', 'model_data'])
    def pdf(self, raw_data, model_data, /, sigma=np.float32(1), weights=np.float32(1.)):
        """Probability Density Function."""
        mu, sig2 = model_data, sigma**2
        return (math.exp(-0.5 * (raw_data - mu) ** 2 / sig2) / math.sqrt(2.0 * math.pi * sig2)) * weights

    @ufunc(data=['raw_data', 'model_data'])
    def loglikelihood_reduced(self, raw_data, model_data, /, sigma=np.float32(1), weights=np.float32(1.)):
        """Log-likelihood up to additive constants."""
        mu, sig2 = model_data, sigma**2
        return (- 0.5 * (raw_data - mu) ** 2 / sig2) * weights

    @ufunc(data=['raw_data', 'model_data'])
    def loglikelihood(self, raw_data, model_data, /, sigma=np.float32(1), weights=np.float32(1.)):
        """Exact log-likelihood (with constants)."""
        mu, sig2 = model_data, sigma**2
        return (- 0.5 * math.log(2.0 * math.pi * sig2) - 0.5 * (raw_data - mu) ** 2 / sig2) * weights

    @ufunc(data=['raw_data', 'model_data'])
    def dloglikelihood(self, raw_data, model_data, /, sigma=np.float32(1), weights=np.float32(1.)):
        """Derivative of log-likelihood w.r.t model parameter."""
        mu, sig2 = model_data, sigma**2
        return ((raw_data - mu) / sig2) * weights

    @ufunc(data=['raw_data', 'model_data'])
    def d2loglikelihood(self, raw_data, model_data, /, sigma=np.float32(1), weights=np.float32(1.)):
        """Second derivative of log-likelihood (observed curvature)."""
        return (-1.0 / sigma ** 2) * weights

    @ufunc(data=['raw_data', 'model_data'])
    def fisher(self, raw_data, model_data, /, sigma=np.float32(1), weights=np.float32(1.)):
        """Expected curvature (Fisher information)."""
        return (1.0 / sigma ** 2) * weights

