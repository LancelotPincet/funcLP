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

class Normal(Distribution) :

    @property
    def default_attributes(self):
        return 'sigma=nb.float32(1), '

    @ufunc(data=['raw_data', 'model_data', 'weights'])
    def pdf(raw_data, model_data, weights=np.float32(1.), /, sigma=np.float32(1)):
        """Probability Density Function."""
        mu, sig2 = model_data, sigma**2
        return (math.exp(-0.5 * (raw_data - mu) ** 2 / sig2) / math.sqrt(2.0 * math.pi * sig2)) * weights

    @ufunc(data=['raw_data', 'model_data', 'weights'])
    def loglikelihood_reduced(raw_data, model_data, weights=np.float32(1.), /, sigma=np.float32(1)):
        """Log-likelihood up to additive constants."""
        mu, sig2 = model_data, sigma**2
        return (- 0.5 * (raw_data - mu) ** 2 / sig2) * weights

    @ufunc(data=['raw_data', 'model_data', 'weights'])
    def loglikelihood(raw_data, model_data, weights=np.float32(1.), /, sigma=np.float32(1)):
        """Exact log-likelihood (with constants)."""
        mu, sig2 = model_data, sigma**2
        return (- 0.5 * math.log(2.0 * math.pi * sig2) - 0.5 * (raw_data - mu) ** 2 / sig2) * weights

    @ufunc(data=['raw_data', 'model_data', 'weights'])
    def dloglikelihood(raw_data, model_data, weights=np.float32(1.), /, sigma=np.float32(1)):
        """Derivative of log-likelihood w.r.t model parameter."""
        mu, sig2 = model_data, sigma**2
        return ((raw_data - mu) / sig2) * weights

    @ufunc(data=['raw_data', 'model_data', 'weights'])
    def d2loglikelihood(raw_data, model_data, weights=np.float32(1.), /, sigma=np.float32(1)):
        """Second derivative of log-likelihood (observed curvature)."""
        return (-1.0 / sigma ** 2) * weights

    @ufunc(data=['raw_data', 'model_data', 'weights'])
    def fisher(raw_data, model_data, weights=np.float32(1.), /, sigma=np.float32(1)):
        """Expected curvature (Fisher information)."""
        return (1.0 / sigma ** 2) * weights

