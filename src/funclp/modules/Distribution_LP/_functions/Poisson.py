#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# Date          : 2026-01-01
# Author        : Lancelot PINCET
# GitHub        : https://github.com/LancelotPincet
# Library       : funcLP



# %% Libraries
from funclp import Distribution, Parameter, ufunc
import numpy as np
import math



PARAMETERS = [Parameter('eps', np.float32(1e-6))]


# %% Function

class Poisson(Distribution) :

    @property
    def default_attributes(self):
        return 'nb.float32(1e-6), ' # eps

    @ufunc(variables=[], data=['raw_data', 'model_data', 'weights'], parameters=PARAMETERS)
    def pdf(raw_data, model_data, weights=np.float32(1.), /, eps=np.float32(1e-6)):
        """Probability Density Function."""
        lam = eps if model_data < eps else model_data
        k = raw_data
        if k < 0 : return 0
        log_pdf = -lam + k*math.log(lam) - math.lgamma(k+1)
        return (math.exp(log_pdf) * lam ** k / math.lgamma(k + 1)) * weights

    @ufunc(variables=[], data=['raw_data', 'model_data', 'weights'], parameters=PARAMETERS)
    def loglikelihood_reduced(raw_data, model_data, weights=np.float32(1.), /, eps=np.float32(1e-6)):
        """Log-likelihood up to additive constants."""
        lam = eps if model_data < eps else model_data
        k = raw_data if raw_data > 0 else 0
        return (k * math.log(lam) - lam) * weights

    @ufunc(variables=[], data=['raw_data', 'model_data', 'weights'], parameters=PARAMETERS)
    def loglikelihood(raw_data, model_data, weights=np.float32(1.), /, eps=np.float32(1e-6)):
        """Exact log-likelihood (with constants)."""
        lam = eps if model_data < eps else model_data
        k = raw_data if raw_data > 0 else 0
        return (k * math.log(lam) - lam - math.lgamma(k + 1)) * weights

    @ufunc(variables=[], data=['raw_data', 'model_data', 'weights'], parameters=PARAMETERS)
    def dloglikelihood(raw_data, model_data, weights=np.float32(1.), /, eps=np.float32(1e-6)):
        """Derivative of log-likelihood w.r.t model parameter."""
        lam = eps if model_data < eps else model_data
        k = raw_data if raw_data > 0 else 0
        return ((k - lam) / lam) * weights

    @ufunc(variables=[], data=['raw_data', 'model_data', 'weights'], parameters=PARAMETERS)
    def d2loglikelihood(raw_data, model_data, weights=np.float32(1.), /, eps=np.float32(1e-6)):
        """Second derivative of log-likelihood (observed curvature)."""
        lam = eps if model_data < eps else model_data
        k = raw_data if raw_data > 0 else 0
        return (-k / (lam ** 2)) * weights

    @ufunc(variables=[], data=['raw_data', 'model_data', 'weights'], parameters=PARAMETERS)
    def fisher(raw_data, model_data, weights=np.float32(1.), /, eps=np.float32(1e-6)):
        """Expected curvature (Fisher information)."""
        lam = eps if model_data < eps else model_data
        return (1.0 / lam) * weights

