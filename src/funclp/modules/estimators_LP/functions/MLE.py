#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# Date          : 2026-01-01
# Author        : Lancelot PINCET
# GitHub        : https://github.com/LancelotPincet
# Library       : funcLP



# %% Libraries
from funclp import Estimator



# %% Function

class MLE(Estimator) :

    def deviance(self, raw_data, model_data, /, weights=1, **kwargs) :
        ''' How well the model fits the data '''
        weights *= -2
        return self.distribution.loglikelihood(raw_data, model_data, weights=weights, **kwargs)

    def loss(self, raw_data, model_data, /, weights=1, **kwargs) :
        ''' Loss for gradient descent '''
        weights *= -1
        return self.distribution.dloglikelihood(raw_data, model_data, weights=weights, **kwargs)

    def observed(self, raw_data, model_data, /, weights=1, **kwargs) :
        ''' Observed Hessian (negative second derivative)'''
        weights *= -1
        return self.distribution.d2loglikelihood(raw_data, model_data, weights=weights, **kwargs)

    def fisher(self, raw_data, model_data, /, weights=1, **kwargs) :
        ''' Expected Hessian (Fisher information) '''
        return self.distribution.fisher(raw_data, model_data, weights=weights, **kwargs)