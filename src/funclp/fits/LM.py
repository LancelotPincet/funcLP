#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# Date          : 2026-01-01
# Author        : Lancelot PINCET
# GitHub        : https://github.com/LancelotPincet
# Library       : funcLP



# %% Libraries
from funclp import Fit



# %% Function

class LM(Fit) :

    # Attributs
    ftol = 1e-6 # Loop stops when chi2 change is lower than ftol.
    xtol = 0 # Loop stops when parameter step is lower than xtol.
    gtol = 0 # Loop stops when gradient maximum is lower than gtol.
    damping = 1e-2 #lm damping
    damping_increase = 7.0 # Factor to apply when damping
    damping_decrease = 0.4 # Factor to apply when damping
    damping_min, damping_max = 1e-6, 1e6 #damping limit values
    eps = 1e-4 #infinetesimal number for jacobian calculations

    def fit(self) :

        # Allocating data in memory
        mode_data = None
        old_chi2, new_chi2, deviance = None, None, None

        # Iterations
        for _ in range(self.max_iterations) :

            #Estimate old chi2
            model_data = self.function(*self.variables, *self.data, **self.parameters, out=model_data, ignore=self.converged)
            model_function(*variables.values(), *parameters.values(), model, converged)
            deviance_function(data, model, weights, deviance, converged)
            chi2_function(deviance, old_chi2, converged)



    def chi2(self, out=None) : #ICI
        self.deviance = self.estimator.deviance(raw_data, model_data, weights=weights, out=self.deviance)
        return deviance.sum(axis=1, out=out)


