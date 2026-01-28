#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# Date          : 2026-01-01
# Author        : Lancelot PINCET
# GitHub        : https://github.com/LancelotPincet
# Library       : funcLP



# %% Libraries
from funclp import Fit
from corelp import prop, rfrom
import numpy as np
import numba as nb
from numba import cuda
chi2 = rfrom('_chi2', 'chi2')
gradient = rfrom('_gradient', 'gradient')
hessian = rfrom('_hessian', 'hessian')
jacobian = rfrom('_jacobian', 'jacobian')



# %% Function

class LM(Fit) :

    # Attributs
    ftol = 1e-6 # Loop stops when chi2 change is lower than ftol.
    xtol = 0 # Loop stops when parameter step is lower than xtol.
    gtol = 0 # Loop stops when gradient maximum is lower than gtol.
    damping_init = 1e-2 #lm damping
    damping_increase = 7.0 # Factor to apply when damping
    damping_decrease = 0.4 # Factor to apply when damping
    damping_min, damping_max = 1e-6, 1e6 #damping limit values
    max_retries = 10 # Maximum of retries for a given hessian

    # Functions [see attached files]
    chi2 = chi2
    gradient = gradient
    hessian = hessian
    jacobian = jacobian



    def fit(self) :

        # Allocating data

        self.jacobian_data = self.xp.empty(shape=(self.nmodels, self.npoints, self.nparameters2fit), dtype=self.dtype)

        self.chi2_data = self.xp.empty(shape=self.nmodels, dtype=self.dtype)
        self.deviance_data = self.xp.empty(shape=(self.nmodels, self.npoints), dtype=self.dtype)
        self.chi2_cache = self.xp.empty(shape=self.nmodels, dtype=self.dtype)

        self.gradient_data = self.xp.empty(shape=(self.nmodels, self.nparameters2fit, 1), dtype=self.dtype)
        self.loss_data = self.xp.empty(shape=(self.nmodels, self.npoints), dtype=self.dtype)

        self.hessian_data = self.xp.empty(shape=(self.nmodels, self.nparameters2fit, self.nparameters2fit), dtype=self.dtype)
        self.hessian_damped = self.xp.empty(shape=(self.nmodels, self.nparameters2fit, self.nparameters2fit), dtype=self.dtype)
        self.fisher_data = self.xp.empty(shape=(self.nmodels, self.npoints), dtype=self.dtype)

        self.improved = self.xp.zeros_like(self.converge)
        
        self.damping_data = self.xp.full(shape=self.nmodels, dtype=self.dtype, fill_value=self.damping_init)

        # Iterations
        for _ in range(self.max_iterations) :

            # Calculate current model data and jacobian
            self.model()
            self.jacobian()

            # Evaluate solving arrays
            self.chi2()
            self.gradient()
            self.hessian()
            
            # Looping on various damping tries
            self.improved[:] = self.converged
            for _ in range(self.max_retries) :

                self.improved