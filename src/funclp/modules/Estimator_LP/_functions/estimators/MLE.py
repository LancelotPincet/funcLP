#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# Date          : 2026-01-01
# Author        : Lancelot PINCET
# GitHub        : https://github.com/LancelotPincet
# Library       : funcLP



# %% Libraries
from funclp import Estimator, kernel_caching
from corelp import prop



# %% Function

class MLE(Estimator) :

    def _cache_suffix(self):
        """Stable suffix for cached estimator kernels."""
        return self.__class__.__name__, self.distribution.__class__.__name__

    def _kernel_source(self, mode, kind):
        """Build source code for one estimator helper kernel."""
        estimator_name, distribution_name = self._cache_suffix()
        default = self.distribution.default_attributes
        kernel_map = {
            "deviance": "loglikelihood",
            "loss": "dloglikelihood",
            "observed": "d2loglikelihood",
            "fisher": "fisher",
        }
        weight_factor = {
            "deviance": "(-2) * weights",
            "loss": "(-1) * weights",
            "observed": "(-1) * weights",
            "fisher": None,
        }
        dist_method = kernel_map[kind]
        if mode == "cpu":
            import_name = f"_{distribution_name}_cpukernel_{dist_method}"
            kernel_name = f"_{estimator_name}_{distribution_name}_cpukernel_{kind}"
            decorator = "@nb.njit(nogil=True, fastmath=True, cache=True)"
        else:
            import_name = f"_{distribution_name}_gpukernel_{dist_method}"
            kernel_name = f"_{estimator_name}_{distribution_name}_gpukernel_{kind}"
            decorator = "@nb.cuda.jit(device=True, fastmath=True, cache=True)"

        imports = [
            "import numba as nb",
            "from numba import cuda",
            f"from ._{distribution_name}_{mode}kernel_{dist_method} import {import_name} as kernel",
        ]
        if weight_factor[kind] is None:
            body = f"return kernel(raw_data, model_data, weights, {default})"
        else:
            body = (
                f"weights = {weight_factor[kind]}\n"
                f"    return kernel(raw_data, model_data, weights, {default})"
            )

        code = f'''
{decorator}
def {kernel_name}(raw_data, model_data, weights):
    {body}
'''
        return '\n'.join(imports) + '\n' + code

    def _cached_kernel(self, mode, kind):
        """Cache and return one estimator helper kernel."""
        estimator_name, distribution_name = self._cache_suffix()
        module_name = f"_{estimator_name}_{distribution_name}_{mode}kernel_{kind}"
        source = self._kernel_source(mode, kind)
        return kernel_caching(module_name, source, object_name=module_name)



    # Deviance

    def deviance(self, raw_data, model_data, weights=1, **kwargs) :
        ''' How well the model fits the data '''
        weights = (-2) * weights
        return self.distribution.loglikelihood(raw_data, model_data, weights, **kwargs)

    @prop(cache=True)
    def cpukernel_deviance(self) :
        return self._cached_kernel("cpu", "deviance")

    @prop(cache=True)
    def gpukernel_deviance(self) :
        return self._cached_kernel("gpu", "deviance")



    # Loss

    def loss(self, raw_data, model_data, weights=1, **kwargs) :
        ''' Loss for gradient descent '''
        weights = (-1) * weights
        return self.distribution.dloglikelihood(raw_data, model_data, weights, **kwargs)

    @prop(cache=True)
    def cpukernel_loss(self) :
        return self._cached_kernel("cpu", "loss")

    @prop(cache=True)
    def gpukernel_loss(self) :
        return self._cached_kernel("gpu", "loss")



    # Observed

    def observed(self, raw_data, model_data, weights=1, **kwargs) :
        ''' Observed Hessian (negative second derivative)'''
        weights = (-1) * weights
        return self.distribution.d2loglikelihood(raw_data, model_data, weights, **kwargs)

    @prop(cache=True)
    def cpukernel_observed(self) :
        return self._cached_kernel("cpu", "observed")

    @prop(cache=True)
    def gpukernel_observed(self) :
        return self._cached_kernel("gpu", "observed")



    # Fisher

    def fisher(self, raw_data, model_data, weights=1, **kwargs) :
        ''' Expected Hessian (Fisher information) '''
        return self.distribution.fisher(raw_data, model_data, weights, **kwargs)
    
    @prop(cache=True)
    def cpukernel_fisher(self) :
        return self._cached_kernel("cpu", "fisher")

    @prop(cache=True)
    def gpukernel_fisher(self) :
        return self._cached_kernel("gpu", "fisher")


