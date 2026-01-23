#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# Date          : 2025-08-28
# Author        : Lancelot PINCET
# GitHub        : https://github.com/LancelotPincet
# Library       : funcLP

"""
A library for defining mathematical functions and fits.
"""



# %% Source import
sources = {
'CudaReference': 'funclp.modules.CudaReference_LP.CudaReference',
'Distribution': 'funclp.modules.Distribution_LP.Distribution',
'Estimator': 'funclp.modules.Estimator_LP.Estimator',
'Fit': 'funclp.modules.Fit_LP.Fit',
'Function': 'funclp.modules.Function_LP.Function',
'make_calculation': 'funclp.modules.make_calculation_LP.make_calculation',
'plot': 'funclp.modules.plot_LP.plot',
'ufunc': 'funclp.modules.ufunc_LP.ufunc',
'use_broadcasting': 'funclp.modules.use_broadcasting_LP.use_broadcasting',
'use_cuda': 'funclp.modules.use_cuda_LP.use_cuda',
'use_inputs': 'funclp.modules.use_inputs_LP.use_inputs',
'use_shapes': 'funclp.modules.use_shapes_LP.use_shapes'
}

from importlib import resources
from contextlib import contextmanager

@contextmanager
def resources_dir():
    with resources.as_file(resources.files("funclp.resources")) as path:
        yield path

# %% Hidden imports
if False :
    import funclp.modules.CudaReference_LP.CudaReference
    import funclp.modules.Distribution_LP.Distribution
    import funclp.modules.Estimator_LP.Estimator
    import funclp.modules.Fit_LP.Fit
    import funclp.modules.Function_LP.Function
    import funclp.modules.make_calculation_LP.make_calculation
    import funclp.modules.plot_LP.plot
    import funclp.modules.ufunc_LP.ufunc
    import funclp.modules.use_broadcasting_LP.use_broadcasting
    import funclp.modules.use_cuda_LP.use_cuda
    import funclp.modules.use_inputs_LP.use_inputs
    import funclp.modules.use_shapes_LP.use_shapes



# %% Lazy imports
from corelp import getmodule
__getattr__, __all__ = getmodule(sources)