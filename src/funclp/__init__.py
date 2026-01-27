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
'distributions': 'funclp.modules.distributions_LP.distributions',
'estimators': 'funclp.modules.estimators_LP.estimators',
'fits': 'funclp.modules.fits_LP.fits',
'functions': 'funclp.modules.functions_LP.functions',
'make_calculation': 'funclp.modules.make_calculation_LP.make_calculation',
'plot': 'funclp.modules.plot_LP.plot',
'ufunc': 'funclp.modules.ufunc_LP.ufunc',
'use_broadcasting': 'funclp.modules.use_broadcasting_LP.use_broadcasting',
'use_cuda': 'funclp.modules.use_cuda_LP.use_cuda',
'use_inputs': 'funclp.modules.use_inputs_LP.use_inputs',
'use_shapes': 'funclp.modules.use_shapes_LP.use_shapes',
'Normal': 'funclp.modules.distributions_LP._functions.Normal',
'Poisson': 'funclp.modules.distributions_LP._functions.Poisson',
'Gamma': 'funclp.modules.distributions_LP._functions.Gamma',
'Binomial': 'funclp.modules.distributions_LP._functions.Binomial',
'LSE': 'funclp.modules.estimators_LP._functions.LSE',
'MLE': 'funclp.modules.estimators_LP._functions.MLE',
'LM': 'funclp.modules.fits_LP._functions.LM',
'Airy': 'funclp.modules.functions_LP._functions.Airy',
'Airy2D': 'funclp.modules.functions_LP._functions.Airy2D',
'GaussianBeam': 'funclp.modules.functions_LP._functions.GaussianBeam',
'Gaussian2D': 'funclp.modules.functions_LP._functions.Gaussian2D',
'IsoGaussian': 'funclp.modules.functions_LP._functions.IsoGaussian',
'Gaussian': 'funclp.modules.functions_LP._functions.Gaussian',
'Polynomial1': 'funclp.modules.functions_LP._functions.Polynomial1',
'Polynomial4': 'funclp.modules.functions_LP._functions.Polynomial4',
'Polynomial2': 'funclp.modules.functions_LP._functions.Polynomial2',
'Polynomial3': 'funclp.modules.functions_LP._functions.Polynomial3',
'Polynomial5': 'funclp.modules.functions_LP._functions.Polynomial5',
'Disc': 'funclp.modules.functions_LP._functions.Disc',
'Rectangle': 'funclp.modules.functions_LP._functions.Rectangle',
'Diamond': 'funclp.modules.functions_LP._functions.Diamond',
'Exponential3': 'funclp.modules.functions_LP._functions.Exponential3',
'Exponential2': 'funclp.modules.functions_LP._functions.Exponential2',
'Exponential1': 'funclp.modules.functions_LP._functions.Exponential1'
}



# %% Lazy imports
from corelp import getmodule
__getattr__, __all__ = getmodule(sources)