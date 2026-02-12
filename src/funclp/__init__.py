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
'use_shapes': 'funclp.modules.use_shapes_LP.use_shapes',
'Binomial': 'funclp.modules.Distribution_LP._functions.Binomial',
'Gamma': 'funclp.modules.Distribution_LP._functions.Gamma',
'Poisson': 'funclp.modules.Distribution_LP._functions.Poisson',
'Normal': 'funclp.modules.Distribution_LP._functions.Normal',
'LSE': 'funclp.modules.Estimator_LP._functions.LSE',
'MLE': 'funclp.modules.Estimator_LP._functions.MLE',
'LM': 'funclp.modules.Fit_LP._functions.LM',
'Diamond': 'funclp.modules.Function_LP._functions.masks.Diamond',
'Rectangle': 'funclp.modules.Function_LP._functions.masks.Rectangle',
'Disc': 'funclp.modules.Function_LP._functions.masks.Disc',
'IsoGaussian': 'funclp.modules.Function_LP._functions.gaussians.IsoGaussian',
'GaussianBeam': 'funclp.modules.Function_LP._functions.gaussians.GaussianBeam',
'Gaussian': 'funclp.modules.Function_LP._functions.gaussians.Gaussian',
'Gaussian2D': 'funclp.modules.Function_LP._functions.gaussians.Gaussian2D',
'Airy2D': 'funclp.modules.Function_LP._functions.airys.Airy2D',
'Airy': 'funclp.modules.Function_LP._functions.airys.Airy',
'Polynomial4': 'funclp.modules.Function_LP._functions.polynomials.Polynomial4',
'Polynomial2': 'funclp.modules.Function_LP._functions.polynomials.Polynomial2',
'Polynomial3': 'funclp.modules.Function_LP._functions.polynomials.Polynomial3',
'Polynomial5': 'funclp.modules.Function_LP._functions.polynomials.Polynomial5',
'Polynomial1': 'funclp.modules.Function_LP._functions.polynomials.Polynomial1',
'Exponential1': 'funclp.modules.Function_LP._functions.exponentials.Exponential1',
'Exponential2': 'funclp.modules.Function_LP._functions.exponentials.Exponential2',
'Exponential3': 'funclp.modules.Function_LP._functions.exponentials.Exponential3'
}



# %% Lazy imports
from corelp import getmodule
__getattr__, __all__ = getmodule(sources)