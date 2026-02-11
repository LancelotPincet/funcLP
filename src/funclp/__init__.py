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
'Normal': 'funclp.modules.Distribution_LP._functions.Normal',
'Poisson': 'funclp.modules.Distribution_LP._functions.Poisson',
'Gamma': 'funclp.modules.Distribution_LP._functions.Gamma',
'Binomial': 'funclp.modules.Distribution_LP._functions.Binomial',
'LSE': 'funclp.modules.Estimator_LP._functions.LSE',
'MLE': 'funclp.modules.Estimator_LP._functions.MLE',
'LM': 'funclp.modules.Fit_LP._functions.LM',
'Airy': 'funclp.modules.Function_LP._functions.airys.Airy',
'Airy2D': 'funclp.modules.Function_LP._functions.airys.Airy2D',
'GaussianBeam': 'funclp.modules.Function_LP._functions.gaussians.GaussianBeam',
'Gaussian2D': 'funclp.modules.Function_LP._functions.gaussians.Gaussian2D',
'IsoGaussian': 'funclp.modules.Function_LP._functions.gaussians.IsoGaussian',
'Gaussian': 'funclp.modules.Function_LP._functions.gaussians.Gaussian',
'Polynomial1': 'funclp.modules.Function_LP._functions.polynomials.Polynomial1',
'Polynomial4': 'funclp.modules.Function_LP._functions.polynomials.Polynomial4',
'Polynomial2': 'funclp.modules.Function_LP._functions.polynomials.Polynomial2',
'Polynomial3': 'funclp.modules.Function_LP._functions.polynomials.Polynomial3',
'Polynomial5': 'funclp.modules.Function_LP._functions.polynomials.Polynomial5',
'Disc': 'funclp.modules.Function_LP._functions.masks.Disc',
'Rectangle': 'funclp.modules.Function_LP._functions.masks.Rectangle',
'Diamond': 'funclp.modules.Function_LP._functions.masks.Diamond',
'Exponential3': 'funclp.modules.Function_LP._functions.exponentials.Exponential3',
'Exponential2': 'funclp.modules.Function_LP._functions.exponentials.Exponential2',
'Exponential1': 'funclp.modules.Function_LP._functions.exponentials.Exponential1'
}



# %% Lazy imports
from corelp import getmodule
__getattr__, __all__ = getmodule(sources)