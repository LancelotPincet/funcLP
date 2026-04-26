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
'JointFunction': 'funclp.modules.JointFunction_LP.JointFunction',
'make_calculation': 'funclp.modules.make_calculation_LP.make_calculation',
'plot': 'funclp.modules.plot_LP.plot',
'ufunc': 'funclp.modules.ufunc_LP.ufunc',
'Binomial': 'funclp.modules.Distribution_LP._functions.distributions.Binomial',
'Gamma': 'funclp.modules.Distribution_LP._functions.distributions.Gamma',
'Poisson': 'funclp.modules.Distribution_LP._functions.distributions.Poisson',
'Normal': 'funclp.modules.Distribution_LP._functions.distributions.Normal',
'LSE': 'funclp.modules.Estimator_LP._functions.estimators.LSE',
'MLE': 'funclp.modules.Estimator_LP._functions.estimators.MLE',
'LM': 'funclp.modules.Fit_LP._functions.optimizers.LM',
'Diamond': 'funclp.modules.Function_LP._functions.masks.Diamond',
'Rectangle': 'funclp.modules.Function_LP._functions.masks.Rectangle',
'Disc': 'funclp.modules.Function_LP._functions.masks.Disc',
'IsoGaussian': 'funclp.modules.Function_LP._functions.gaussians.IsoGaussian',
'Gaussian3D': 'funclp.modules.Function_LP._functions.gaussians.Gaussian3D',
'GaussianBeam': 'funclp.modules.Function_LP._functions.gaussians.GaussianBeam',
'Gaussian': 'funclp.modules.Function_LP._functions.gaussians.Gaussian',
'Gaussian2D': 'funclp.modules.Function_LP._functions.gaussians.Gaussian2D',
'Airy2D': 'funclp.modules.Function_LP._functions.airys.Airy2D',
'Airy': 'funclp.modules.Function_LP._functions.airys.Airy',
'Spline': 'funclp.modules.Function_LP._functions.splines.Spline',
'Spline2D': 'funclp.modules.Function_LP._functions.splines.Spline2D',
'Spline3D': 'funclp.modules.Function_LP._functions.splines.Spline3D',
'Polynomial4': 'funclp.modules.Function_LP._functions.polynomials.Polynomial4',
'Polynomial2': 'funclp.modules.Function_LP._functions.polynomials.Polynomial2',
'Polynomial3': 'funclp.modules.Function_LP._functions.polynomials.Polynomial3',
'Polynomial5': 'funclp.modules.Function_LP._functions.polynomials.Polynomial5',
'Polynomial1': 'funclp.modules.Function_LP._functions.polynomials.Polynomial1',
'Exponential1': 'funclp.modules.Function_LP._functions.exponentials.Exponential1',
'Exponential2': 'funclp.modules.Function_LP._functions.exponentials.Exponential2',
'Exponential3': 'funclp.modules.Function_LP._functions.exponentials.Exponential3',
'JointChannel': 'funclp.modules.JointFunction_LP._functions.JointChannel',
'use_broadcasting': 'funclp.modules.make_calculation_LP._functions.use_broadcasting',
'use_cuda': 'funclp.modules.make_calculation_LP._functions.use_cuda',
'use_shapes': 'funclp.modules.make_calculation_LP._functions.use_shapes',
'use_inputs': 'funclp.modules.make_calculation_LP._functions.use_inputs',
'Parameter': 'funclp.modules.ufunc_LP._functions.Parameter'
}



# %% Lazy imports
from corelp import getmodule
__getattr__, __all__ = getmodule(sources)