#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# Date          : 2025-12-20
# Author        : Lancelot PINCET
# GitHub        : https://github.com/LancelotPincet
# Library       : funcLP
# Module        : Fit

"""
This file allows to test Fit

Fit : Class defining fitting algorithms.
"""



# %% Libraries
from corelp import debug
import pytest
from funclp import LM, IsoGaussian, Spline2D, MLE, Poisson
import numpy as np
from time import perf_counter
try :
    import cupy as cp
except ImportError :
    cp = None
from scipy.optimize import curve_fit
debug_folder = debug(__file__)



# %% Function test
def test_function() :
    '''
    Test Fit function
    '''
    
    # Making coordinates
    v = np.linspace(-500, 500, 11)
    X, Y = np.meshgrid(v, v)
    npoints = 100000
    do_spline = True
    
    # Making experimental data
    sigma = 0.21*670/1.5 #nm
    mux = np.random.uniform(-50, 50, npoints)
    muy = np.random.uniform(-50, 50, npoints)
    N = 2000 #photons
    groundtruth_function = IsoGaussian(sig=sigma, mux=mux, muy=muy, pix=100, integ=N)
    data = groundtruth_function(X, Y)
    data = np.random.poisson(data)

    # Spline interpolation
    if do_spline :
        model_function = IsoGaussian(sig=sigma, mux=0, muy=0, pix=100, integ=N)
        model_data = model_function(X, Y)
        spline = Spline2D(model_data, X, Y)
        tx, ty, coeffs = spline.tx, spline.ty, spline.coeffs

    # Making curve_fit fit as reference
    if npoints <= 1000 :
        cf_mux = np.zeros(npoints)
        cf_muy = np.zeros(npoints)
        cf_sig = np.zeros(npoints)
        f = IsoGaussian(cuda=False, mux=mux, muy=muy, sig=0.21*670/1.5, pix=100, integ=N)
        def model(XY, mux, muy):
            return f(XY[0], XY[1], mux=mux, muy=muy).ravel().astype(float)
        tic = perf_counter()
        for i in range(npoints):
            try:
                p0 = [0., 0.]
                popt, _ = curve_fit(model, (X, Y), data[i].ravel().astype(float), p0=p0, method='lm')
                cf_mux[i], cf_muy[i] = popt
            except RuntimeError:
                cf_mux[i], cf_muy[i] = np.nan, np.nan
        toc = perf_counter()
        cf_error_mux = cf_mux - groundtruth_function.mux
        cf_error_muy = cf_muy - groundtruth_function.muy
        print(f'\ncurve_fit took {toc-tic:.3f}s')
        print(f'curve_fit converged: {npoints - np.sum(np.isnan(cf_sig))}/{npoints}')
        print(f'curve_fit Mux error: {np.nanmean(cf_error_mux):.3f} +/- {np.nanstd(cf_error_mux):.3f}')
        print(f'curve_fit Muy error: {np.nanmean(cf_error_muy):.3f} +/- {np.nanstd(cf_error_muy):.3f}')
    


    # Making Fit
    if do_spline :
        function = Spline2D(cuda=False, tx=tx, ty=ty, coeffs=coeffs, amp=np.ones(npoints))
    else :
        function = IsoGaussian(cuda=False, sig=np.full(npoints, 0.21*670/1.5), pix=100, integ=N)
    function.amp_fit = False
    function.offset_fit = False
    estimator = MLE(Poisson())
    fit = LM(function, estimator)
    tic = perf_counter()
    fit(data, X, Y)
    toc = perf_counter()
    error_mux = function.mux - groundtruth_function.mux
    error_muy = function.muy - groundtruth_function.muy
    toc = perf_counter()
    print(f'\nCPU took {toc-tic:.3f}s')
    print(f'CPU converged: {np.sum(fit.converged > 0)}/{npoints}')
    print(f'CPU Mux error: {np.mean(error_mux):.3f} +/- {np.std(error_mux):.3f}')
    print(f'CPU Muy error: {np.mean(error_muy):.3f} +/- {np.std(error_muy):.3f}')



    # Making Fit (GPU)
    if cp is not None :
        data, X, Y = cp.asarray(data), cp.asarray(X), cp.asarray(Y)
        if do_spline :
            function = Spline2D(cuda=True, tx=tx, ty=ty, coeffs=coeffs, amp=np.ones(npoints))
        else :
            function = IsoGaussian(cuda=True, sig=np.full(npoints, 0.21*670/1.5), pix=100, integ=N)
        function.amp_fit = False
        function.offset_fit = False
        estimator = MLE(Poisson())
        fit = LM(function, estimator)
        fit(data, X, Y)
        cp.cuda.Stream.null.synchronize()
        function.mux = 0
        function.muy = 0
        tic = perf_counter()
        fit(data, X, Y)
        toc = perf_counter()
        error_mux = cp.asnumpy(function.mux) - groundtruth_function.mux
        error_muy = cp.asnumpy(function.muy) - groundtruth_function.muy
        print(f'\nGPU took {toc-tic:.3f}s')
        print(f'GPU converged: {np.sum(fit.converged > 0)}/{npoints}')
        print(f'GPU Mux error: {np.mean(error_mux):.3f} +/- {np.std(error_mux):.3f}')
        print(f'GPU Muy error: {np.mean(error_muy):.3f} +/- {np.std(error_muy):.3f}')





# %% Test function run
if __name__ == "__main__":
    from corelp import test
    test(__file__)