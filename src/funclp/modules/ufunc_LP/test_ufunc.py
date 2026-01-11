#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# Date          : 2025-12-13
# Author        : Lancelot PINCET
# GitHub        : https://github.com/LancelotPincet
# Library       : funcLP
# Module        : ufunc

"""
This file allows to test ufunc

ufunc : Decorator class defining universal function factory object from python kernel function, will create kernels, vectorized functions, jitted functions, stack functions, all on CPU / Parallel CPU / GPU.
"""



# %% Libraries
from corelp import debug
from funclp import ufunc
from time import perf_counter
import numpy as np
import warnings
from numba.cuda.dispatcher import NumbaPerformanceWarning
try :
    import cupy as cp
except ImportError :
    cp = None
debug_folder = debug(__file__)



# %% Function test
def test_args() :
    '''
    Test ufunc function
    '''
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", category=NumbaPerformanceWarning)

        class MyClass() :
            @ufunc() # <-- HERE IS THE DECORATOR TO USE
            def nodata(x, y, /, a, b=1) :
                return a * x + b * y
            @ufunc(data=['constant']) # <-- HERE IS THE DECORATOR TO USE
            def novar(constant, /, a, b=1) :
                return a + b + constant
            @ufunc(data=['constant']) # <-- HERE IS THE DECORATOR TO USE
            def nopar(x, y, constant=1, /) :
                return x + y + constant
            cuda = False
        
        # --- Instanciation ---

        X, Y = (10, 20)
        N = 5
        instance = MyClass()
        x = np.arange(X).reshape((1, X))
        y = np.arange(Y).reshape((Y, 1))
        constant = np.ones((N, Y, X))
        a = np.ones(N)
        b = np.ones(N)

        # --- Calculations ---

        cpu_out = instance.nodata(x, y, b=b, a=a) # nodata
        cpu_out = instance.nopar(x, y, constant) # nopar
        cpu_out = instance.novar(constant, a, b=b) # novar



# %% Function test
def test_func() :
    '''
    Test ufunc function
    '''
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", category=NumbaPerformanceWarning)

        class MyClass() :
            @ufunc(data=['constant']) # <-- HERE IS THE DECORATOR TO USE
            def myfunc(x, y, constant, /, a, b=0) :
                return a * x + b * y + constant
            cuda = False
        
        # --- Instanciation ---

        X, Y = (10, 20)
        N = 5
        instance = MyClass()
        x = np.arange(X).reshape((1, X))
        y = np.arange(Y).reshape((Y, 1))
        constant = np.ones((N, Y, X))
        a = np.ones(N)
        b = np.ones(N)

        # --- Calculations ---

        cpu_out = instance.myfunc(x, y, constant, a=a, b=b) # full
        assert cpu_out.shape == (5, 20, 10)
        cpu_out = instance.myfunc(x, y, constant[0, :, :], a=1, b=1) # nomodel
        assert cpu_out.shape == (20, 10)
        cpu_out = instance.myfunc(1, 1, constant[:, 0, 0], a=a, b=b) # nopoint
        assert cpu_out.shape == (5,)
        cpu_out = instance.myfunc(1, 1, constant[:, :, :], a=1, b=1) # data full
        assert cpu_out.shape == (5, 20, 10)
        cpu_out = instance.myfunc(x, y, constant[:, :, :], a=1, b=1) # data model
        assert cpu_out.shape == (5, 20, 10)
        cpu_out = instance.myfunc(1, 1, constant[:, :, :], a=a, b=b) # data point
        assert cpu_out.shape == (5, 20, 10)
        cpu_out = instance.myfunc(x, y, 1, a=a, b=b) # no data full
        assert cpu_out.shape == (5, 20, 10)
        cpu_out = instance.myfunc(1, 1, 1, a=a, b=b) # no data with model
        assert cpu_out.shape == (5,)
        cpu_out = instance.myfunc(x, y, 1, a=1, b=1) # no data with point
        assert cpu_out.shape == (20, 10)
        cpu_out = instance.myfunc(1, 1, constant[0, 0, 0], a=1, b=1) # Scalar
        assert cpu_out.shape == ()



# %% Function test
def test_main() :
    '''
    Test ufunc function
    '''
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", category=NumbaPerformanceWarning)

        class MyClass() :
            @ufunc(main=True, data=['constant']) # <-- HERE IS THE DECORATOR TO USE
            def myfunc(x, y, constant, /, a=1, b=1) :
                return a * x + b * y + constant
            @ufunc()
            def otherfunc(x, /, a=1) :
                return x + a
            cuda = False
        
        # --- Instanciation ---

        X, Y = (10, 20)
        N = 5
        instance = MyClass()
        x = np.arange(X).reshape((1, X))
        y = np.arange(Y).reshape((Y, 1))
        constant = np.ones((N, Y, X))
        a = np.ones(N)
        b = np.ones(N)

        # --- Calculations ---

        cpu_out = instance.myfunc(x, y, constant, a=a, b=b) # full
        assert cpu_out.shape == (5, 20, 10)
        cpu_out = instance.myfunc(x, y, constant[0, :, :], a=1, b=1) # nomodel
        assert cpu_out.shape == (20, 10)
        cpu_out = instance.myfunc(1, 1, constant[:, 0, 0], a=a, b=b) # nopoint
        assert cpu_out.shape == (5,)
        cpu_out = instance.myfunc(1, 1, constant[:, :, :], a=1, b=1) # data full
        assert cpu_out.shape == (5, 20, 10)
        cpu_out = instance.myfunc(x, y, constant[:, :, :], a=1, b=1) # data model
        assert cpu_out.shape == (5, 20, 10)
        cpu_out = instance.myfunc(1, 1, constant[:, :, :], a=a, b=b) # data point
        assert cpu_out.shape == (5, 20, 10)
        cpu_out = instance.myfunc(x, y, 1, a=a, b=b) # no data full
        assert cpu_out.shape == (5, 20, 10)
        cpu_out = instance.myfunc(1, 1, 1, a=a, b=b) # no data with model
        assert cpu_out.shape == (5,)
        cpu_out = instance.myfunc(x, y, 1, a=1, b=1) # no data with point
        assert cpu_out.shape == (20, 10)
        cpu_out = instance.myfunc(1, 1, constant[0, 0, 0], a=1, b=1) # Scalar
        assert cpu_out.shape == ()



# %% Function test
def test_stack() :
    '''
    Test ufunc function
    '''
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", category=NumbaPerformanceWarning)

        class MyClass() :
            @ufunc(data=['constant']) # <-- HERE IS THE DECORATOR TO USE
            def myfunc(x, y, constant, /, a, b=1) :
                return a * x + b * y + constant
            cuda = False
        
        # --- Instanciation ---

        total = 1e7 # total bytes
        N = int(total**(1/3))
        instance = MyClass()
        x = np.arange(N).reshape((1, N))
        y = np.arange(N).reshape((N, 1))
        constant = np.ones((N, N, N))
        a = np.arange(N)
        b = 0.5

        # --- CPU calculations ---

        cpu_out = instance.myfunc(x, y, constant, a=a, b=b)
        tic = perf_counter()
        cpu_out = instance.myfunc(x, y, constant, a=a, b=b)
        toc = perf_counter()
        cpu_time = toc-tic

        # --- GPU calculations ---

        if cp is None : return
        instance.cuda = True
        x = cp.asarray(x)
        y = cp.asarray(y)
        constant = cp.asarray(constant)
        a = cp.asarray(a)
        b = cp.asarray(b)

        gpu_out = instance.myfunc(x, y, constant, a=a, b=b)
        tic = perf_counter()
        gpu_out = instance.myfunc(x, y, constant, a=a, b=b)
        toc = perf_counter()
        gpu_time = toc-tic
        gpu_out = cp.asnumpy(gpu_out)

        # --- Error ---
        error = np.abs(cpu_out - gpu_out).max()
        assert error < 1e-3

        # --- Time ---
        assert cpu_time > gpu_time
        # print(f'CPU time: {cpu_time}s')
        # print(f'GPU time: {gpu_time}s')




# %% Test function run
if __name__ == "__main__":
    from corelp import test
    test(__file__)