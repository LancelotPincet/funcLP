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
from corelp import print, debug
from funclp import ufunc
import numpy as np
from numba import cuda
debug_folder = debug(__file__)



# %% Function test
def test_function() :
    '''
    Test ufunc function
    '''

    class MyClass() :
        @ufunc() # <-- HERE IS THE DECORATOR TO USE
        def myfunc(x, /, array, *, a, b) :
            return a * x + b
    
        def deco(*args, **kwargs) :
    # --- Instanciation ---

    instance = MyClass()

    # For flat array
    x = np.arange(20, dtype=np.float32) # variable
    one = np.float32(1) # a parameter
    zero = np.float32(0) # a parameter
    out = np.empty(20, dtype=np.float32) # output

    # For stack array
    xs = np.asarray([x, x], dtype=np.float32) # variable
    ones = np.ones(2, dtype=np.float32) # a parameter
    zeros = np.zeros(2, dtype=np.float32) # a parameter
    outs = np.empty((2, 20), dtype=np.float32) # output
    
    # --- CPU calculations ---

    # default changes when changing target
    instance.target = 'cpu'
    instance.myfunc(x, one, zero, out) # default function for flat array (1D)
    instance.stackmyfunc(xs, ones, zeros, outs) # default function for stack (2D)
    
    # other jitted functions
    instance.cpukernel_myfunc(x[0], one[0], zero[0]) # scalar function jitted for cpu
    instance.cpu_myfunc(x, one, zero, out) # jitted function for flat arrays on cpu
    instance.cpu_stackmyfunc(xs, ones, zeros, outs) # jitted function for stack array on cpu
    instance.cpuvec_myfunc(x, one, zero, out) # vectorized function for ndarrays on cpu

    # --- Parallel CPU cores calculations ---

    # default changes when changing target
    instance.target = 'par'
    instance.myfunc(x, one, zero, out) # default function for flat array (1D)
    instance.stackmyfunc(xs, ones, zeros, outs) # default function for stack (2D)
    
    # other jitted functions
    instance.par_myfunc(x, one, zero, out) # jitted function for flat arrays on parallel cpu
    instance.par_stackmyfunc(xs, ones, zeros, outs) # jitted function for stack array on parallel cpu
    instance.parvec_myfunc(x, one, zero, out) # vectorized function for ndarrays on parallel cpu

    # CUDA GPU calculations
    if cuda.is_available() :

        # Passing to device
        x = cuda.to_device(x)
        one = cuda.to_device(one)
        zero = cuda.to_device(zero)
        out = cuda.to_device(out)
        xs = cuda.to_device(xs)
        ones = cuda.to_device(ones)
        zeros = cuda.to_device(zeros)
        outs = cuda.to_device(outs)
        threads = 128
        blocks = (out.size + threads - 1) // threads

        # default changes when changing target
        instance.target = 'gpu'
        instance.myfunc[blocks, threads](x, one, zero, out) # default function for flat array (1D)
        instance.stackmyfunc[blocks, threads](xs, ones, zeros, outs) # default function for stack (2D)
    
        # other jitted functions
        instance.gpu_myfunc[blocks, threads](x, one, zero, out) # jitted function for flat arrays on gpu
        instance.gpu_stackmyfunc[blocks, threads](xs, ones, zeros, outs) # jitted function for stack array on gpu
        instance.gpuvec_myfunc[blocks, threads](x, one, zero, out) # vectorized function for ndarrays on gpu



# %% Test function run
if __name__ == "__main__":
    from corelp import test
    test(__file__)