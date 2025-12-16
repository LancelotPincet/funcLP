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
try :
    import cupy as cp
except ImportError :
    cp = None
debug_folder = debug(__file__)



# %% Function test
def test_function() :
    '''
    Test ufunc function
    '''

    class MyClass() :
        @ufunc() # <-- HERE IS THE DECORATOR TO USE
        def myfunc(x, y, /, constant, *, a, b) :
            return a * x + b * y + constant
        cuda = False
    
    # --- Instanciation ---

    instance = MyClass()
    v = np.arange(20)
    x, y = np.meshgrid(v, v)
    constant = np.ones((5, 20, 20))
    a = np.arange(5)
    b = 0.5

    # --- CPU calculations ---

    # cpu_out = instance.myfunc(x, y, constant, a=a, b=b)

    # --- GPU calculations ---

    if cp is None : return
    instance.cuda = True

    gpu_out = instance.myfunc(x, y, constant, a=a, b=b)




# %% Test function run
if __name__ == "__main__":
    from corelp import test
    test(__file__)