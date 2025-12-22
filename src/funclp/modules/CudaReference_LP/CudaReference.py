#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# Date          : 2025-12-21
# Author        : Lancelot PINCET
# GitHub        : https://github.com/LancelotPincet
# Library       : funcLP
# Module        : CudaReference

"""
This class serves as a parent class and gives parameters defining cuda usage.
"""



# %% Libraries
from corelp import prop



# %% Class
class CudaReference() :
    '''
    This class serves as a parent class and gives parameters defining cuda usage.
    
    Attributes
    ----------
    cuda_reference : object
        Object on which we get all the cuda parameters, if None uses default from "_cudaattribut".
    cuda : bool
        True to calculate on GPU, False for CPU, None for automatic.
    cpu2gpu : float
        Number of bytes from which we consider calculating on GPU is worth.
    '''

    cuda_reference = None # Object on which we get all the cuda parameters, if None uses default

    _cuda = None # True / False / None for auto
    @prop(link='cuda_reference')
    def cuda(self) :
        return "cuda"

    _cpu2gpu = 1e6 #bytes
    @prop(link='cuda_reference')
    def cpu2gpu(self) :
        return "cpu2gpu"



# %% Test function run
if __name__ == "__main__":
    from corelp import test
    test(__file__)