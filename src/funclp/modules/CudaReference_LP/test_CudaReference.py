#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# Date          : 2025-12-21
# Author        : Lancelot PINCET
# GitHub        : https://github.com/LancelotPincet
# Library       : funcLP
# Module        : CudaReference

"""
Tests for the CudaReference class.

CudaReference : Base class that provides parameters defining CUDA usage.
"""

# %% Libraries
import pytest
import numpy as np
from funclp import CudaReference, IsoGaussian


class TestCudaReference:
    def test_name(self):
        ref = CudaReference()
        assert ref.name == "CudaReference"

    def test_default_cuda_none(self):
        ref = CudaReference()
        assert ref.cuda is None

    def test_default_cpu2gpu(self):
        ref = CudaReference()
        assert ref.cpu2gpu == 1e6


class TestCudaReferenceInheritance:
    def test_inherits_from(self):
        func = IsoGaussian()
        assert isinstance(func, CudaReference)

    def test_cuda_reference_chain(self):
        func = IsoGaussian()
        assert func.cuda_reference is not None


class TestCudaReferenceAttributes:
    def test_can_set_cuda(self):
        func = IsoGaussian(cuda=False)
        assert func.cuda is False

    def test_can_set_cpu2gpu(self):
        func = IsoGaussian(cpu2gpu=1e3)
        assert func.cpu2gpu == 1e3


# %% Test function run
if __name__ == "__main__":
    from corelp import test
    test(__file__)