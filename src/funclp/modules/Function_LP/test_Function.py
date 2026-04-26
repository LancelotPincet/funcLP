#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# Date          : 2025-12-17
# Author        : Lancelot PINCET
# GitHub        : https://github.com/LancelotPincet
# Library       : funcLP
# Module        : Function

"""
Tests for the Function class.

Function : Abstract base class for function models.
"""

# %% Libraries
import pytest
import numpy as np
from funclp import IsoGaussian


# %% Test IsoGaussian (concrete implementation)
class TestIsoGaussian:
    @pytest.fixture
    def instance(self):
        return IsoGaussian()

    @pytest.fixture
    def instance_with_params(self):
        return IsoGaussian(mux=np.float32(1.0), muy=np.float32(2.0), sig=np.float32(0.5))

    def test_name(self, instance):
        assert instance.name == "IsoGaussian"

    def test_parameters(self, instance_with_params):
        assert instance_with_params.mux == np.float32(1.0)
        assert instance_with_params.muy == np.float32(2.0)
        assert instance_with_params.sig == np.float32(0.5)

    def test_nmodels_scalar(self, instance):
        assert instance.nmodels == 0

    def test_nmodels_vector(self, instance_with_params):
        assert instance_with_params.nmodels == 1

    def test_nmodels_broadcast(self):
        func = IsoGaussian(mux=np.arange(3), muy=np.float32(1.0), sig=np.float32(0.5))
        assert func.nmodels == 3


class TestIsoGaussianComputation:
    def test_cpu_function(self):
        func = IsoGaussian(cuda=False, mux=np.float32(0.0), muy=np.float32(0.0), sig=np.float32(1.0))
        x = np.linspace(-3, 3, 7)
        result = func(x)
        assert result.shape == (1, 7)
        assert not np.isnan(result).any()

    def test_gpu_function(self):
        func = IsoGaussian(cuda=True, mux=np.float32(0.0), muy=np.float32(0.0), sig=np.float32(1.0))
        x = np.linspace(-3, 3, 7)
        result = func(x)
        assert result.shape == (1, 7)
        assert not np.isnan(result).any()


class TestIsoGaussianDerivatives:
    def test_d_mux_exists(self):
        func = IsoGaussian()
        assert hasattr(func, "d_mux")

    def test_d_sig_exists(self):
        func = IsoGaussian()
        assert hasattr(func, "d_sig")


class TestIsoGaussianFit:
    def test_fit_attributes(self):
        func = IsoGaussian(mux=np.float32(1.0), mux_fit=True)
        assert func.mux_fit is True

    def test_bounds_default(self):
        func = IsoGaussian()
        assert func.mux_min == -np.inf
        assert func.mux_max == np.inf


class TestIsoGaussianErrors:
    def test_missing_constant_raises(self):
        with pytest.raises(SyntaxError):
            _ = IsoGaussian(muy=np.float32(1.0))

    def test_invalid_dtype_raises(self):
        with pytest.raises(TypeError):
            _ = IsoGaussian(mux="not_a_number")


# %% Test function run
if __name__ == "__main__":
    from corelp import test
    test(__file__)