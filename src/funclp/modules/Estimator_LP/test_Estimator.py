#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# Date          : 2025-12-19
# Author        : Lancelot PINCET
# GitHub        : https://github.com/LancelotPincet
# Library       : funcLP
# Module        : Estimator

"""
Tests for the Estimator class.

Estimator : Abstract base class for estimator functions used in fitting.
"""

# %% Libraries
import pytest
import numpy as np
from funclp import MLE, LSE, Normal, Poisson


class TestMLE:
    def test_name(self):
        estimator = MLE(Normal())
        assert estimator.name == "MLE"

    def test_distribution_set(self):
        dist = Normal()
        estimator = MLE(dist)
        assert estimator.distribution is dist


class TestLSE:
    def test_name(self):
        estimator = LSE()
        assert estimator.name == "LSE"

    def test_automatic_normal_distribution(self):
        estimator = LSE()
        assert estimator.distribution is not None
        assert estimator.distribution.__class__.__name__ == "Normal"

    def test_distribution_not_allowed(self):
        with pytest.raises(SyntaxError):
            _ = LSE(Normal())


class TestMLEWithNormal:
    @pytest.fixture
    def estimator(self):
        return MLE(Normal())

    def test_deviance(self, estimator):
        raw = np.float32(5.0)
        model = np.float32(4.0)
        weights = np.float32(1.0)
        result = estimator.deviance(raw, model, weights)
        assert isinstance(result, (float, np.floating))

    def test_loss(self, estimator):
        raw = np.float32(5.0)
        model = np.float32(4.0)
        weights = np.float32(1.0)
        result = estimator.loss(raw, model, weights)
        assert isinstance(result, (float, np.floating))

    def test_observed(self, estimator):
        raw = np.float32(5.0)
        model = np.float32(4.0)
        weights = np.float32(1.0)
        result = estimator.observed(raw, model, weights)
        assert isinstance(result, (float, np.floating))

    def test_fisher(self, estimator):
        raw = np.float32(5.0)
        model = np.float32(4.0)
        weights = np.float32(1.0)
        result = estimator.fisher(raw, model, weights)
        assert isinstance(result, (float, np.floating))


class TestMLEWithPoisson:
    @pytest.fixture
    def estimator(self):
        return MLE(Poisson())

    def test_deviance(self, estimator):
        raw = np.float32(5.0)
        model = np.float32(4.0)
        weights = np.float32(1.0)
        result = estimator.deviance(raw, model, weights)
        assert isinstance(result, (float, np.floating))

    def test_loss(self, estimator):
        raw = np.float32(5.0)
        model = np.float32(4.0)
        weights = np.float32(1.0)
        result = estimator.loss(raw, model, weights)
        assert isinstance(result, (float, np.floating))


# %% Test function run
if __name__ == "__main__":
    from corelp import test
    test(__file__)