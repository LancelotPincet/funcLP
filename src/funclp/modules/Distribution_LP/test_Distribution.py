#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# Date          : 2025-12-19
# Author        : Lancelot PINCET
# GitHub        : https://github.com/LancelotPincet
# Library       : funcLP
# Module        : Distribution

"""
Tests for the Distribution class.

Distribution : Abstract base class for noise distributions in data.
"""

# %% Libraries
import pytest
import numpy as np
from funclp import Normal, Poisson, Binomial, Gamma


class TestNormal:
    def test_name(self):
        dist = Normal()
        assert dist.name == "Normal"

    def test_pdf(self):
        dist = Normal()
        raw = np.float32(5.0)
        model = np.float32(4.0)
        weights = np.float32(1.0)
        result = dist.pdf(raw, model, weights)
        assert result > 0

    def test_loglikelihood_reduced(self):
        dist = Normal()
        raw = np.float32(5.0)
        model = np.float32(4.0)
        weights = np.float32(1.0)
        result = dist.loglikelihood_reduced(raw, model, weights)
        assert isinstance(result, (float, np.floating))

    def test_loglikelihood(self):
        dist = Normal()
        raw = np.float32(5.0)
        model = np.float32(4.0)
        weights = np.float32(1.0)
        result = dist.loglikelihood(raw, model, weights)
        assert isinstance(result, (float, np.floating))

    def test_dloglikelihood(self):
        dist = Normal()
        raw = np.float32(5.0)
        model = np.float32(4.0)
        weights = np.float32(1.0)
        result = dist.dloglikelihood(raw, model, weights)
        assert isinstance(result, (float, np.floating))

    def test_d2loglikelihood(self):
        dist = Normal()
        raw = np.float32(5.0)
        model = np.float32(4.0)
        weights = np.float32(1.0)
        result = dist.d2loglikelihood(raw, model, weights)
        assert isinstance(result, (float, np.floating))

    def test_fisher(self):
        dist = Normal()
        raw = np.float32(5.0)
        model = np.float32(4.0)
        weights = np.float32(1.0)
        result = dist.fisher(raw, model, weights)
        assert result > 0


class TestPoisson:
    def test_name(self):
        dist = Poisson()
        assert dist.name == "Poisson"

    def test_pdf(self):
        dist = Poisson()
        raw = np.float32(5.0)
        model = np.float32(4.0)
        weights = np.float32(1.0)
        result = dist.pdf(raw, model, weights)
        assert result >= 0

    def test_loglikelihood(self):
        dist = Poisson()
        raw = np.float32(5.0)
        model = np.float32(4.0)
        weights = np.float32(1.0)
        result = dist.loglikelihood(raw, model, weights)
        assert isinstance(result, (float, np.floating))

    def test_dloglikelihood(self):
        dist = Poisson()
        raw = np.float32(5.0)
        model = np.float32(4.0)
        weights = np.float32(1.0)
        result = dist.dloglikelihood(raw, model, weights)
        assert isinstance(result, (float, np.floating))

    def test_d2loglikelihood(self):
        dist = Poisson()
        raw = np.float32(5.0)
        model = np.float32(4.0)
        weights = np.float32(1.0)
        result = dist.d2loglikelihood(raw, model, weights)
        assert isinstance(result, (float, np.floating))

    def test_fisher(self):
        dist = Poisson()
        raw = np.float32(5.0)
        model = np.float32(4.0)
        weights = np.float32(1.0)
        result = dist.fisher(raw, model, weights)
        assert result >= 0


class TestBinomial:
    def test_name(self):
        dist = Binomial()
        assert dist.name == "Binomial"

    def test_pdf(self):
        dist = Binomial()
        raw = np.float32(5.0)
        model = np.float32(0.5)
        weights = np.float32(1.0)
        result = dist.pdf(raw, model, weights)
        assert result >= 0


class TestGamma:
    def test_name(self):
        dist = Gamma()
        assert dist.name == "Gamma"

    def test_pdf(self):
        dist = Gamma()
        raw = np.float32(5.0)
        model = np.float32(4.0)
        weights = np.float32(1.0)
        result = dist.pdf(raw, model, weights)
        assert result >= 0


# %% Test function run
if __name__ == "__main__":
    from corelp import test
    test(__file__)