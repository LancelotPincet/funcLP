#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# Date          : 2026-05-25
# Author        : Lancelot PINCET
# GitHub        : https://github.com/LancelotPincet
# Library       : funcLP

"""Generate all dynamic kernels into kernel_caching cache folder."""



# %% Libraries
from pathlib import Path
import inspect
import numpy as np

import funclp
from funclp import Distribution, Estimator, Fit, Function



# %% Helpers
def _iter_function_classes():
    """Iterate concrete Function classes from function modules."""
    classes = []
    for name in funclp.__all__:
        obj = getattr(funclp, name)
        if not inspect.isclass(obj):
            continue
        if not issubclass(obj, Function) or obj is Function:
            continue
        if "funclp.modules.Function_LP._functions" not in obj.__module__:
            continue
        classes.append(obj)
    return classes


def _iter_distribution_classes():
    """Iterate concrete Distribution classes."""
    classes = []
    for name in funclp.__all__:
        obj = getattr(funclp, name)
        if not inspect.isclass(obj):
            continue
        if not issubclass(obj, Distribution) or obj is Distribution:
            continue
        if "funclp.modules.Distribution_LP._functions.distributions" not in obj.__module__:
            continue
        classes.append(obj)
    return classes


def _iter_estimator_classes():
    """Iterate concrete Estimator classes."""
    classes = []
    for name in funclp.__all__:
        obj = getattr(funclp, name)
        if not inspect.isclass(obj):
            continue
        if not issubclass(obj, Estimator) or obj is Estimator:
            continue
        if "funclp.modules.Estimator_LP._functions.estimators" not in obj.__module__:
            continue
        classes.append(obj)
    return classes


def _iter_optimizer_classes():
    """Iterate concrete Fit optimizer classes."""
    classes = []
    for name in funclp.__all__:
        obj = getattr(funclp, name)
        if not inspect.isclass(obj):
            continue
        if not issubclass(obj, Fit) or obj is Fit:
            continue
        if "funclp.modules.Fit_LP._functions.optimizers" not in obj.__module__:
            continue
        classes.append(obj)
    return classes


def _instantiate_function(function_cls):
    """Instantiate a Function class with dummy constants when needed."""
    if function_cls.__name__ == "Spline":
        x = np.linspace(-1, 1, 16, dtype=np.float32)
        model = np.exp(-x**2).astype(np.float32)
        return function_cls(model, x)

    kwargs = {}
    constants = list(function_cls.function.constants)
    for constant in constants:
        if constant == "coeffs":
            kwargs[constant] = np.ones((8, 8), dtype=np.float32)
        elif constant in {"tx", "ty", "tz", "t"}:
            kwargs[constant] = np.linspace(-1, 1, 12, dtype=np.float32)
        else:
            kwargs[constant] = np.float32(1.0)
    return function_cls(**kwargs)


def _iter_instantiable_function_classes():
    """Yield function classes that can be instantiated for fit-kernel generation."""
    for function_cls in _iter_function_classes():
        try:
            _ = _instantiate_function(function_cls)
        except Exception:
            continue
        yield function_cls


def _instantiate_estimators_for_all_distributions():
    """Instantiate all estimator/distribution combinations used by fit kernels."""
    estimators = []
    distribution_classes = _iter_distribution_classes()
    for estimator_cls in _iter_estimator_classes():
        if estimator_cls.__name__ == "LSE":
            estimators.append(estimator_cls())
            continue
        for distribution_cls in distribution_classes:
            estimators.append(estimator_cls(distribution_cls()))
    return estimators



# %% Tests
def test_cache_all_ufunc_dynamic_kernels():
    """Import all function modules and trigger ufunc dynamic kernel generation."""
    function_files = Path(__file__).parents[2] / "Function_LP" / "_functions"
    for file in function_files.glob("**/*.py"):
        if file.name.startswith("_"):
            continue
        module_name = file.stem
        _ = getattr(funclp, module_name)


def test_cache_all_estimator_dynamic_kernels():
    """Generate all cached estimator helper kernels."""
    for estimator in _instantiate_estimators_for_all_distributions():
        _ = estimator.cpukernel_deviance
        _ = estimator.gpukernel_deviance
        _ = estimator.cpukernel_loss
        _ = estimator.gpukernel_loss
        _ = estimator.cpukernel_observed
        _ = estimator.gpukernel_observed
        _ = estimator.cpukernel_fisher
        _ = estimator.gpukernel_fisher


def test_cache_all_fit_assembly_dynamic_kernels():
    """Generate all cached assembly kernels for function/estimator combos."""
    optimizer_classes = _iter_optimizer_classes()
    estimators = _instantiate_estimators_for_all_distributions()
    for function_cls in _iter_instantiable_function_classes():
        for estimator in estimators:
            for optimizer_cls in optimizer_classes:
                fit = optimizer_cls(_instantiate_function(function_cls), estimator)
                _ = fit.cpu_assembly
                _ = fit.gpu_assembly


def test_cache_all_fit_trial_dynamic_kernels():
    """Generate all cached trial kernels for optimizer classes that use them."""
    optimizer_classes = _iter_optimizer_classes()
    estimators = _instantiate_estimators_for_all_distributions()
    for function_cls in _iter_instantiable_function_classes():
        for estimator in estimators:
            for optimizer_cls in optimizer_classes:
                fit = optimizer_cls(_instantiate_function(function_cls), estimator)
                if hasattr(fit, "cpu_trial_chi2"):
                    _ = fit.cpu_trial_chi2
                if hasattr(fit, "gpu_trial_chi2"):
                    _ = fit.gpu_trial_chi2



# %% Test function run
if __name__ == "__main__":
    from corelp import test

    test(__file__)
