#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# Date          : 2026-05-25
# Author        : Lancelot PINCET
# GitHub        : https://github.com/LancelotPincet
# Library       : funcLP
# Module        : kernel_caching

"""
This function caches internally dynamic numba kernels as files.
"""



# %% Libraries
from pathlib import Path
import importlib
import sys



# %% Parameters
CACHE_FOLDER = Path(__file__).parent / "_functions" / "cached"
CACHE_PACKAGE = "funclp.modules.kernel_caching_LP._functions.cached"



# %% Function
def kernel_caching(module_name, source, object_name=None, *, package=CACHE_PACKAGE, folder=CACHE_FOLDER):
    """Cache a generated module and optionally return one object from it."""
    if module_name.endswith(".py"):
        module_name = module_name[:-3]

    folder = Path(folder)
    folder.mkdir(parents=True, exist_ok=True)
    init_file = folder / "__init__.py"
    if not init_file.exists():
        init_file.write_text("")

    file = folder / f"{module_name}.py"
    previous = file.read_text() if file.exists() else None
    changed = previous != source
    if changed:
        file.write_text(source)

    module_path = f"{package}.{module_name}"
    if changed:
        importlib.invalidate_caches()
    if module_path in sys.modules:
        if changed:
            module = importlib.reload(sys.modules[module_path])
        else:
            module = sys.modules[module_path]
    else:
        module = importlib.import_module(module_path)

    if object_name is None:
        return module
    return getattr(module, object_name)


# %% Test function run
if __name__ == "__main__":
    from corelp import test
    test(__file__)
