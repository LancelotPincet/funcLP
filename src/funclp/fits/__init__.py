#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# Date          : 2025-12-17
# Author        : Lancelot PINCET
# GitHub        : https://github.com/LancelotPincet
# Library       : funcLP
# Module        : function

"""
Submodule where several predefined fits objects are defined.
"""



# %% libraries
import importlib
from pathlib import Path



# %% functions

_root = Path(__file__).parent

modules = { # {"function" : ".relative.path.to.file"}
    file.stem: "." + ".".join(file.relative_to(_root).with_suffix("").parts)
    for file in _root.rglob("*.py")
    if not file.name.startswith("_") and file.name != "__init__.py"
}



# %% Lazy loading

__all__ = list(modules.keys())

def __getattr__(name):
    if name not in modules:
        raise AttributeError(f"module funclp.function has no attribute {name}")

    module_path = modules[name]

    # Import the module lazily
    module = importlib.import_module(module_path, __name__)

    # Get the attribute from the imported module
    attr = getattr(module, name)

    # Cache it on this module so future access is fast
    globals()[name] = attr

    return attr