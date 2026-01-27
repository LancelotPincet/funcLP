#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# Date          : 2026-01-27
# Author        : Lancelot PINCET
# GitHub        : https://github.com/LancelotPincet
# Library       : funcLP
# Module        : estimators

"""
Contains all estimators definitions.
"""



# %% Libraries
from pathlib import Path

path = Path(__file__)
estimators = [file.stem for file in (path.parent / 'functions').rglob('*.py') if not file.stem.startswith('_')]



# %% Test function run
if __name__ == "__main__":
    from corelp import test
    test(__file__)