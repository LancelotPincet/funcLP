#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# Date          : 2026-04-26
# Author        : Lancelot PINCET
# GitHub        : https://github.com/LancelotPincet
# Library       : funcLP

from pathlib import Path

print("Generating cached files for functions...")

_function_files = Path(__file__).parent.parent / 'Function_LP' / '_functions'
for file in _function_files.glob('**/*.py'):
    if not file.name.startswith('_'):
        module_name = file.stem
        print(f"  Importing {module_name}...")
        exec(f'from funclp import {module_name}')

print("Done.")