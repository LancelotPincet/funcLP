#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# Author        : Lancelot PINCET
# GitHub        : https://github.com/LancelotPincet



class Parameter:
    '''Metadata for a fit function parameter.'''

    def __init__(self, name, default, *, estimate=None, bounds=(None, None), fit=None):
        self.name = name
        self.default = default
        self.estimate = estimate
        self.bounds = bounds
        self.fit = estimate is not None if fit is None else fit

