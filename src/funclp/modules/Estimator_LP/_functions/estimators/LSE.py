#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# Date          : 2026-01-01
# Author        : Lancelot PINCET
# GitHub        : https://github.com/LancelotPincet
# Library       : funcLP



# %% Libraries
from .MLE import MLE
from funclp import Normal



# %% Function

class LSE(MLE) :

    def __init__(self, distribution=None, **kwargs) :
        if distribution is not None :
            raise SyntaxError('LSE cannot have a distribution')
        super().__init__(Normal(), **kwargs)