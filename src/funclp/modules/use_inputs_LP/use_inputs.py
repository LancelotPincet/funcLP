#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# Date          : 2025-12-16
# Author        : Lancelot PINCET
# GitHub        : https://github.com/LancelotPincet
# Library       : funcLP
# Module        : use_inputs

"""
This function takes called inputs and translate them into usable objects.
"""



# %% Libraries



# %% Function
def use_inputs(self, args, kwargs) :
    '''
    This function takes called inputs and translate them into usable objects.
    
    Parameters
    ----------
    self : ufunc
        ufunc object which was called.
    args : tuple
        Positional arguments used at function call time.
    kwargs : tuple
        Keyword arguments used at function call time.

    Returns
    -------
    variables : list
        list of variables arrays (CPU or GPU).
    data : list
        list of data arrays (CPU or GPU).
    parameters : list
        list of parameters arrays (CPU or GPU).
    '''

    # Separate inputs
    bound = self.signature.bind(*args, **kwargs)
    bound.apply_defaults()

    # Bounding
    variables = [bound.arguments[key] for key in self.variables]
    data = [bound.arguments[key] for key in self.data]
    parameters = [bound.arguments[key] for key in self.parameters]

    return variables, data, parameters





# %% Test function run
if __name__ == "__main__":
    from corelp import test
    test(__file__)