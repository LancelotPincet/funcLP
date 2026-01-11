#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# Date          : 2025-12-20
# Author        : Lancelot PINCET
# GitHub        : https://github.com/LancelotPincet
# Library       : funcLP
# Module        : Fit

"""
This file allows to test Fit

Fit : Class defining fitting algorithms.
"""



# %% Libraries
from corelp import debug
import pytest
from funclp import Fit
debug_folder = debug(__file__)



# %% Function test
def test_function() :
    '''
    Test Fit function
    '''
    print('Hello world!')



# %% Instance fixture
@pytest.fixture()
def instance() :
    '''
    Create a new instance at each test function
    '''
    return Fit()

def test_instance(instance) :
    '''
    Test on fixture
    '''
    pass


# %% Returns test
@pytest.mark.parametrize("args, kwargs, expected, message", [
    #([], {}, None, ""),
    ([], {}, None, ""),
])
def test_returns(args, kwargs, expected, message) :
    '''
    Test Fit return values
    '''
    assert Fit(*args, **kwargs) == expected, message



# %% Error test
@pytest.mark.parametrize("args, kwargs, error, error_message", [
    #([], {}, None, ""),
    ([], {}, None, ""),
])
def test_errors(args, kwargs, error, error_message) :
    '''
    Test Fit error values
    '''
    with pytest.raises(error, match=error_message) :
        Fit(*args, **kwargs)



# %% Test function run
if __name__ == "__main__":
    from corelp import test
    test(__file__)