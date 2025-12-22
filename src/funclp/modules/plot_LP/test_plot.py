#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# Date          : 2025-12-18
# Author        : Lancelot PINCET
# GitHub        : https://github.com/LancelotPincet
# Library       : funcLP
# Module        : plot

"""
This file allows to test plot

plot : Makes a plot of a function object.
"""



# %% Libraries
from corelp import print, debug
import pytest
from funclp import plot
debug_folder = debug(__file__)



# %% Function test
def test_function() :
    '''
    Test plot function
    '''
    print('Hello world!')



# %% Instance fixture
@pytest.fixture()
def instance() :
    '''
    Create a new instance at each test function
    '''
    return plot()

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
    Test plot return values
    '''
    assert plot(*args, **kwargs) == expected, message



# %% Error test
@pytest.mark.parametrize("args, kwargs, error, error_message", [
    #([], {}, None, ""),
    ([], {}, None, ""),
])
def test_errors(args, kwargs, error, error_message) :
    '''
    Test plot error values
    '''
    with pytest.raises(error, match=error_message) :
        plot(*args, **kwargs)



# %% Test function run
if __name__ == "__main__":
    from corelp import test
    test(__file__)