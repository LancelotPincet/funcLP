#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# Date          : 2026-01-27
# Author        : Lancelot PINCET
# GitHub        : https://github.com/LancelotPincet
# Library       : template_[library]
# Module        : fits

"""
This file allows to test fits

fits : Contains all fitting algorithm definitions.
"""



# %% Libraries
from corelp import print, debug
import pytest
from template_[lowerlib] import fits
debug_folder = debug(__file__)



# %% Function test
def test_function() :
    '''
    Test fits function
    '''
    print('Hello world!')



# %% Instance fixture
@pytest.fixture()
def instance() :
    '''
    Create a new instance at each test function
    '''
    return fits()

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
    Test fits return values
    '''
    assert fits(*args, **kwargs) == expected, message



# %% Error test
@pytest.mark.parametrize("args, kwargs, error, error_message", [
    #([], {}, None, ""),
    ([], {}, None, ""),
])
def test_errors(args, kwargs, error, error_message) :
    '''
    Test fits error values
    '''
    with pytest.raises(error, match=error_message) :
        fits(*args, **kwargs)



# %% Test function run
if __name__ == "__main__":
    from corelp import test
    test(__file__)