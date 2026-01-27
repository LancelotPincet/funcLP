#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# Date          : 2026-01-27
# Author        : Lancelot PINCET
# GitHub        : https://github.com/LancelotPincet
# Library       : template_[library]
# Module        : estimators

"""
This file allows to test estimators

estimators : Contains all estimators definitions.
"""



# %% Libraries
from corelp import print, debug
import pytest
from template_[lowerlib] import estimators
debug_folder = debug(__file__)



# %% Function test
def test_function() :
    '''
    Test estimators function
    '''
    print('Hello world!')



# %% Instance fixture
@pytest.fixture()
def instance() :
    '''
    Create a new instance at each test function
    '''
    return estimators()

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
    Test estimators return values
    '''
    assert estimators(*args, **kwargs) == expected, message



# %% Error test
@pytest.mark.parametrize("args, kwargs, error, error_message", [
    #([], {}, None, ""),
    ([], {}, None, ""),
])
def test_errors(args, kwargs, error, error_message) :
    '''
    Test estimators error values
    '''
    with pytest.raises(error, match=error_message) :
        estimators(*args, **kwargs)



# %% Test function run
if __name__ == "__main__":
    from corelp import test
    test(__file__)