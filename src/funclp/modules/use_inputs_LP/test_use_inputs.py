#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# Date          : 2025-12-16
# Author        : Lancelot PINCET
# GitHub        : https://github.com/LancelotPincet
# Library       : funcLP
# Module        : use_inputs

"""
This file allows to test use_inputs

use_inputs : This function takes called inputs and translate them into usable objects.
"""



# %% Libraries
from corelp import print, debug
import pytest
from funclp import use_inputs
debug_folder = debug(__file__)



# %% Function test
def test_function() :
    '''
    Test use_inputs function
    '''
    print('Hello world!')



# %% Instance fixture
@pytest.fixture()
def instance() :
    '''
    Create a new instance at each test function
    '''
    return use_inputs()

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
    Test use_inputs return values
    '''
    assert use_inputs(*args, **kwargs) == expected, message



# %% Error test
@pytest.mark.parametrize("args, kwargs, error, error_message", [
    #([], {}, None, ""),
    ([], {}, None, ""),
])
def test_errors(args, kwargs, error, error_message) :
    '''
    Test use_inputs error values
    '''
    with pytest.raises(error, match=error_message) :
        use_inputs(*args, **kwargs)



# %% Test function run
if __name__ == "__main__":
    from corelp import test
    test(__file__)