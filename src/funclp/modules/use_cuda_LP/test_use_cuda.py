#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# Date          : 2025-12-17
# Author        : Lancelot PINCET
# GitHub        : https://github.com/LancelotPincet
# Library       : funcLP
# Module        : use_cuda

"""
This file allows to test use_cuda

use_cuda : Defines cuda parameters.
"""



# %% Libraries
from corelp import print, debug
import pytest
from funclp import use_cuda
debug_folder = debug(__file__)



# %% Function test
def test_function() :
    '''
    Test use_cuda function
    '''
    print('Hello world!')



# %% Instance fixture
@pytest.fixture()
def instance() :
    '''
    Create a new instance at each test function
    '''
    return use_cuda()

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
    Test use_cuda return values
    '''
    assert use_cuda(*args, **kwargs) == expected, message



# %% Error test
@pytest.mark.parametrize("args, kwargs, error, error_message", [
    #([], {}, None, ""),
    ([], {}, None, ""),
])
def test_errors(args, kwargs, error, error_message) :
    '''
    Test use_cuda error values
    '''
    with pytest.raises(error, match=error_message) :
        use_cuda(*args, **kwargs)



# %% Test function run
if __name__ == "__main__":
    from corelp import test
    test(__file__)