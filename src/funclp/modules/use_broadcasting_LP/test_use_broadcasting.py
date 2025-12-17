#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# Date          : 2025-12-17
# Author        : Lancelot PINCET
# GitHub        : https://github.com/LancelotPincet
# Library       : funcLP
# Module        : use_broadcasting

"""
This file allows to test use_broadcasting

use_broadcasting : Defines inputs after broadcasting ready to use.
"""



# %% Libraries
from corelp import print, debug
import pytest
from funclp import use_broadcasting
debug_folder = debug(__file__)



# %% Function test
def test_function() :
    '''
    Test use_broadcasting function
    '''
    print('Hello world!')



# %% Instance fixture
@pytest.fixture()
def instance() :
    '''
    Create a new instance at each test function
    '''
    return use_broadcasting()

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
    Test use_broadcasting return values
    '''
    assert use_broadcasting(*args, **kwargs) == expected, message



# %% Error test
@pytest.mark.parametrize("args, kwargs, error, error_message", [
    #([], {}, None, ""),
    ([], {}, None, ""),
])
def test_errors(args, kwargs, error, error_message) :
    '''
    Test use_broadcasting error values
    '''
    with pytest.raises(error, match=error_message) :
        use_broadcasting(*args, **kwargs)



# %% Test function run
if __name__ == "__main__":
    from corelp import test
    test(__file__)