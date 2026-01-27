#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# Date          : 2026-01-27
# Author        : Lancelot PINCET
# GitHub        : https://github.com/LancelotPincet
# Library       : template_[library]
# Module        : distributions

"""
This file allows to test distributions

distributions : Contains all distributions definitions.
"""



# %% Libraries
from corelp import print, debug
import pytest
from template_[lowerlib] import distributions
debug_folder = debug(__file__)



# %% Function test
def test_function() :
    '''
    Test distributions function
    '''
    print('Hello world!')



# %% Instance fixture
@pytest.fixture()
def instance() :
    '''
    Create a new instance at each test function
    '''
    return distributions()

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
    Test distributions return values
    '''
    assert distributions(*args, **kwargs) == expected, message



# %% Error test
@pytest.mark.parametrize("args, kwargs, error, error_message", [
    #([], {}, None, ""),
    ([], {}, None, ""),
])
def test_errors(args, kwargs, error, error_message) :
    '''
    Test distributions error values
    '''
    with pytest.raises(error, match=error_message) :
        distributions(*args, **kwargs)



# %% Test function run
if __name__ == "__main__":
    from corelp import test
    test(__file__)