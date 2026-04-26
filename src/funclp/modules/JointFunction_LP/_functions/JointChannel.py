#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# Author        : Lancelot PINCET
# GitHub        : https://github.com/LancelotPincet



# %% Libraries
import numpy as np



# %% Channel definition
class JointChannel:
    '''Single channel definition for JointFunction.

    Parameters
    ----------
    function : Function
        Channel function model.
    variables : dict, optional
        Mapping from joint/global variable names to this channel variable names.
        Example: {'x': 'xx', 'y': 'yy'} means joint x -> channel xx and
        joint y -> channel yy.
    affine : dict, optional
        Fixed affine transforms applied on channel variables.
        Keys are tuples of channel variable names, values are homogeneous
        affine matrices with shape (n + 1, n + 1).
        Example: {('x', 'y'): matrix_3x3}.
    data : dict, optional
        Mapping from user-facing per-channel data names to this channel data names.
        Usually optional because data inputs remain channel-local and can be passed
        as tuples/lists in the channel data order.
    prefix : str, optional
        Prefix used for channel-specific parameters and constants.
    '''

    def __init__(self, function, variables=None, affine=None, data=None, prefix=None):
        self.function = function
        self.variables = {} if variables is None else dict(variables)
        self.data = {} if data is None else dict(data)
        self.prefix = prefix
        self.affine = {} if affine is None else dict(affine)
        self._normalize_affine()

    def _normalize_affine(self):
        normalized = {}
        for group, matrix in self.affine.items():
            if isinstance(group, str):
                group = (group,)
            else:
                group = tuple(group)
            if len(group) == 0:
                raise ValueError('affine groups cannot be empty')
            matrix = np.asarray(matrix, dtype=np.float32)
            expected = (len(group) + 1, len(group) + 1)
            if matrix.shape != expected:
                raise ValueError(f'affine matrix for {group} must have shape {expected}')
            bottom = np.zeros(len(group) + 1, dtype=np.float32)
            bottom[-1] = 1
            if not np.allclose(matrix[-1], bottom):
                raise ValueError(f'affine matrix for {group} must use homogeneous bottom row {bottom.tolist()}')
            normalized[group] = matrix
        self.affine = normalized

