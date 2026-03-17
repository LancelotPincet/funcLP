#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# Date          : 2026-01-01
# Author        : Lancelot PINCET
# Library       : funcLP

import numpy as np
import numba as nb
import math

# %% tools
def get_amp(y):
    return np.nanmax(y) - np.nanmin(y)

def get_offset(y):
    return np.nanmin(y)

def get_mean(y, x):
    y0 = y - np.nanmin(y)
    num = np.nansum(x * y0)
    denom = np.nansum(y0)
    return num / denom


@nb.njit(nogil=True, inline="always")
def find_span(t, k, x):
    n = len(t) - k - 1
    if x >= t[n]:
        return n - 1
    if x <= t[k]:
        return k
    lo = k
    hi = n
    while hi - lo > 1:
        mid = (lo + hi) >> 1
        if t[mid] <= x:
            lo = mid
        else:
            hi = mid
    return lo


@nb.njit(nogil=True, inline="always")
def bspline1d(tx, coeffs, kx, x):
    """
    Evaluate a 1D B-spline at x.
    De Boor's algorithm, fully inlined — no dynamic allocation,
    CUDA device function compatible.
    Supports kx up to 5.
    """
    if x < tx[kx] or x > tx[len(tx) - kx - 1]:
        return nb.float32(0.0)
    ix = find_span(tx, kx, x)

    # Local fixed-size scratch (max degree 5 => 6 values)
    # We use scalar variables to avoid np.zeros
    d0 = nb.float32(0.0)
    d1 = nb.float32(0.0)
    d2 = nb.float32(0.0)
    d3 = nb.float32(0.0)
    d4 = nb.float32(0.0)
    d5 = nb.float32(0.0)

    # Load the kx+1 relevant coefficients
    base = ix - kx
    if kx >= 0: d0 = coeffs[base + 0]
    if kx >= 1: d1 = coeffs[base + 1]
    if kx >= 2: d2 = coeffs[base + 2]
    if kx >= 3: d3 = coeffs[base + 3]
    if kx >= 4: d4 = coeffs[base + 4]
    if kx >= 5: d5 = coeffs[base + 5]

    # De Boor's recursion (triangular scheme), r=1..kx
    for r in range(1, kx + 1):
        for j in range(kx, r - 1, -1):
            tj  = tx[ix - kx + j]
            tjr = tx[ix + 1 + j - r]
            denom = tjr - tj
            if denom != 0.0:
                alpha = (x - tj) / denom
            else:
                alpha = nb.float32(0.0)
            if j == 5: d5 = (nb.float32(1.0) - alpha) * d4 + alpha * d5
            if j == 4: d4 = (nb.float32(1.0) - alpha) * d3 + alpha * d4
            if j == 3: d3 = (nb.float32(1.0) - alpha) * d2 + alpha * d3
            if j == 2: d2 = (nb.float32(1.0) - alpha) * d1 + alpha * d2
            if j == 1: d1 = (nb.float32(1.0) - alpha) * d0 + alpha * d1
            if j == 0: d0 = alpha * d0

    # The result sits at d[kx] after the full triangular sweep
    # but de Boor converges to d0 after kx steps when sweeping downward
    if kx == 0: return d0
    if kx == 1: return d1
    if kx == 2: return d2
    if kx == 3: return d3
    if kx == 4: return d4
    return d5


@nb.njit(nogil=True, inline="always")
def bspline2d(tx, ty, coeffs, kx, ky, x, y):
    """
    Evaluate a 2D tensor-product B-spline at (x, y).
    Reduces to ky+1 calls of 1D de Boor along x, then one along y.
    No dynamic allocation — CUDA device function compatible.
    """
    if (x < tx[kx] or x > tx[len(tx) - kx - 1] or y < ty[ky] or y > ty[len(ty) - ky - 1]):
        return nb.float32(0.0)
    ix = find_span(tx, kx, x)
    iy = find_span(ty, ky, y)
    base_x = ix - kx
    base_y = iy - ky

    # Evaluate the 1D spline in x for each of the ky+1 rows,
    # collect into row values (again, fixed-size scalars).
    r0 = nb.float32(0.0)
    r1 = nb.float32(0.0)
    r2 = nb.float32(0.0)
    r3 = nb.float32(0.0)
    r4 = nb.float32(0.0)
    r5 = nb.float32(0.0)

    for row in range(ky + 1):
        # Inline de Boor in x for row `base_y + row`
        d0 = nb.float32(0.0)
        d1 = nb.float32(0.0)
        d2 = nb.float32(0.0)
        d3 = nb.float32(0.0)
        d4 = nb.float32(0.0)
        d5 = nb.float32(0.0)
        if kx >= 0: d0 = coeffs[base_y + row, base_x + 0]
        if kx >= 1: d1 = coeffs[base_y + row, base_x + 1]
        if kx >= 2: d2 = coeffs[base_y + row, base_x + 2]
        if kx >= 3: d3 = coeffs[base_y + row, base_x + 3]
        if kx >= 4: d4 = coeffs[base_y + row, base_x + 4]
        if kx >= 5: d5 = coeffs[base_y + row, base_x + 5]

        for r in range(1, kx + 1):
            for j in range(kx, r - 1, -1):
                tj  = tx[ix - kx + j]
                tjr = tx[ix + 1 + j - r]
                denom = tjr - tj
                alpha = (x - tj) / denom if denom != 0.0 else nb.float32(0.0)
                if j == 5: d5 = (nb.float32(1.0) - alpha) * d4 + alpha * d5
                if j == 4: d4 = (nb.float32(1.0) - alpha) * d3 + alpha * d4
                if j == 3: d3 = (nb.float32(1.0) - alpha) * d2 + alpha * d3
                if j == 2: d2 = (nb.float32(1.0) - alpha) * d1 + alpha * d2
                if j == 1: d1 = (nb.float32(1.0) - alpha) * d0 + alpha * d1

        rx = d0 if kx == 0 else (d1 if kx == 1 else (d2 if kx == 2 else (d3 if kx == 3 else (d4 if kx == 4 else d5))))
        if row == 0: r0 = rx
        if row == 1: r1 = rx
        if row == 2: r2 = rx
        if row == 3: r3 = rx
        if row == 4: r4 = rx
        if row == 5: r5 = rx

    # Now de Boor in y over the row values
    for r in range(1, ky + 1):
        for j in range(ky, r - 1, -1):
            tj  = ty[iy - ky + j]
            tjr = ty[iy + 1 + j - r]
            denom = tjr - tj
            alpha = (y - tj) / denom if denom != 0.0 else nb.float32(0.0)
            if j == 5: r5 = (nb.float32(1.0) - alpha) * r4 + alpha * r5
            if j == 4: r4 = (nb.float32(1.0) - alpha) * r3 + alpha * r4
            if j == 3: r3 = (nb.float32(1.0) - alpha) * r2 + alpha * r3
            if j == 2: r2 = (nb.float32(1.0) - alpha) * r1 + alpha * r2
            if j == 1: r1 = (nb.float32(1.0) - alpha) * r0 + alpha * r1

    return r0 if ky == 0 else (r1 if ky == 1 else (r2 if ky == 2 else (r3 if ky == 3 else (r4 if ky == 4 else r5))))


@nb.njit(nogil=True, inline="always")
def bspline3d(tx, ty, tz, coeffs, kx, ky, kz, x, y, z):
    if (x < tx[kx] or x > tx[len(tx) - kx - 1] or y < ty[ky] or y > ty[len(ty) - ky - 1] or z < tz[kz] or z > tz[len(tz) - kz - 1]):
        return nb.float32(0.0)
    ix = find_span(tx, kx, x)
    iy = find_span(ty, ky, y)
    iz = find_span(tz, kz, z)
    base_x = ix - kx
    base_y = iy - ky
    base_z = iz - kz

    p0 = nb.float32(0.0)
    p1 = nb.float32(0.0)
    p2 = nb.float32(0.0)
    p3 = nb.float32(0.0)
    p4 = nb.float32(0.0)
    p5 = nb.float32(0.0)

    for plane in range(kz + 1):
        r0 = nb.float32(0.0)
        r1 = nb.float32(0.0)
        r2 = nb.float32(0.0)
        r3 = nb.float32(0.0)
        r4 = nb.float32(0.0)
        r5 = nb.float32(0.0)

        for row in range(ky + 1):
            d0 = nb.float32(0.0)
            d1 = nb.float32(0.0)
            d2 = nb.float32(0.0)
            d3 = nb.float32(0.0)
            d4 = nb.float32(0.0)
            d5 = nb.float32(0.0)
            if kx >= 0: d0 = coeffs[base_z + plane, base_y + row, base_x + 0]
            if kx >= 1: d1 = coeffs[base_z + plane, base_y + row, base_x + 1]
            if kx >= 2: d2 = coeffs[base_z + plane, base_y + row, base_x + 2]
            if kx >= 3: d3 = coeffs[base_z + plane, base_y + row, base_x + 3]
            if kx >= 4: d4 = coeffs[base_z + plane, base_y + row, base_x + 4]
            if kx >= 5: d5 = coeffs[base_z + plane, base_y + row, base_x + 5]

            for r in range(1, kx + 1):
                for j in range(kx, r - 1, -1):
                    tj  = tx[ix - kx + j]
                    tjr = tx[ix + 1 + j - r]
                    denom = tjr - tj
                    alpha = (x - tj) / denom if denom != 0.0 else nb.float32(0.0)
                    if j == 5: d5 = (nb.float32(1.0) - alpha) * d4 + alpha * d5
                    if j == 4: d4 = (nb.float32(1.0) - alpha) * d3 + alpha * d4
                    if j == 3: d3 = (nb.float32(1.0) - alpha) * d2 + alpha * d3
                    if j == 2: d2 = (nb.float32(1.0) - alpha) * d1 + alpha * d2
                    if j == 1: d1 = (nb.float32(1.0) - alpha) * d0 + alpha * d1

            rx = d0 if kx == 0 else (d1 if kx == 1 else (d2 if kx == 2 else (d3 if kx == 3 else (d4 if kx == 4 else d5))))
            if row == 0: r0 = rx
            if row == 1: r1 = rx
            if row == 2: r2 = rx
            if row == 3: r3 = rx
            if row == 4: r4 = rx
            if row == 5: r5 = rx

        for r in range(1, ky + 1):
            for j in range(ky, r - 1, -1):
                tj  = ty[iy - ky + j]
                tjr = ty[iy + 1 + j - r]
                denom = tjr - tj
                alpha = (y - tj) / denom if denom != 0.0 else nb.float32(0.0)
                if j == 5: r5 = (nb.float32(1.0) - alpha) * r4 + alpha * r5
                if j == 4: r4 = (nb.float32(1.0) - alpha) * r3 + alpha * r4
                if j == 3: r3 = (nb.float32(1.0) - alpha) * r2 + alpha * r3
                if j == 2: r2 = (nb.float32(1.0) - alpha) * r1 + alpha * r2
                if j == 1: r1 = (nb.float32(1.0) - alpha) * r0 + alpha * r1

        ry = r0 if ky == 0 else (r1 if ky == 1 else (r2 if ky == 2 else (r3 if ky == 3 else (r4 if ky == 4 else r5))))
        if plane == 0: p0 = ry
        if plane == 1: p1 = ry
        if plane == 2: p2 = ry
        if plane == 3: p3 = ry
        if plane == 4: p4 = ry
        if plane == 5: p5 = ry

    for r in range(1, kz + 1):
        for j in range(kz, r - 1, -1):
            tj  = tz[iz - kz + j]
            tjr = tz[iz + 1 + j - r]
            denom = tjr - tj
            alpha = (z - tj) / denom if denom != 0.0 else nb.float32(0.0)
            if j == 5: p5 = (nb.float32(1.0) - alpha) * p4 + alpha * p5
            if j == 4: p4 = (nb.float32(1.0) - alpha) * p3 + alpha * p4
            if j == 3: p3 = (nb.float32(1.0) - alpha) * p2 + alpha * p3
            if j == 2: p2 = (nb.float32(1.0) - alpha) * p1 + alpha * p2
            if j == 1: p1 = (nb.float32(1.0) - alpha) * p0 + alpha * p1

    return p0 if kz == 0 else (p1 if kz == 1 else (p2 if kz == 2 else (p3 if kz == 3 else (p4 if kz == 4 else p5))))