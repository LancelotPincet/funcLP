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


@nb.njit(nogil=True, inline="always", fastmath=True, cache=True)
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


@nb.njit(nogil=True, inline="always", fastmath=True, cache=True)
def bspline1d(tx, coeffs, kx, x):
    """
    Evaluate a 1D B-spline at x.
    De Boor's algorithm, fully inlined — no dynamic allocation,
    CUDA device function compatible.
    Supports kx up to 5.
    """
    kx = nb.int32(kx)
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


@nb.njit(nogil=True, inline="always", fastmath=True, cache=True)
def bspline2d(tx, ty, coeffs, kx, ky, x, y):
    """
    Evaluate a 2D tensor-product B-spline at (x, y).
    Reduces to ky+1 calls of 1D de Boor along x, then one along y.
    No dynamic allocation — CUDA device function compatible.
    """
    kx = nb.int32(kx)
    ky = nb.int32(ky)
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


@nb.njit(nogil=True, inline="always", fastmath=True, cache=True)
def bspline3d(tx, ty, tz, coeffs, kx, ky, kz, x, y, z):
    kx = nb.int32(kx)
    ky = nb.int32(ky)
    kz = nb.int32(kz)
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


@nb.njit(nogil=True, inline="always", fastmath=True, cache=True)
def bspline1d_dx(t, coeffs, k, x):
    """
    Evaluate d/dx of a 1D B-spline at x.
    Supports k up to 5.
    """
    k = nb.int32(k)

    if k <= 0:
        return nb.float32(0.0)

    if x < t[k] or x > t[len(t) - k - 1]:
        return nb.float32(0.0)

    ix = find_span(t, k, x)
    base = ix - k

    # Derivative control points for degree k-1 spline
    d0 = nb.float32(0.0)
    d1 = nb.float32(0.0)
    d2 = nb.float32(0.0)
    d3 = nb.float32(0.0)
    d4 = nb.float32(0.0)

    for j in range(k):
        denom = t[base + j + k + 1] - t[base + j + 1]
        v = nb.float32(0.0)
        if denom != 0.0:
            v = nb.float32(k) * (coeffs[base + j + 1] - coeffs[base + j]) / denom

        if j == 0:
            d0 = v
        elif j == 1:
            d1 = v
        elif j == 2:
            d2 = v
        elif j == 3:
            d3 = v
        elif j == 4:
            d4 = v

    # De Boor on derivative spline of degree k-1
    deg = k - 1

    for r in range(1, deg + 1):
        for j in range(deg, r - 1, -1):
            tj = t[ix - deg + j]
            tjr = t[ix + 1 + j - r]
            denom = tjr - tj
            alpha = (x - tj) / denom if denom != 0.0 else nb.float32(0.0)

            if j == 4:
                d4 = (nb.float32(1.0) - alpha) * d3 + alpha * d4
            if j == 3:
                d3 = (nb.float32(1.0) - alpha) * d2 + alpha * d3
            if j == 2:
                d2 = (nb.float32(1.0) - alpha) * d1 + alpha * d2
            if j == 1:
                d1 = (nb.float32(1.0) - alpha) * d0 + alpha * d1

    if deg == 0:
        return d0
    elif deg == 1:
        return d1
    elif deg == 2:
        return d2
    elif deg == 3:
        return d3
    else:
        return d4


@nb.njit(nogil=True, inline="always", fastmath=True, cache=True)
def bspline2d_dx(tx, ty, coeffs, kx, ky, x, y):
    """
    Evaluate d/dx of a 2D tensor-product B-spline at (x, y).
    """
    kx = nb.int32(kx)
    ky = nb.int32(ky)

    if (x < tx[kx] or x > tx[len(tx) - kx - 1] or
        y < ty[ky] or y > ty[len(ty) - ky - 1]):
        return nb.float32(0.0)

    if kx <= 0:
        return nb.float32(0.0)

    ix = find_span(tx, kx, x)
    iy = find_span(ty, ky, y)
    base_x = ix - kx
    base_y = iy - ky

    # First evaluate d/dx along x for each active row
    r0 = nb.float32(0.0)
    r1 = nb.float32(0.0)
    r2 = nb.float32(0.0)
    r3 = nb.float32(0.0)
    r4 = nb.float32(0.0)
    r5 = nb.float32(0.0)

    for row in range(ky + 1):
        # derivative spline along x has degree kx-1 and kx control values
        d0 = nb.float32(0.0)
        d1 = nb.float32(0.0)
        d2 = nb.float32(0.0)
        d3 = nb.float32(0.0)
        d4 = nb.float32(0.0)

        # Build derivative control points:
        # c'_i = k * (c_{i+1} - c_i) / (t_{i+k+1} - t_{i+1})
        for j in range(kx):
            denom = tx[base_x + j + kx + 1] - tx[base_x + j + 1]
            v = nb.float32(0.0)
            if denom != 0.0:
                v = nb.float32(kx) * (
                    coeffs[base_y + row, base_x + j + 1] -
                    coeffs[base_y + row, base_x + j]
                ) / denom

            if j == 0: d0 = v
            elif j == 1: d1 = v
            elif j == 2: d2 = v
            elif j == 3: d3 = v
            elif j == 4: d4 = v

        # De Boor on derivative spline of degree kx-1
        deg = kx - 1
        ixd = ix  # same span location works on the derivative spline support
        for r in range(1, deg + 1):
            for j in range(deg, r - 1, -1):
                tj = tx[ixd - deg + j]
                tjr = tx[ixd + 1 + j - r]
                denom = tjr - tj
                alpha = (x - tj) / denom if denom != 0.0 else nb.float32(0.0)

                if j == 4: d4 = (nb.float32(1.0) - alpha) * d3 + alpha * d4
                if j == 3: d3 = (nb.float32(1.0) - alpha) * d2 + alpha * d3
                if j == 2: d2 = (nb.float32(1.0) - alpha) * d1 + alpha * d2
                if j == 1: d1 = (nb.float32(1.0) - alpha) * d0 + alpha * d1

        rx = d0 if deg == 0 else (d1 if deg == 1 else (d2 if deg == 2 else (d3 if deg == 3 else d4)))

        if row == 0: r0 = rx
        if row == 1: r1 = rx
        if row == 2: r2 = rx
        if row == 3: r3 = rx
        if row == 4: r4 = rx
        if row == 5: r5 = rx

    # Then regular spline evaluation in y
    for r in range(1, ky + 1):
        for j in range(ky, r - 1, -1):
            tj = ty[iy - ky + j]
            tjr = ty[iy + 1 + j - r]
            denom = tjr - tj
            alpha = (y - tj) / denom if denom != 0.0 else nb.float32(0.0)

            if j == 5: r5 = (nb.float32(1.0) - alpha) * r4 + alpha * r5
            if j == 4: r4 = (nb.float32(1.0) - alpha) * r3 + alpha * r4
            if j == 3: r3 = (nb.float32(1.0) - alpha) * r2 + alpha * r3
            if j == 2: r2 = (nb.float32(1.0) - alpha) * r1 + alpha * r2
            if j == 1: r1 = (nb.float32(1.0) - alpha) * r0 + alpha * r1

    return r0 if ky == 0 else (r1 if ky == 1 else (r2 if ky == 2 else (r3 if ky == 3 else (r4 if ky == 4 else r5))))


@nb.njit(nogil=True, inline="always", fastmath=True, cache=True)
def bspline2d_dy(tx, ty, coeffs, kx, ky, x, y):
    """
    Evaluate d/dy of a 2D tensor-product B-spline at (x, y).
    """
    kx = nb.int32(kx)
    ky = nb.int32(ky)

    if (x < tx[kx] or x > tx[len(tx) - kx - 1] or
        y < ty[ky] or y > ty[len(ty) - ky - 1]):
        return nb.float32(0.0)

    if ky <= 0:
        return nb.float32(0.0)

    ix = find_span(tx, kx, x)
    iy = find_span(ty, ky, y)
    base_x = ix - kx
    base_y = iy - ky

    # First evaluate regular spline along x for each active row
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

        if kx >= 0: d0 = coeffs[base_y + row, base_x + 0]
        if kx >= 1: d1 = coeffs[base_y + row, base_x + 1]
        if kx >= 2: d2 = coeffs[base_y + row, base_x + 2]
        if kx >= 3: d3 = coeffs[base_y + row, base_x + 3]
        if kx >= 4: d4 = coeffs[base_y + row, base_x + 4]
        if kx >= 5: d5 = coeffs[base_y + row, base_x + 5]

        for r in range(1, kx + 1):
            for j in range(kx, r - 1, -1):
                tj = tx[ix - kx + j]
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

    # Build derivative control points along y:
    # r'_i = ky * (r_{i+1} - r_i) / (t_{i+ky+1} - t_{i+1})
    q0 = nb.float32(0.0)
    q1 = nb.float32(0.0)
    q2 = nb.float32(0.0)
    q3 = nb.float32(0.0)
    q4 = nb.float32(0.0)

    for j in range(ky):
        a = r0 if j == 0 else (r1 if j == 1 else (r2 if j == 2 else (r3 if j == 3 else r4)))
        b = r1 if j == 0 else (r2 if j == 1 else (r3 if j == 2 else (r4 if j == 3 else r5)))
        denom = ty[base_y + j + ky + 1] - ty[base_y + j + 1]
        v = nb.float32(0.0)
        if denom != 0.0:
            v = nb.float32(ky) * (b - a) / denom

        if j == 0: q0 = v
        elif j == 1: q1 = v
        elif j == 2: q2 = v
        elif j == 3: q3 = v
        elif j == 4: q4 = v

    # De Boor on derivative spline in y of degree ky-1
    deg = ky - 1
    iyd = iy
    for r in range(1, deg + 1):
        for j in range(deg, r - 1, -1):
            tj = ty[iyd - deg + j]
            tjr = ty[iyd + 1 + j - r]
            denom = tjr - tj
            alpha = (y - tj) / denom if denom != 0.0 else nb.float32(0.0)

            if j == 4: q4 = (nb.float32(1.0) - alpha) * q3 + alpha * q4
            if j == 3: q3 = (nb.float32(1.0) - alpha) * q2 + alpha * q3
            if j == 2: q2 = (nb.float32(1.0) - alpha) * q1 + alpha * q2
            if j == 1: q1 = (nb.float32(1.0) - alpha) * q0 + alpha * q1

    return q0 if deg == 0 else (q1 if deg == 1 else (q2 if deg == 2 else (q3 if deg == 3 else q4)))


@nb.njit(nogil=True, inline="always", fastmath=True, cache=True)
def bspline3d_dx(tx, ty, tz, coeffs, kx, ky, kz, x, y, z):
    kx = nb.int32(kx)
    ky = nb.int32(ky)
    kz = nb.int32(kz)

    if kx <= 0:
        return nb.float32(0.0)

    if (x < tx[kx] or x > tx[len(tx) - kx - 1] or
        y < ty[ky] or y > ty[len(ty) - ky - 1] or
        z < tz[kz] or z > tz[len(tz) - kz - 1]):
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
            # derivative control points along x
            d0 = nb.float32(0.0)
            d1 = nb.float32(0.0)
            d2 = nb.float32(0.0)
            d3 = nb.float32(0.0)
            d4 = nb.float32(0.0)

            for j in range(kx):
                denom = tx[base_x + j + kx + 1] - tx[base_x + j + 1]
                v = nb.float32(0.0)
                if denom != 0.0:
                    v = nb.float32(kx) * (
                        coeffs[base_z + plane, base_y + row, base_x + j + 1] -
                        coeffs[base_z + plane, base_y + row, base_x + j]
                    ) / denom

                if j == 0: d0 = v
                elif j == 1: d1 = v
                elif j == 2: d2 = v
                elif j == 3: d3 = v
                elif j == 4: d4 = v

            degx = kx - 1
            for r in range(1, degx + 1):
                for j in range(degx, r - 1, -1):
                    tj = tx[ix - degx + j]
                    tjr = tx[ix + 1 + j - r]
                    denom = tjr - tj
                    alpha = (x - tj) / denom if denom != 0.0 else nb.float32(0.0)

                    if j == 4: d4 = (nb.float32(1.0) - alpha) * d3 + alpha * d4
                    if j == 3: d3 = (nb.float32(1.0) - alpha) * d2 + alpha * d3
                    if j == 2: d2 = (nb.float32(1.0) - alpha) * d1 + alpha * d2
                    if j == 1: d1 = (nb.float32(1.0) - alpha) * d0 + alpha * d1

            rx = d0 if degx == 0 else (d1 if degx == 1 else (d2 if degx == 2 else (d3 if degx == 3 else d4)))

            if row == 0: r0 = rx
            if row == 1: r1 = rx
            if row == 2: r2 = rx
            if row == 3: r3 = rx
            if row == 4: r4 = rx
            if row == 5: r5 = rx

        # regular spline in y
        for r in range(1, ky + 1):
            for j in range(ky, r - 1, -1):
                tj = ty[iy - ky + j]
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

    # regular spline in z
    for r in range(1, kz + 1):
        for j in range(kz, r - 1, -1):
            tj = tz[iz - kz + j]
            tjr = tz[iz + 1 + j - r]
            denom = tjr - tj
            alpha = (z - tj) / denom if denom != 0.0 else nb.float32(0.0)

            if j == 5: p5 = (nb.float32(1.0) - alpha) * p4 + alpha * p5
            if j == 4: p4 = (nb.float32(1.0) - alpha) * p3 + alpha * p4
            if j == 3: p3 = (nb.float32(1.0) - alpha) * p2 + alpha * p3
            if j == 2: p2 = (nb.float32(1.0) - alpha) * p1 + alpha * p2
            if j == 1: p1 = (nb.float32(1.0) - alpha) * p0 + alpha * p1

    return p0 if kz == 0 else (p1 if kz == 1 else (p2 if kz == 2 else (p3 if kz == 3 else (p4 if kz == 4 else p5))))


@nb.njit(nogil=True, inline="always", fastmath=True, cache=True)
def bspline3d_dy(tx, ty, tz, coeffs, kx, ky, kz, x, y, z):
    kx = nb.int32(kx)
    ky = nb.int32(ky)
    kz = nb.int32(kz)

    if ky <= 0:
        return nb.float32(0.0)

    if (x < tx[kx] or x > tx[len(tx) - kx - 1] or
        y < ty[ky] or y > ty[len(ty) - ky - 1] or
        z < tz[kz] or z > tz[len(tz) - kz - 1]):
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

        # regular spline in x for each row
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
                    tj = tx[ix - kx + j]
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

        # derivative control points in y
        q0 = nb.float32(0.0)
        q1 = nb.float32(0.0)
        q2 = nb.float32(0.0)
        q3 = nb.float32(0.0)
        q4 = nb.float32(0.0)

        for j in range(ky):
            a = r0 if j == 0 else (r1 if j == 1 else (r2 if j == 2 else (r3 if j == 3 else r4)))
            b = r1 if j == 0 else (r2 if j == 1 else (r3 if j == 2 else (r4 if j == 3 else r5)))
            denom = ty[base_y + j + ky + 1] - ty[base_y + j + 1]
            v = nb.float32(0.0)
            if denom != 0.0:
                v = nb.float32(ky) * (b - a) / denom

            if j == 0: q0 = v
            elif j == 1: q1 = v
            elif j == 2: q2 = v
            elif j == 3: q3 = v
            elif j == 4: q4 = v

        degy = ky - 1
        for r in range(1, degy + 1):
            for j in range(degy, r - 1, -1):
                tj = ty[iy - degy + j]
                tjr = ty[iy + 1 + j - r]
                denom = tjr - tj
                alpha = (y - tj) / denom if denom != 0.0 else nb.float32(0.0)

                if j == 4: q4 = (nb.float32(1.0) - alpha) * q3 + alpha * q4
                if j == 3: q3 = (nb.float32(1.0) - alpha) * q2 + alpha * q3
                if j == 2: q2 = (nb.float32(1.0) - alpha) * q1 + alpha * q2
                if j == 1: q1 = (nb.float32(1.0) - alpha) * q0 + alpha * q1

        ry = q0 if degy == 0 else (q1 if degy == 1 else (q2 if degy == 2 else (q3 if degy == 3 else q4)))

        if plane == 0: p0 = ry
        if plane == 1: p1 = ry
        if plane == 2: p2 = ry
        if plane == 3: p3 = ry
        if plane == 4: p4 = ry
        if plane == 5: p5 = ry

    # regular spline in z
    for r in range(1, kz + 1):
        for j in range(kz, r - 1, -1):
            tj = tz[iz - kz + j]
            tjr = tz[iz + 1 + j - r]
            denom = tjr - tj
            alpha = (z - tj) / denom if denom != 0.0 else nb.float32(0.0)

            if j == 5: p5 = (nb.float32(1.0) - alpha) * p4 + alpha * p5
            if j == 4: p4 = (nb.float32(1.0) - alpha) * p3 + alpha * p4
            if j == 3: p3 = (nb.float32(1.0) - alpha) * p2 + alpha * p3
            if j == 2: p2 = (nb.float32(1.0) - alpha) * p1 + alpha * p2
            if j == 1: p1 = (nb.float32(1.0) - alpha) * p0 + alpha * p1

    return p0 if kz == 0 else (p1 if kz == 1 else (p2 if kz == 2 else (p3 if kz == 3 else (p4 if kz == 4 else p5))))


@nb.njit(nogil=True, inline="always", fastmath=True, cache=True)
def bspline3d_dz(tx, ty, tz, coeffs, kx, ky, kz, x, y, z):
    kx = nb.int32(kx)
    ky = nb.int32(ky)
    kz = nb.int32(kz)

    if kz <= 0:
        return nb.float32(0.0)

    if (x < tx[kx] or x > tx[len(tx) - kx - 1] or
        y < ty[ky] or y > ty[len(ty) - ky - 1] or
        z < tz[kz] or z > tz[len(tz) - kz - 1]):
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

    # first compute regular x/y spline on each active z-plane
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
                    tj = tx[ix - kx + j]
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
                tj = ty[iy - ky + j]
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

    # derivative control points in z
    q0 = nb.float32(0.0)
    q1 = nb.float32(0.0)
    q2 = nb.float32(0.0)
    q3 = nb.float32(0.0)
    q4 = nb.float32(0.0)

    for j in range(kz):
        a = p0 if j == 0 else (p1 if j == 1 else (p2 if j == 2 else (p3 if j == 3 else p4)))
        b = p1 if j == 0 else (p2 if j == 1 else (p3 if j == 2 else (p4 if j == 3 else p5)))
        denom = tz[base_z + j + kz + 1] - tz[base_z + j + 1]
        v = nb.float32(0.0)
        if denom != 0.0:
            v = nb.float32(kz) * (b - a) / denom

        if j == 0: q0 = v
        elif j == 1: q1 = v
        elif j == 2: q2 = v
        elif j == 3: q3 = v
        elif j == 4: q4 = v

    degz = kz - 1
    for r in range(1, degz + 1):
        for j in range(degz, r - 1, -1):
            tj = tz[iz - degz + j]
            tjr = tz[iz + 1 + j - r]
            denom = tjr - tj
            alpha = (z - tj) / denom if denom != 0.0 else nb.float32(0.0)

            if j == 4: q4 = (nb.float32(1.0) - alpha) * q3 + alpha * q4
            if j == 3: q3 = (nb.float32(1.0) - alpha) * q2 + alpha * q3
            if j == 2: q2 = (nb.float32(1.0) - alpha) * q1 + alpha * q2
            if j == 1: q1 = (nb.float32(1.0) - alpha) * q0 + alpha * q1

    return q0 if degz == 0 else (q1 if degz == 1 else (q2 if degz == 2 else (q3 if degz == 3 else q4)))