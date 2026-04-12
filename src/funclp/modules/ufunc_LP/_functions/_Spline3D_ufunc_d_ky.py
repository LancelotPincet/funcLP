
from ._Spline3D_cpukernel_function import _Spline3D_cpukernel_function as kernel
from funclp import ufunc
import math
@ufunc(data=[], constants=['tx', 'ty', 'tz', 'coeffs'], fastmath=False)
def d_ky(x, y, z, /, mux, muy, muz, amp, offset, kx, ky, kz, tx, ty, tz, coeffs):
    eps = 1e-3 * max(1.0, abs(ky))
    f_plus = kernel(x, y, z, mux, muy, muz, amp, offset, kx, ky + eps, kz, tx, ty, tz, coeffs)
    f_minus = kernel(x, y, z, mux, muy, muz, amp, offset, kx, ky - eps, kz, tx, ty, tz, coeffs)
    if math.isfinite(f_plus) and math.isfinite(f_minus):
        return (f_plus - f_minus) / (2.0 * eps)
    f_x = kernel(x, y, z, mux, muy, muz, amp, offset, kx, ky, kz, tx, ty, tz, coeffs)
    if math.isfinite(f_plus) and math.isfinite(f_x):
        return (f_plus - f_x) / eps
    if math.isfinite(f_minus) and math.isfinite(f_x):
        return (f_x - f_minus) / eps
    return math.nan
