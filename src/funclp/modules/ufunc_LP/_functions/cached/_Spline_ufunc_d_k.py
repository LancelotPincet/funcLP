
from ._Spline_cpukernel_function import _Spline_cpukernel_function as kernel
from funclp import Parameter, ufunc
import math
@ufunc(variables=['x'], data=[], parameters=[Parameter('mu', 0.0), Parameter('amp', 1.0), Parameter('offset', 0.0), Parameter('k', 3)], constants=['t', 'coeffs'], fastmath=False)
def d_k(x, /, mu, amp, offset, k, t, coeffs):
    eps = 1e-3 * max(1.0, abs(k))
    f_plus = kernel(x, mu, amp, offset, k + eps, t, coeffs)
    f_minus = kernel(x, mu, amp, offset, k - eps, t, coeffs)
    if math.isfinite(f_plus) and math.isfinite(f_minus):
        return (f_plus - f_minus) / (2.0 * eps)
    f_x = kernel(x, mu, amp, offset, k, t, coeffs)
    if math.isfinite(f_plus) and math.isfinite(f_x):
        return (f_plus - f_x) / eps
    if math.isfinite(f_minus) and math.isfinite(f_x):
        return (f_x - f_minus) / eps
    return math.nan
