
from ._Airy_cpukernel_function import _Airy_cpukernel_function as kernel
from funclp import Parameter, ufunc
import math
@ufunc(variables=['x'], data=[], parameters=[Parameter('mu', 0.0), Parameter('amp', 1.0), Parameter('offset', 0.0), Parameter('wl', 550.0), Parameter('NA', 1.5), Parameter('tol', 1.0)], constants=[], fastmath=False)
def d_mu(x, /, mu, amp, offset, wl, NA, tol):
    eps = 1e-3 * max(1.0, abs(mu))
    f_plus = kernel(x, mu + eps, amp, offset, wl, NA, tol)
    f_minus = kernel(x, mu - eps, amp, offset, wl, NA, tol)
    if math.isfinite(f_plus) and math.isfinite(f_minus):
        return (f_plus - f_minus) / (2.0 * eps)
    f_x = kernel(x, mu, amp, offset, wl, NA, tol)
    if math.isfinite(f_plus) and math.isfinite(f_x):
        return (f_plus - f_x) / eps
    if math.isfinite(f_minus) and math.isfinite(f_x):
        return (f_x - f_minus) / eps
    return math.nan
