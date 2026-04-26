
from ._Rectangle_cpukernel_function import _Rectangle_cpukernel_function as kernel
from funclp import Parameter, ufunc
import math
@ufunc(variables=['x', 'y'], data=[], parameters=[Parameter('l', 1.0), Parameter('ratio', 1.0), Parameter('mux', 0.0), Parameter('muy', 0.0), Parameter('amp', 1.0), Parameter('offset', 0.0), Parameter('theta', 0.0)], constants=[], fastmath=False)
def d_theta(x, y, /, l, ratio, mux, muy, amp, offset, theta):
    eps = 1e-3 * max(1.0, abs(theta))
    f_plus = kernel(x, y, l, ratio, mux, muy, amp, offset, theta + eps)
    f_minus = kernel(x, y, l, ratio, mux, muy, amp, offset, theta - eps)
    if math.isfinite(f_plus) and math.isfinite(f_minus):
        return (f_plus - f_minus) / (2.0 * eps)
    f_x = kernel(x, y, l, ratio, mux, muy, amp, offset, theta)
    if math.isfinite(f_plus) and math.isfinite(f_x):
        return (f_plus - f_x) / eps
    if math.isfinite(f_minus) and math.isfinite(f_x):
        return (f_x - f_minus) / eps
    return math.nan
