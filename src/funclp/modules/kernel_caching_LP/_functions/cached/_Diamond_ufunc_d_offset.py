
from ._Diamond_cpukernel_function import _Diamond_cpukernel_function as kernel
from funclp import Parameter, ufunc
import math
@ufunc(variables=['x', 'y'], data=[], parameters=[Parameter('d', 1.0), Parameter('mux', 0.0), Parameter('muy', 0.0), Parameter('amp', 1.0), Parameter('offset', 0.0)], constants=[], fastmath=False)
def d_offset(x, y, /, d, mux, muy, amp, offset):
    eps = 1e-3 * max(1.0, abs(offset))
    f_plus = kernel(x, y, d, mux, muy, amp, offset + eps)
    f_minus = kernel(x, y, d, mux, muy, amp, offset - eps)
    if math.isfinite(f_plus) and math.isfinite(f_minus):
        return (f_plus - f_minus) / (2.0 * eps)
    f_x = kernel(x, y, d, mux, muy, amp, offset)
    if math.isfinite(f_plus) and math.isfinite(f_x):
        return (f_plus - f_x) / eps
    if math.isfinite(f_minus) and math.isfinite(f_x):
        return (f_x - f_minus) / eps
    return math.nan
