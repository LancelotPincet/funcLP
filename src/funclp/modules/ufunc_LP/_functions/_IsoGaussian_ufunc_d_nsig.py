
from ._IsoGaussian_cpukernel_function import _IsoGaussian_cpukernel_function as kernel
from funclp import ufunc
import math
@ufunc(data=[], constants=[], fastmath=False)
def d_nsig(x, y, /, mux, muy, sig, amp, offset, pixx, pixy, nsig):
    eps = 1e-4
    f_plus = kernel(x, y, mux, muy, sig, amp, offset, pixx, pixy, nsig + eps)
    f_minus = kernel(x, y, mux, muy, sig, amp, offset, pixx, pixy, nsig - eps)
    if math.isfinite(f_plus) and math.isfinite(f_minus):
        return (f_plus - f_minus) / (2.0 * eps)
    f_x = kernel(x, y, mux, muy, sig, amp, offset, pixx, pixy, nsig)
    if math.isfinite(f_plus) and math.isfinite(f_x):
        return (f_plus - f_x) / eps
    if math.isfinite(f_minus) and math.isfinite(f_x):
        return (f_x - f_minus) / eps
    return math.nan
