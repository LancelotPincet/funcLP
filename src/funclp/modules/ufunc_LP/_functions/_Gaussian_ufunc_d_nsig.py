
from ._Gaussian_cpukernel_function import _Gaussian_cpukernel_function as kernel
from funclp import ufunc
import math
@ufunc(data=[], constants=[], fastmath=False)
def d_nsig(x, /, mu, sig, amp, offset, pix, nsig):
    eps = 1e-3 * max(1.0, abs(nsig))
    f_plus = kernel(x, mu, sig, amp, offset, pix, nsig + eps)
    f_minus = kernel(x, mu, sig, amp, offset, pix, nsig - eps)
    if math.isfinite(f_plus) and math.isfinite(f_minus):
        return (f_plus - f_minus) / (2.0 * eps)
    f_x = kernel(x, mu, sig, amp, offset, pix, nsig)
    if math.isfinite(f_plus) and math.isfinite(f_x):
        return (f_plus - f_x) / eps
    if math.isfinite(f_minus) and math.isfinite(f_x):
        return (f_x - f_minus) / eps
    return math.nan
