
from ._IsoGaussian_cpukernel_function import _IsoGaussian_cpukernel_function as kernel
from funclp import ufunc
import numpy as np
@ufunc(data=[], constants=[], fastmath=False)
def d_nsig(x, y, /, mux, muy, sig, amp, offset, pixx, pixy, nsig, eps=1e-4):
    f_plus = kernel(x, y, mux, muy, sig, amp, offset, pixx, pixy, nsig + eps)
    f_minus = kernel(x, y, mux, muy, sig, amp, offset, pixx, pixy, nsig - eps)
    if np.isfinite(f_plus) and np.isfinite(f_minus):
        return (f_plus - f_minus) / (2.0 * eps)
    f_x = kernel(x, y, mux, muy, sig, amp, offset, pixx, pixy, nsig)
    if np.isfinite(f_plus) and np.isfinite(f_x):
        return (f_plus - f_x) / eps
    if np.isfinite(f_minus) and np.isfinite(f_x):
        return (f_x - f_minus) / eps
    return np.nan
