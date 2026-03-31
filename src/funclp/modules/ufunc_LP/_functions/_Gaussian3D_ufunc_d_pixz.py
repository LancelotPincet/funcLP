
from ._Gaussian3D_cpukernel_function import _Gaussian3D_cpukernel_function as kernel
from funclp import ufunc
import numpy as np
@ufunc(data=[], constants=[], fastmath=False)
def d_pixz(x, y, z, /, mux, muy, muz, sigx, sigy, sigz, amp, offset, pixx, pixy, pixz, nsig, theta, phi, eps=1e-4):
    f_plus = kernel(x, y, z, mux, muy, muz, sigx, sigy, sigz, amp, offset, pixx, pixy, pixz + eps, nsig, theta, phi)
    f_minus = kernel(x, y, z, mux, muy, muz, sigx, sigy, sigz, amp, offset, pixx, pixy, pixz - eps, nsig, theta, phi)
    if np.isfinite(f_plus) and np.isfinite(f_minus):
        return (f_plus - f_minus) / (2.0 * eps)
    f_x = kernel(x, y, z, mux, muy, muz, sigx, sigy, sigz, amp, offset, pixx, pixy, pixz, nsig, theta, phi)
    if np.isfinite(f_plus) and np.isfinite(f_x):
        return (f_plus - f_x) / eps
    if np.isfinite(f_minus) and np.isfinite(f_x):
        return (f_x - f_minus) / eps
    return np.nan
