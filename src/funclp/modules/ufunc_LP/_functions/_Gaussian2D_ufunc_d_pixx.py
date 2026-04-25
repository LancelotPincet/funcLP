
from ._Gaussian2D_cpukernel_function import _Gaussian2D_cpukernel_function as kernel
from funclp import Parameter, ufunc
import math
@ufunc(variables=['x', 'y'], data=[], parameters=[Parameter('mux', 0.0), Parameter('muy', 0.0), Parameter('sigx', 0.15915494309189535), Parameter('sigy', 0.15915494309189535), Parameter('amp', 1.0), Parameter('offset', 0.0), Parameter('pixx', -1.0), Parameter('pixy', -1.0), Parameter('nsig', -1.0), Parameter('theta', 0.0)], constants=[], fastmath=False)
def d_pixx(x, y, /, mux, muy, sigx, sigy, amp, offset, pixx, pixy, nsig, theta):
    eps = 1e-3 * max(1.0, abs(pixx))
    f_plus = kernel(x, y, mux, muy, sigx, sigy, amp, offset, pixx + eps, pixy, nsig, theta)
    f_minus = kernel(x, y, mux, muy, sigx, sigy, amp, offset, pixx - eps, pixy, nsig, theta)
    if math.isfinite(f_plus) and math.isfinite(f_minus):
        return (f_plus - f_minus) / (2.0 * eps)
    f_x = kernel(x, y, mux, muy, sigx, sigy, amp, offset, pixx, pixy, nsig, theta)
    if math.isfinite(f_plus) and math.isfinite(f_x):
        return (f_plus - f_x) / eps
    if math.isfinite(f_minus) and math.isfinite(f_x):
        return (f_x - f_minus) / eps
    return math.nan
