
from ._Gaussian3D_cpukernel_function import _Gaussian3D_cpukernel_function as kernel
from funclp import Parameter, ufunc
import math
@ufunc(variables=['x', 'y', 'z'], data=[], parameters=[Parameter('mux', 0.0), Parameter('muy', 0.0), Parameter('muz', 0.0), Parameter('sigx', 0.06349363593424097), Parameter('sigy', 0.06349363593424097), Parameter('sigz', 0.06349363593424097), Parameter('amp', 1.0), Parameter('offset', 0.0), Parameter('pixx', -1.0), Parameter('pixy', -1.0), Parameter('pixz', -1.0), Parameter('nsig', -1.0), Parameter('theta', 0.0), Parameter('phi', 0.0)], constants=[], fastmath=False)
def d_muy(x, y, z, /, mux, muy, muz, sigx, sigy, sigz, amp, offset, pixx, pixy, pixz, nsig, theta, phi):
    eps = 1e-3 * max(1.0, abs(muy))
    f_plus = kernel(x, y, z, mux, muy + eps, muz, sigx, sigy, sigz, amp, offset, pixx, pixy, pixz, nsig, theta, phi)
    f_minus = kernel(x, y, z, mux, muy - eps, muz, sigx, sigy, sigz, amp, offset, pixx, pixy, pixz, nsig, theta, phi)
    if math.isfinite(f_plus) and math.isfinite(f_minus):
        return (f_plus - f_minus) / (2.0 * eps)
    f_x = kernel(x, y, z, mux, muy, muz, sigx, sigy, sigz, amp, offset, pixx, pixy, pixz, nsig, theta, phi)
    if math.isfinite(f_plus) and math.isfinite(f_x):
        return (f_plus - f_x) / eps
    if math.isfinite(f_minus) and math.isfinite(f_x):
        return (f_x - f_minus) / eps
    return math.nan
