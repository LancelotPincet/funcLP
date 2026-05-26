
import numba as nb
from funclp.modules.Function_LP._functions.gaussians._gaussians import gausfunc

@nb.njit(cache=True, nogil=True, fastmath=True)
def _IsoGaussian_cpukernel_derivatives(x, y, mux, muy, sig, amp, offset, pixx, pixy, nsig, jacobian, bool2fit):
    count = 0
    s = sig
    if abs(s) < 1e-12:
        s = 1e-12
    exx = gausfunc(x, mux, s, 1.0, 0.0, pixx, nsig)
    exy = gausfunc(y, muy, s, 1.0, 0.0, pixy, nsig)
    base = exx * exy
    dx = x - mux
    dy = y - muy
    inv_sig2 = 1.0 / (s * s)
    r2 = dx * dx + dy * dy
    if bool2fit[0]:
        jacobian[count] = amp * base * dx * inv_sig2
        count += 1
    if bool2fit[1]:
        jacobian[count] = amp * base * dy * inv_sig2
        count += 1
    if bool2fit[2]:
        jacobian[count] = amp * base * r2 / (s * s * s)
        count += 1
    if bool2fit[3]:
        jacobian[count] = base
        count += 1
    if bool2fit[4]:
        jacobian[count] = 1.0
        count += 1
    if bool2fit[5]:
        eps = 1e-3 * max(1.0, abs(pixx))
        exx_plus = gausfunc(x, mux, s, 1.0, 0.0, pixx + eps, nsig)
        exx_minus = gausfunc(x, mux, s, 1.0, 0.0, pixx - eps, nsig)
        jacobian[count] = amp * (exx_plus - exx_minus) * exy / (2.0 * eps)
        count += 1
    if bool2fit[6]:
        eps = 1e-3 * max(1.0, abs(pixy))
        exy_plus = gausfunc(y, muy, s, 1.0, 0.0, pixy + eps, nsig)
        exy_minus = gausfunc(y, muy, s, 1.0, 0.0, pixy - eps, nsig)
        jacobian[count] = amp * exx * (exy_plus - exy_minus) / (2.0 * eps)
        count += 1
    if bool2fit[7]:
        eps = 1e-3 * max(1.0, abs(nsig))
        plus = gausfunc(x, mux, s, 1.0, 0.0, pixx, nsig + eps) * gausfunc(y, muy, s, 1.0, 0.0, pixy, nsig + eps)
        minus = gausfunc(x, mux, s, 1.0, 0.0, pixx, nsig - eps) * gausfunc(y, muy, s, 1.0, 0.0, pixy, nsig - eps)
        jacobian[count] = amp * (plus - minus) / (2.0 * eps)
        count += 1
