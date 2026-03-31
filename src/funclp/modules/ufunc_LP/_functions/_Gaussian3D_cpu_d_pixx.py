
from ._Gaussian3D_cpukernel_d_pixx import _Gaussian3D_cpukernel_d_pixx as kernel
import numba as nb
@nb.njit(nogil=True, cache=True, fastmath=False, parallel=True)
def _Gaussian3D_cpu_d_pixx(x, y, z, mux, muy, muz, sigx, sigy, sigz, amp, offset, pixx, pixy, pixz, nsig, theta, phi, eps, out, ignore) :
    nmodels, npoints = out.shape
    for model in nb.prange(nmodels) :
        if ignore[model] : continue
        for point in range(npoints) :
            out[model, point] = kernel(x[point], y[point], z[point], mux[model], muy[model], muz[model], sigx[model], sigy[model], sigz[model], amp[model], offset[model], pixx[model], pixy[model], pixz[model], nsig[model], theta[model], phi[model], eps[model], )
