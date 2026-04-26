
from ._GaussianBeam_cpukernel_d_m2 import _GaussianBeam_cpukernel_d_m2 as kernel
import numba as nb
@nb.njit(nogil=True, cache=True, fastmath=True, parallel=True)
def _GaussianBeam_cpu_d_m2(z, w0, z0, m2, wl, n, out, ignore) :
    nmodels, npoints = out.shape
    for model in nb.prange(nmodels) :
        if ignore[model] : continue
        for point in range(npoints) :
            out[model, point] = kernel(z[point], w0[model], z0[model], m2[model], wl[model], n[model], )
