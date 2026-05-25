
from ._GaussianBeam_gpukernel_function import _GaussianBeam_gpukernel_function as kernel
import numba as nb
from numba import cuda
@nb.cuda.jit(cache=True, fastmath=True)
def _GaussianBeam_gpu_function(z, w0, z0, m2, wl, n, out, ignore) :
    nmodels, npoints = out.shape
    model, point = nb.cuda.grid(2)
    if model < nmodels and point < npoints and not ignore[model] :
        out[model, point] = kernel(z[point], w0[model], z0[model], m2[model], wl[model], n[model], )
