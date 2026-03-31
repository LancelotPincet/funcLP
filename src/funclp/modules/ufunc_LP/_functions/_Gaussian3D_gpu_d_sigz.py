
from ._Gaussian3D_gpukernel_d_sigz import _Gaussian3D_gpukernel_d_sigz as kernel
import numba as nb
from numba import cuda
@nb.cuda.jit(cache=True, fastmath=False)
def _Gaussian3D_gpu_d_sigz(x, y, z, mux, muy, muz, sigx, sigy, sigz, amp, offset, pixx, pixy, pixz, nsig, theta, phi, eps, out, ignore) :
    nmodels, npoints = out.shape
    model, point = nb.cuda.grid(2)
    if model < nmodels and point < npoints and not ignore[model] :
        out[model, point] = kernel(x[point], y[point], z[point], mux[model], muy[model], muz[model], sigx[model], sigy[model], sigz[model], amp[model], offset[model], pixx[model], pixy[model], pixz[model], nsig[model], theta[model], phi[model], eps[model], )
