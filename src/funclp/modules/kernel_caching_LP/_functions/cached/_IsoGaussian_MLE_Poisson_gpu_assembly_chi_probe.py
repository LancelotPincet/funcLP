import numba as nb
from numba import cuda
from ._IsoGaussian_gpukernel_function import _IsoGaussian_gpukernel_function as model_scalar
from ._MLE_Poisson_gpukernel_deviance import _MLE_Poisson_gpukernel_deviance as deviance_scalar


TPB = 128

@nb.cuda.jit(cache=True)
def _IsoGaussian_MLE_Poisson_gpu_assembly_chi_probe(
    raw_data, x, y, mux, muy, sig, amp, offset, pixx, pixy, nsig, weights, chi2, gradient, hessian, bool2fit, ignore
):
    model = nb.cuda.blockIdx.x
    tid = nb.cuda.threadIdx.x
    bdim = nb.cuda.blockDim.x

    nmodels, npoints = raw_data.shape

    if model >= nmodels or ignore[model]:
        return

    chi_local = nb.float32(0.0)

    block_mux = mux[model]
    block_muy = muy[model]
    block_sig = sig[model]
    block_amp = amp[model]
    block_offset = offset[model]
    block_pixx = pixx[model]
    block_pixy = pixy[model]
    block_nsig = nsig[model]

    for point in range(tid, npoints, bdim):
        thread_x = x[point]
        thread_y = y[point]
        
        thread_raw_data = raw_data[model, point]
        thread_weight = weights[model, point]

        mod = model_scalar(thread_x, thread_y, block_mux, block_muy, block_sig, block_amp, block_offset, block_pixx, block_pixy, block_nsig)
        dev = deviance_scalar(thread_raw_data, mod, thread_weight)
        chi_local += dev

    s_chi = nb.cuda.shared.array(TPB, nb.float32)
    s_chi[tid] = chi_local

    nb.cuda.syncthreads()

    stride = bdim // 2
    while stride > 0:
        if tid < stride:
            s_chi[tid] += s_chi[tid + stride]
        nb.cuda.syncthreads()
        stride //= 2

    if tid == 0:
        chi2[model] = s_chi[0]
