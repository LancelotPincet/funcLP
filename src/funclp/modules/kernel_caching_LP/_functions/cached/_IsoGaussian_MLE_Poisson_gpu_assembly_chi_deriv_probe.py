import numba as nb
from numba import cuda
from ._IsoGaussian_gpukernel_function import _IsoGaussian_gpukernel_function as model_scalar
from ._MLE_Poisson_gpukernel_deviance import _MLE_Poisson_gpukernel_deviance as deviance_scalar
from ._IsoGaussian_gpukernel_derivatives import _IsoGaussian_gpukernel_derivatives as derivatives_scalar


TPB = 128
MAX_PARAMS = 8

@nb.cuda.jit(cache=True)
def _IsoGaussian_MLE_Poisson_gpu_assembly_chi_deriv_probe(
    raw_data, x, y, mux, muy, sig, amp, offset, pixx, pixy, nsig, weights, chi2, gradient, hessian, bool2fit, ignore
):
    model = nb.cuda.blockIdx.x
    tid = nb.cuda.threadIdx.x
    bdim = nb.cuda.blockDim.x

    nmodels, npoints = raw_data.shape
    nparams = gradient.shape[1]

    if model >= nmodels or ignore[model]:
        return

    chi_local = nb.float32(0.0)
    dummy_local = nb.float32(0.0)
    jacob_local = nb.cuda.local.array(MAX_PARAMS, nb.float32)

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

        derivatives_scalar(thread_x, thread_y, block_mux, block_muy, block_sig, block_amp, block_offset, block_pixx, block_pixy, block_nsig, jacob_local, bool2fit)

        for p in range(nparams):
            dummy_local += jacob_local[p]

    s_chi = nb.cuda.shared.array(TPB, nb.float32)
    s_dummy = nb.cuda.shared.array(TPB, nb.float32)
    s_chi[tid] = chi_local
    s_dummy[tid] = dummy_local

    nb.cuda.syncthreads()

    stride = bdim // 2
    while stride > 0:
        if tid < stride:
            s_chi[tid] += s_chi[tid + stride]
            s_dummy[tid] += s_dummy[tid + stride]
        nb.cuda.syncthreads()
        stride //= 2

    if tid == 0:
        chi2[model] = s_chi[0]
        if nparams > 0:
            gradient[model, 0] = s_dummy[0]
