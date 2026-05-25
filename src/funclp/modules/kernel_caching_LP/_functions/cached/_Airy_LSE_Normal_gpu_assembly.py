import numba as nb
from numba import cuda
from ._Airy_gpukernel_function import _Airy_gpukernel_function as model_scalar
from ._LSE_Normal_gpukernel_deviance import _LSE_Normal_gpukernel_deviance as deviance_scalar
from ._LSE_Normal_gpukernel_loss import _LSE_Normal_gpukernel_loss as loss_scalar
from ._LSE_Normal_gpukernel_fisher import _LSE_Normal_gpukernel_fisher as fisher_scalar
from ._Airy_gpukernel_d_mu import _Airy_gpukernel_d_mu as d_mu
from ._Airy_gpukernel_d_amp import _Airy_gpukernel_d_amp as d_amp
from ._Airy_gpukernel_d_offset import _Airy_gpukernel_d_offset as d_offset
from ._Airy_gpukernel_d_wl import _Airy_gpukernel_d_wl as d_wl
from ._Airy_gpukernel_d_NA import _Airy_gpukernel_d_NA as d_NA
from ._Airy_gpukernel_d_tol import _Airy_gpukernel_d_tol as d_tol


TPB = 128
MAX_PARAMS = 8
NHESS = int(8 * (8 + 1) // 2)

@nb.cuda.jit(cache=True)
def _Airy_LSE_Normal_gpu_assembly(
    raw_data, x, mu, amp, offset, wl, NA, tol, weights, chi2, gradient, hessian, bool2fit, ignore
):
    model = nb.cuda.blockIdx.x
    tid = nb.cuda.threadIdx.x
    bdim = nb.cuda.blockDim.x

    nmodels, npoints = raw_data.shape
    nparams = gradient.shape[1]

    if model >= nmodels or ignore[model]:
        return

    chi_local = nb.float32(0.0)
    grad_local = nb.cuda.local.array(MAX_PARAMS, nb.float32)
    hess_local = nb.cuda.local.array(NHESS, nb.float32)
    jacob_local = nb.cuda.local.array(MAX_PARAMS, nb.float32)

    for p in range(MAX_PARAMS):
        grad_local[p] = 0.0
    for idx in range(NHESS):
        hess_local[idx] = 0.0

    block_mu = mu[model]
    block_amp = amp[model]
    block_offset = offset[model]
    block_wl = wl[model]
    block_NA = NA[model]
    block_tol = tol[model]

    for point in range(tid, npoints, bdim):
        thread_x = x[point]
        
        thread_raw_data = raw_data[model, point]
        thread_weight = weights[model, point]

        mod = model_scalar(thread_x, block_mu, block_amp, block_offset, block_wl, block_NA, block_tol)
        dev = deviance_scalar(thread_raw_data, mod, thread_weight)
        los = loss_scalar(thread_raw_data, mod, thread_weight)
        fis = fisher_scalar(thread_raw_data, mod, thread_weight)
        chi_local += dev

        count = 0

        if bool2fit[0]:
            jacob_local[count] = d_mu(thread_x, block_mu, block_amp, block_offset, block_wl, block_NA, block_tol)
            count += 1

        if bool2fit[1]:
            jacob_local[count] = d_amp(thread_x, block_mu, block_amp, block_offset, block_wl, block_NA, block_tol)
            count += 1

        if bool2fit[2]:
            jacob_local[count] = d_offset(thread_x, block_mu, block_amp, block_offset, block_wl, block_NA, block_tol)
            count += 1

        if bool2fit[3]:
            jacob_local[count] = d_wl(thread_x, block_mu, block_amp, block_offset, block_wl, block_NA, block_tol)
            count += 1

        if bool2fit[4]:
            jacob_local[count] = d_NA(thread_x, block_mu, block_amp, block_offset, block_wl, block_NA, block_tol)
            count += 1

        if bool2fit[5]:
            jacob_local[count] = d_tol(thread_x, block_mu, block_amp, block_offset, block_wl, block_NA, block_tol)
            count += 1


        for p in range(nparams):
            Jp = jacob_local[p]
            grad_local[p] += Jp * los
            for q in range(p, nparams):
                idx = p * MAX_PARAMS - (p * (p - 1)) // 2 + (q - p)
                hess_local[idx] += Jp * jacob_local[q] * fis

    s_chi = nb.cuda.shared.array(TPB, nb.float32)
    s_grad = nb.cuda.shared.array((TPB, MAX_PARAMS), nb.float32)
    s_hess = nb.cuda.shared.array((TPB, NHESS), nb.float32)

    s_chi[tid] = chi_local
    for p in range(MAX_PARAMS):
        s_grad[tid, p] = grad_local[p]
    for idx in range(NHESS):
        s_hess[tid, idx] = hess_local[idx]

    nb.cuda.syncthreads()

    stride = bdim // 2
    while stride > 0:
        if tid < stride:
            s_chi[tid] += s_chi[tid + stride]
            for p in range(nparams):
                s_grad[tid, p] += s_grad[tid + stride, p]
            for p in range(nparams):
                for q in range(p, nparams):
                    idx = p * MAX_PARAMS - (p * (p - 1)) // 2 + (q - p)
                    s_hess[tid, idx] += s_hess[tid + stride, idx]
        nb.cuda.syncthreads()
        stride //= 2

    if tid == 0:
        chi2[model] = s_chi[0]
        for p in range(nparams):
            gradient[model, p] = s_grad[0, p]
        for p in range(nparams):
            for q in range(p, nparams):
                idx = p * MAX_PARAMS - (p * (p - 1)) // 2 + (q - p)
                v = s_hess[0, idx]
                hessian[model, p, q] = v
                hessian[model, q, p] = v
