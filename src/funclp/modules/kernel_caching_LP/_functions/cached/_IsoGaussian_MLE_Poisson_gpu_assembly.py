import numba as nb
from numba import cuda
from ._IsoGaussian_gpukernel_function import _IsoGaussian_gpukernel_function as model_scalar
from ._MLE_Poisson_gpukernel_deviance import _MLE_Poisson_gpukernel_deviance as deviance_scalar
from ._MLE_Poisson_gpukernel_loss import _MLE_Poisson_gpukernel_loss as loss_scalar
from ._MLE_Poisson_gpukernel_fisher import _MLE_Poisson_gpukernel_fisher as fisher_scalar
from ._IsoGaussian_gpukernel_d_mux import _IsoGaussian_gpukernel_d_mux as d_mux
from ._IsoGaussian_gpukernel_d_muy import _IsoGaussian_gpukernel_d_muy as d_muy
from ._IsoGaussian_gpukernel_d_sig import _IsoGaussian_gpukernel_d_sig as d_sig
from ._IsoGaussian_gpukernel_d_amp import _IsoGaussian_gpukernel_d_amp as d_amp
from ._IsoGaussian_gpukernel_d_offset import _IsoGaussian_gpukernel_d_offset as d_offset
from ._IsoGaussian_gpukernel_d_pixx import _IsoGaussian_gpukernel_d_pixx as d_pixx
from ._IsoGaussian_gpukernel_d_pixy import _IsoGaussian_gpukernel_d_pixy as d_pixy
from ._IsoGaussian_gpukernel_d_nsig import _IsoGaussian_gpukernel_d_nsig as d_nsig
from funclp.modules.Function_LP._functions.gaussians._gaussians import gausfunc


TPB = 128
MAX_PARAMS = 8
NHESS = int(8 * (8 + 1) // 2)

@nb.cuda.jit(cache=True, fastmath=True)
def _IsoGaussian_MLE_Poisson_gpu_assembly(
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
    grad_local = nb.cuda.local.array(MAX_PARAMS, nb.float32)
    hess_local = nb.cuda.local.array(NHESS, nb.float32)
    jacob_local = nb.cuda.local.array(MAX_PARAMS, nb.float32)

    for p in range(MAX_PARAMS):
        grad_local[p] = 0.0
    for idx in range(NHESS):
        hess_local[idx] = 0.0

    block_mux = mux[model]
    block_muy = muy[model]
    block_sig = sig[model]
    block_amp = amp[model]
    block_offset = offset[model]
    block_pixx = pixx[model]
    block_pixy = pixy[model]
    block_nsig = nsig[model]

    safe_sig = block_sig
    if abs(safe_sig) < 1e-12:
        safe_sig = 1e-12
    inv_sig2 = 1.0 / (safe_sig * safe_sig)
    inv_sig3 = inv_sig2 / safe_sig

    for point in range(tid, npoints, bdim):
        thread_x = x[point]
        thread_y = y[point]
        thread_raw_data = raw_data[model, point]
        thread_weight = weights[model, point]


        exx = gausfunc(thread_x, block_mux, safe_sig, 1.0, 0.0, block_pixx, block_nsig)
        exy = gausfunc(thread_y, block_muy, safe_sig, 1.0, 0.0, block_pixy, block_nsig)
        base = exx * exy
        mod = block_amp * base + block_offset

        dev = deviance_scalar(thread_raw_data, mod, thread_weight)
        los = loss_scalar(thread_raw_data, mod, thread_weight)
        fis = fisher_scalar(thread_raw_data, mod, thread_weight)
        chi_local += dev

        dx = thread_x - block_mux
        dy = thread_y - block_muy
        r2 = dx * dx + dy * dy

        count = 0
        if bool2fit[0]:
            jacob_local[count] = block_amp * base * dx * inv_sig2
            count += 1
        if bool2fit[1]:
            jacob_local[count] = block_amp * base * dy * inv_sig2
            count += 1
        if bool2fit[2]:
            jacob_local[count] = block_amp * base * r2 * inv_sig3
            count += 1
        if bool2fit[3]:
            jacob_local[count] = base
            count += 1
        if bool2fit[4]:
            jacob_local[count] = 1.0
            count += 1
        if bool2fit[5]:
            jacob_local[count] = d_pixx(thread_x, thread_y, block_mux, block_muy, safe_sig, block_amp, block_offset, block_pixx, block_pixy, block_nsig)
            count += 1
        if bool2fit[6]:
            jacob_local[count] = d_pixy(thread_x, thread_y, block_mux, block_muy, safe_sig, block_amp, block_offset, block_pixx, block_pixy, block_nsig)
            count += 1
        if bool2fit[7]:
            jacob_local[count] = d_nsig(thread_x, thread_y, block_mux, block_muy, safe_sig, block_amp, block_offset, block_pixx, block_pixy, block_nsig)
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
