import numba as nb
from numba import cuda
from ._Rectangle_gpukernel_function import _Rectangle_gpukernel_function as model_scalar
from ._LSE_Normal_gpukernel_deviance import _LSE_Normal_gpukernel_deviance as deviance_scalar
from ._LSE_Normal_gpukernel_loss import _LSE_Normal_gpukernel_loss as loss_scalar
from ._LSE_Normal_gpukernel_fisher import _LSE_Normal_gpukernel_fisher as fisher_scalar
from ._Rectangle_gpukernel_d_l import _Rectangle_gpukernel_d_l as d_l
from ._Rectangle_gpukernel_d_ratio import _Rectangle_gpukernel_d_ratio as d_ratio
from ._Rectangle_gpukernel_d_mux import _Rectangle_gpukernel_d_mux as d_mux
from ._Rectangle_gpukernel_d_muy import _Rectangle_gpukernel_d_muy as d_muy
from ._Rectangle_gpukernel_d_amp import _Rectangle_gpukernel_d_amp as d_amp
from ._Rectangle_gpukernel_d_offset import _Rectangle_gpukernel_d_offset as d_offset
from ._Rectangle_gpukernel_d_theta import _Rectangle_gpukernel_d_theta as d_theta


TPB = 128
MAX_PARAMS = 8
NHESS = int(8 * (8 + 1) // 2)

@nb.cuda.jit(cache=True)
def _Rectangle_LSE_Normal_gpu_assembly(
    raw_data, x, y, l, ratio, mux, muy, amp, offset, theta, weights, chi2, gradient, hessian, bool2fit, ignore
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

    block_l = l[model]
    block_ratio = ratio[model]
    block_mux = mux[model]
    block_muy = muy[model]
    block_amp = amp[model]
    block_offset = offset[model]
    block_theta = theta[model]

    for point in range(tid, npoints, bdim):
        thread_x = x[point]
        thread_y = y[point]
        
        thread_raw_data = raw_data[model, point]
        thread_weight = weights[model, point]

        mod = model_scalar(thread_x, thread_y, block_l, block_ratio, block_mux, block_muy, block_amp, block_offset, block_theta)
        dev = deviance_scalar(thread_raw_data, mod, thread_weight)
        los = loss_scalar(thread_raw_data, mod, thread_weight)
        fis = fisher_scalar(thread_raw_data, mod, thread_weight)
        chi_local += dev

        count = 0

        if bool2fit[0]:
            jacob_local[count] = d_l(thread_x, thread_y, block_l, block_ratio, block_mux, block_muy, block_amp, block_offset, block_theta)
            count += 1

        if bool2fit[1]:
            jacob_local[count] = d_ratio(thread_x, thread_y, block_l, block_ratio, block_mux, block_muy, block_amp, block_offset, block_theta)
            count += 1

        if bool2fit[2]:
            jacob_local[count] = d_mux(thread_x, thread_y, block_l, block_ratio, block_mux, block_muy, block_amp, block_offset, block_theta)
            count += 1

        if bool2fit[3]:
            jacob_local[count] = d_muy(thread_x, thread_y, block_l, block_ratio, block_mux, block_muy, block_amp, block_offset, block_theta)
            count += 1

        if bool2fit[4]:
            jacob_local[count] = d_amp(thread_x, thread_y, block_l, block_ratio, block_mux, block_muy, block_amp, block_offset, block_theta)
            count += 1

        if bool2fit[5]:
            jacob_local[count] = d_offset(thread_x, thread_y, block_l, block_ratio, block_mux, block_muy, block_amp, block_offset, block_theta)
            count += 1

        if bool2fit[6]:
            jacob_local[count] = d_theta(thread_x, thread_y, block_l, block_ratio, block_mux, block_muy, block_amp, block_offset, block_theta)
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
