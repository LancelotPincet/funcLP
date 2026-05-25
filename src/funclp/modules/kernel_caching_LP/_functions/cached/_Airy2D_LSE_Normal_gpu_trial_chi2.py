
import numba as nb
from numba import cuda
from ._Airy2D_gpukernel_function import _Airy2D_gpukernel_function as model_scalar
from ._LSE_Normal_gpukernel_deviance import _LSE_Normal_gpukernel_deviance as deviance_scalar

TPB = 128

@nb.cuda.jit(cache=True)
def _Airy2D_LSE_Normal_gpu_trial_chi2(
    raw_data, x, y, mux, muy, amp, offset, wl, NA, tol, weights, chi2, parameters, indices, steps, gradient,
    hessian, damping, nu, damping_max, damping_min, converged, improved, ignore
):
    model = nb.cuda.blockIdx.x
    tid = nb.cuda.threadIdx.x
    bdim = nb.cuda.blockDim.x

    nmodels, npoints = raw_data.shape
    nparams = steps.shape[1]

    if model >= nmodels or ignore[model]:
        return

    chi_local = nb.float32(0.0)

    block_mux = mux[model]
    block_muy = muy[model]
    block_amp = amp[model]
    block_offset = offset[model]
    block_wl = wl[model]
    block_NA = NA[model]
    block_tol = tol[model]

    for point in range(tid, npoints, bdim):
        thread_x = x[point]
        thread_y = y[point]
        
        thread_raw_data = raw_data[model, point]
        thread_weight = weights[model, point]

        pred = model_scalar(thread_x, thread_y, block_mux, block_muy, block_amp, block_offset, block_wl, block_NA, block_tol)
        dev = deviance_scalar(thread_raw_data, pred, thread_weight)
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
        new_chi2 = s_chi[0]
        old_chi2 = chi2[model]

        pred = 0.0
        for param in range(nparams):
            step = steps[model, param]
            pred += step * (
                -gradient[model, param]
                + damping[model] * max(1e-12, abs(hessian[model, param, param])) * step
            )
        pred *= 0.5

        ared = old_chi2 - new_chi2
        rho = ared / pred if pred > 1e-12 else -1.0
        rho = min(max(rho, -1e6), 1e6)

        if (not ignore[model]) and (pred > 1e-12) and (ared > 0.0):
            improved[model] = True
            chi2[model] = new_chi2
            tmp = 1.0 - (2.0 * rho - 1.0) ** 3
            if tmp < 1.0 / 3.0:
                tmp = 1.0 / 3.0
            elif tmp > 10.0:
                tmp = 10.0
            damping[model] *= tmp
            nu[model] = 2.0
            if damping[model] < damping_min:
                damping[model] = damping_min
        else:
            if not ignore[model]:
                for param in range(nparams):
                    parameters[model, indices[param]] -= steps[model, param]
            if damping[model] >= damping_max:
                converged[model] = -1
                improved[model] = True
            else:
                damping[model] *= nu[model]
                nu[model] *= 2.0
                if damping[model] > damping_max:
                    damping[model] = damping_max
