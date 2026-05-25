import numpy as np
import numba as nb
from ._Spline_cpukernel_function import _Spline_cpukernel_function as model_scalar
from ._MLE_Normal_cpukernel_deviance import _MLE_Normal_cpukernel_deviance as deviance_scalar
from ._MLE_Normal_cpukernel_loss import _MLE_Normal_cpukernel_loss as loss_scalar
from ._MLE_Normal_cpukernel_fisher import _MLE_Normal_cpukernel_fisher as fisher_scalar
from ._Spline_cpukernel_d_mu import _Spline_cpukernel_d_mu as d_mu
from ._Spline_cpukernel_d_amp import _Spline_cpukernel_d_amp as d_amp
from ._Spline_cpukernel_d_offset import _Spline_cpukernel_d_offset as d_offset
from ._Spline_cpukernel_d_k import _Spline_cpukernel_d_k as d_k

MAX_PARAMS = 8
NHESS = int(8 * (8 + 1) // 2)

@nb.njit(parallel=True, nogil=True, fastmath=True, cache=True)
def _Spline_MLE_Normal_cpu_assembly(
    raw_data, x, mu, amp, offset, k, t, coeffs, weights, chi2, gradient, hessian, bool2fit, ignore
):
    nmodels, npoints = raw_data.shape
    nparams = gradient.shape[1]

    for model in nb.prange(nmodels):
        if ignore[model]:
            continue

        chi_local = nb.float32(0.0)
        grad_local = np.zeros(MAX_PARAMS, dtype=np.float32)
        hess_local = np.zeros(NHESS, dtype=np.float32)
        jacob_local = np.empty(MAX_PARAMS, dtype=np.float32)

        model_mu = mu[model]
        model_amp = amp[model]
        model_offset = offset[model]
        model_k = k[model]

        for point in range(npoints):
            point_x = x[point]
            
            point_raw_data = raw_data[model, point]
            point_weight = weights[model, point]

            mod = model_scalar(point_x, model_mu, model_amp, model_offset, model_k, t, coeffs)
            dev = deviance_scalar(point_raw_data, mod, point_weight)
            los = loss_scalar(point_raw_data, mod, point_weight)
            fis = fisher_scalar(point_raw_data, mod, point_weight)

            chi_local += dev

            count = 0
            if bool2fit[0]:
                jacob_local[count] = d_mu(point_x, model_mu, model_amp, model_offset, model_k, t, coeffs)
                count += 1
            if bool2fit[1]:
                jacob_local[count] = d_amp(point_x, model_mu, model_amp, model_offset, model_k, t, coeffs)
                count += 1
            if bool2fit[2]:
                jacob_local[count] = d_offset(point_x, model_mu, model_amp, model_offset, model_k, t, coeffs)
                count += 1
            if bool2fit[3]:
                jacob_local[count] = d_k(point_x, model_mu, model_amp, model_offset, model_k, t, coeffs)
                count += 1

            for p in range(nparams):
                Jp = jacob_local[p]
                grad_local[p] += Jp * los
                for q in range(p, nparams):
                    idx = p * MAX_PARAMS - (p * (p - 1)) // 2 + (q - p)
                    hess_local[idx] += Jp * jacob_local[q] * fis

        chi2[model] = chi_local

        for p in range(nparams):
            gradient[model, p] = grad_local[p]

        for p in range(nparams):
            for q in range(p, nparams):
                idx = p * MAX_PARAMS - (p * (p - 1)) // 2 + (q - p)
                v = hess_local[idx]
                hessian[model, p, q] = v
                hessian[model, q, p] = v
