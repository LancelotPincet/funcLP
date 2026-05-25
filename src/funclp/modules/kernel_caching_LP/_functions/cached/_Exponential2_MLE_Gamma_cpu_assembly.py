import numpy as np
import numba as nb
from ._Exponential2_cpukernel_function import _Exponential2_cpukernel_function as model_scalar
from ._MLE_Gamma_cpukernel_deviance import _MLE_Gamma_cpukernel_deviance as deviance_scalar
from ._MLE_Gamma_cpukernel_loss import _MLE_Gamma_cpukernel_loss as loss_scalar
from ._MLE_Gamma_cpukernel_fisher import _MLE_Gamma_cpukernel_fisher as fisher_scalar
from ._Exponential2_cpukernel_d_tau1 import _Exponential2_cpukernel_d_tau1 as d_tau1
from ._Exponential2_cpukernel_d_tau2 import _Exponential2_cpukernel_d_tau2 as d_tau2
from ._Exponential2_cpukernel_d_amp1 import _Exponential2_cpukernel_d_amp1 as d_amp1
from ._Exponential2_cpukernel_d_amp2 import _Exponential2_cpukernel_d_amp2 as d_amp2
from ._Exponential2_cpukernel_d_offset import _Exponential2_cpukernel_d_offset as d_offset

MAX_PARAMS = 8
NHESS = int(8 * (8 + 1) // 2)

@nb.njit(parallel=True, nogil=True, fastmath=True, cache=True)
def _Exponential2_MLE_Gamma_cpu_assembly(
    raw_data, t, tau1, tau2, amp1, amp2, offset, weights, chi2, gradient, hessian, bool2fit, ignore
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

        model_tau1 = tau1[model]
        model_tau2 = tau2[model]
        model_amp1 = amp1[model]
        model_amp2 = amp2[model]
        model_offset = offset[model]

        for point in range(npoints):
            point_t = t[point]
            
            point_raw_data = raw_data[model, point]
            point_weight = weights[model, point]

            mod = model_scalar(point_t, model_tau1, model_tau2, model_amp1, model_amp2, model_offset)
            dev = deviance_scalar(point_raw_data, mod, point_weight)
            los = loss_scalar(point_raw_data, mod, point_weight)
            fis = fisher_scalar(point_raw_data, mod, point_weight)

            chi_local += dev

            count = 0
            if bool2fit[0]:
                jacob_local[count] = d_tau1(point_t, model_tau1, model_tau2, model_amp1, model_amp2, model_offset)
                count += 1
            if bool2fit[1]:
                jacob_local[count] = d_tau2(point_t, model_tau1, model_tau2, model_amp1, model_amp2, model_offset)
                count += 1
            if bool2fit[2]:
                jacob_local[count] = d_amp1(point_t, model_tau1, model_tau2, model_amp1, model_amp2, model_offset)
                count += 1
            if bool2fit[3]:
                jacob_local[count] = d_amp2(point_t, model_tau1, model_tau2, model_amp1, model_amp2, model_offset)
                count += 1
            if bool2fit[4]:
                jacob_local[count] = d_offset(point_t, model_tau1, model_tau2, model_amp1, model_amp2, model_offset)
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
