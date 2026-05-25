import numpy as np
import numba as nb
from ._Polynomial5_cpukernel_function import _Polynomial5_cpukernel_function as model_scalar
from ._MLE_Binomial_cpukernel_deviance import _MLE_Binomial_cpukernel_deviance as deviance_scalar
from ._MLE_Binomial_cpukernel_loss import _MLE_Binomial_cpukernel_loss as loss_scalar
from ._MLE_Binomial_cpukernel_fisher import _MLE_Binomial_cpukernel_fisher as fisher_scalar
from ._Polynomial5_cpukernel_d_a import _Polynomial5_cpukernel_d_a as d_a
from ._Polynomial5_cpukernel_d_b import _Polynomial5_cpukernel_d_b as d_b
from ._Polynomial5_cpukernel_d_c import _Polynomial5_cpukernel_d_c as d_c
from ._Polynomial5_cpukernel_d_d import _Polynomial5_cpukernel_d_d as d_d
from ._Polynomial5_cpukernel_d_e import _Polynomial5_cpukernel_d_e as d_e
from ._Polynomial5_cpukernel_d_f import _Polynomial5_cpukernel_d_f as d_f

MAX_PARAMS = 8
NHESS = int(8 * (8 + 1) // 2)

@nb.njit(parallel=True, nogil=True, fastmath=True, cache=True)
def _Polynomial5_MLE_Binomial_cpu_assembly(
    raw_data, x, a, b, c, d, e, f, weights, chi2, gradient, hessian, bool2fit, ignore
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

        model_a = a[model]
        model_b = b[model]
        model_c = c[model]
        model_d = d[model]
        model_e = e[model]
        model_f = f[model]

        for point in range(npoints):
            point_x = x[point]
            
            point_raw_data = raw_data[model, point]
            point_weight = weights[model, point]

            mod = model_scalar(point_x, model_a, model_b, model_c, model_d, model_e, model_f)
            dev = deviance_scalar(point_raw_data, mod, point_weight)
            los = loss_scalar(point_raw_data, mod, point_weight)
            fis = fisher_scalar(point_raw_data, mod, point_weight)

            chi_local += dev

            count = 0
            if bool2fit[0]:
                jacob_local[count] = d_a(point_x, model_a, model_b, model_c, model_d, model_e, model_f)
                count += 1
            if bool2fit[1]:
                jacob_local[count] = d_b(point_x, model_a, model_b, model_c, model_d, model_e, model_f)
                count += 1
            if bool2fit[2]:
                jacob_local[count] = d_c(point_x, model_a, model_b, model_c, model_d, model_e, model_f)
                count += 1
            if bool2fit[3]:
                jacob_local[count] = d_d(point_x, model_a, model_b, model_c, model_d, model_e, model_f)
                count += 1
            if bool2fit[4]:
                jacob_local[count] = d_e(point_x, model_a, model_b, model_c, model_d, model_e, model_f)
                count += 1
            if bool2fit[5]:
                jacob_local[count] = d_f(point_x, model_a, model_b, model_c, model_d, model_e, model_f)
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
