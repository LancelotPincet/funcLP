import numpy as np
import numba as nb
from ._GaussianBeam_cpukernel_function import _GaussianBeam_cpukernel_function as model_scalar
from ._MLE_Poisson_cpukernel_deviance import _MLE_Poisson_cpukernel_deviance as deviance_scalar
from ._MLE_Poisson_cpukernel_loss import _MLE_Poisson_cpukernel_loss as loss_scalar
from ._MLE_Poisson_cpukernel_fisher import _MLE_Poisson_cpukernel_fisher as fisher_scalar
from ._GaussianBeam_cpukernel_d_w0 import _GaussianBeam_cpukernel_d_w0 as d_w0
from ._GaussianBeam_cpukernel_d_z0 import _GaussianBeam_cpukernel_d_z0 as d_z0
from ._GaussianBeam_cpukernel_d_m2 import _GaussianBeam_cpukernel_d_m2 as d_m2
from ._GaussianBeam_cpukernel_d_wl import _GaussianBeam_cpukernel_d_wl as d_wl
from ._GaussianBeam_cpukernel_d_n import _GaussianBeam_cpukernel_d_n as d_n

MAX_PARAMS = 8
NHESS = int(8 * (8 + 1) // 2)

@nb.njit(parallel=True, nogil=True, fastmath=True, cache=True)
def _GaussianBeam_MLE_Poisson_cpu_assembly(
    raw_data, z, w0, z0, m2, wl, n, weights, chi2, gradient, hessian, bool2fit, ignore
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

        model_w0 = w0[model]
        model_z0 = z0[model]
        model_m2 = m2[model]
        model_wl = wl[model]
        model_n = n[model]

        for point in range(npoints):
            point_z = z[point]
            
            point_raw_data = raw_data[model, point]
            point_weight = weights[model, point]

            mod = model_scalar(point_z, model_w0, model_z0, model_m2, model_wl, model_n)
            dev = deviance_scalar(point_raw_data, mod, point_weight)
            los = loss_scalar(point_raw_data, mod, point_weight)
            fis = fisher_scalar(point_raw_data, mod, point_weight)

            chi_local += dev

            count = 0
            if bool2fit[0]:
                jacob_local[count] = d_w0(point_z, model_w0, model_z0, model_m2, model_wl, model_n)
                count += 1
            if bool2fit[1]:
                jacob_local[count] = d_z0(point_z, model_w0, model_z0, model_m2, model_wl, model_n)
                count += 1
            if bool2fit[2]:
                jacob_local[count] = d_m2(point_z, model_w0, model_z0, model_m2, model_wl, model_n)
                count += 1
            if bool2fit[3]:
                jacob_local[count] = d_wl(point_z, model_w0, model_z0, model_m2, model_wl, model_n)
                count += 1
            if bool2fit[4]:
                jacob_local[count] = d_n(point_z, model_w0, model_z0, model_m2, model_wl, model_n)
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
