import numpy as np
import numba as nb
from ._IsoGaussian_cpukernel_function import _IsoGaussian_cpukernel_function as model_scalar
from ._MLE_Poisson_cpukernel_deviance import _MLE_Poisson_cpukernel_deviance as deviance_scalar
from ._MLE_Poisson_cpukernel_loss import _MLE_Poisson_cpukernel_loss as loss_scalar
from ._MLE_Poisson_cpukernel_fisher import _MLE_Poisson_cpukernel_fisher as fisher_scalar
from ._IsoGaussian_cpukernel_d_mux import _IsoGaussian_cpukernel_d_mux as d_mux
from ._IsoGaussian_cpukernel_d_muy import _IsoGaussian_cpukernel_d_muy as d_muy
from ._IsoGaussian_cpukernel_d_sig import _IsoGaussian_cpukernel_d_sig as d_sig
from ._IsoGaussian_cpukernel_d_amp import _IsoGaussian_cpukernel_d_amp as d_amp
from ._IsoGaussian_cpukernel_d_offset import _IsoGaussian_cpukernel_d_offset as d_offset
from ._IsoGaussian_cpukernel_d_pixx import _IsoGaussian_cpukernel_d_pixx as d_pixx
from ._IsoGaussian_cpukernel_d_pixy import _IsoGaussian_cpukernel_d_pixy as d_pixy
from ._IsoGaussian_cpukernel_d_nsig import _IsoGaussian_cpukernel_d_nsig as d_nsig
from funclp.modules.Function_LP._functions.gaussians._gaussians import gausfunc


MAX_PARAMS = 8
NHESS = int(8 * (8 + 1) // 2)

@nb.njit(parallel=True, nogil=True, fastmath=True, cache=True)
def _IsoGaussian_MLE_Poisson_cpu_assembly(
    raw_data, x, y, mux, muy, sig, amp, offset, pixx, pixy, nsig, weights, chi2, gradient, hessian, bool2fit, ignore
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

        model_mux = mux[model]
        model_muy = muy[model]
        model_sig = sig[model]
        model_amp = amp[model]
        model_offset = offset[model]
        model_pixx = pixx[model]
        model_pixy = pixy[model]
        model_nsig = nsig[model]

        safe_sig = model_sig
        if abs(safe_sig) < 1e-12:
            safe_sig = 1e-12
        inv_sig2 = 1.0 / (safe_sig * safe_sig)
        inv_sig3 = inv_sig2 / safe_sig

        for point in range(npoints):
            point_x = x[point]
            point_y = y[point]
            point_raw_data = raw_data[model, point]
            point_weight = weights[model, point]


            exx = gausfunc(point_x, model_mux, safe_sig, 1.0, 0.0, model_pixx, model_nsig)
            exy = gausfunc(point_y, model_muy, safe_sig, 1.0, 0.0, model_pixy, model_nsig)
            base = exx * exy
            mod = model_amp * base + model_offset

            dev = deviance_scalar(point_raw_data, mod, point_weight)
            los = loss_scalar(point_raw_data, mod, point_weight)
            fis = fisher_scalar(point_raw_data, mod, point_weight)
            chi_local += dev

            dx = point_x - model_mux
            dy = point_y - model_muy
            r2 = dx * dx + dy * dy

            count = 0
            if bool2fit[0]:
                jacob_local[count] = model_amp * base * dx * inv_sig2
                count += 1
            if bool2fit[1]:
                jacob_local[count] = model_amp * base * dy * inv_sig2
                count += 1
            if bool2fit[2]:
                jacob_local[count] = model_amp * base * r2 * inv_sig3
                count += 1
            if bool2fit[3]:
                jacob_local[count] = base
                count += 1
            if bool2fit[4]:
                jacob_local[count] = 1.0
                count += 1
            if bool2fit[5]:
                jacob_local[count] = d_pixx(point_x, point_y, model_mux, model_muy, safe_sig, model_amp, model_offset, model_pixx, model_pixy, model_nsig)
                count += 1
            if bool2fit[6]:
                jacob_local[count] = d_pixy(point_x, point_y, model_mux, model_muy, safe_sig, model_amp, model_offset, model_pixx, model_pixy, model_nsig)
                count += 1
            if bool2fit[7]:
                jacob_local[count] = d_nsig(point_x, point_y, model_mux, model_muy, safe_sig, model_amp, model_offset, model_pixx, model_pixy, model_nsig)
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
