
import numba as nb
from ._Diamond_cpukernel_function import _Diamond_cpukernel_function as model_scalar
from ._MLE_Poisson_cpukernel_deviance import _MLE_Poisson_cpukernel_deviance as deviance_scalar

@nb.njit(parallel=True, nogil=True, fastmath=True, cache=True)
def _Diamond_MLE_Poisson_cpu_trial_chi2(
    raw_data, x, y, d, mux, muy, amp, offset, weights, chi2, parameters, indices, steps, gradient,
    hessian, damping, nu, damping_max, damping_min, converged, improved, ignore
):
    nmodels, npoints = raw_data.shape
    nparams = steps.shape[1]

    for model in nb.prange(nmodels):
        if ignore[model]:
            continue

        chi_local = 0.0

        model_d = d[model]
        model_mux = mux[model]
        model_muy = muy[model]
        model_amp = amp[model]
        model_offset = offset[model]

        for point in range(npoints):
            point_x = x[point]
            point_y = y[point]
            
            point_raw_data = raw_data[model, point]
            point_weight = weights[model, point]

            mod = model_scalar(point_x, point_y, model_d, model_mux, model_muy, model_amp, model_offset)
            dev = deviance_scalar(point_raw_data, mod, point_weight)
            chi_local += dev

        new_chi2 = chi_local
        old_chi2 = chi2[model]

        pred = 0.0
        for param in range(nparams):
            pred += steps[model, param] * (
                -gradient[model, param]
                + damping[model] * max(1e-12, abs(hessian[model, param, param])) * steps[model, param]
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
