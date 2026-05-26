#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# Date          : 2026-01-01
# Author        : Lancelot PINCET
# GitHub        : https://github.com/LancelotPincet
# Library       : funcLP



# %% Libraries
import numpy as np
import scipy.special as sc
from funclp import Function, Parameter, ufunc
from corelp import rfrom
gausfunc, get_mean, get_std, get_amp, get_offset = rfrom("._gaussians", "gausfunc", "get_mean", "get_std", "get_amp", "get_offset")



# %% Parameters

def mu(res, *args) :
    return get_mean(res, args[0])
def sig(res, *args) :
    return get_std(res, args[0])
def amp(res, *args) :
    return get_amp(res)
def offset(res, *args) :
    return get_offset(res)



# %% Function

class Gaussian(Function):

    @ufunc(
        variables=["x"],
        parameters=[
            Parameter("mu", 0., estimate=mu),
            Parameter("sig", 1/np.sqrt(2*np.pi), estimate=sig, bounds=(0, None)),
            Parameter("amp", 1., estimate=amp),
            Parameter("offset", 0., estimate=offset),
            Parameter("pix", -1.),
            Parameter("nsig", -1.),
        ],
    )
    def function(x, /, mu=0., sig=1/np.sqrt(2*np.pi), amp=1., offset=0., pix=-1., nsig=-1.) :
        return gausfunc(x, mu, sig, amp, offset, pix, nsig)
    


    # Parameters derivatives
    @ufunc()
    def d_mu(x, /, mu, sig, amp, offset, pix, nsig) :
        if abs(sig) < 1e-12 :
            sig = np.float32(1e-12)
        ex = gausfunc(x, mu, sig, 1, 0, pix, nsig)
        return amp * ex * (x - mu) / sig**2
    @ufunc()
    def d_sig(x, /, mu, sig, amp, offset, pix, nsig) :
        if abs(sig) < 1e-12 :
            sig = np.float32(1e-12)
        ex = gausfunc(x, mu, sig, 1, 0, pix, nsig)
        return amp * ex * (x - mu)**2 / sig**3
    @ufunc()
    def d_amp(x, /, mu, sig, amp, offset, pix, nsig) :
        if abs(sig) < 1e-12 :
            sig = np.float32(1e-12)
        ex = gausfunc(x, mu, sig, 1, 0, pix, nsig)
        return ex
    @ufunc()
    def d_offset(x, /, mu, sig, amp, offset, pix, nsig) :
        return 1



    # Other attributes
    
    @property
    def integ(self) :
        return self.amp * np.sqrt(2 * np.pi) * self.sig / np.abs(self.pix)
    @integ.setter
    def integ(self,value) :
        self.amp = value / np.sqrt(2 * np.pi) / self.sig * np.abs(self.pix)
    @property
    def proba(self) :
        return np.erf(self.nsig / np.sqrt(2))
    @proba.setter
    def proba(self,value) :
        self.nsig = sc.erfinv(value) * np.sqrt(2)
    @property
    def w(self) :
        return 2 * self.sig
    @w.setter
    def w(self,value) :
        self.sig = value / 2
    @property
    def FWHM(self) :
        return np.sqrt(2 * np.log(2)) * self.w
    @FWHM.setter
    def FWHM(self,value) :
        self.w = value / np.sqrt(2 * np.log(2))

    def _cpu_assembly_extra_imports_source(self, estimator, function_name, estimator_name, distribution_name, parameters):
        return "from funclp.modules.Function_LP._functions.gaussians._gaussians import gausfunc"

    def _gpu_assembly_extra_imports_source(self, estimator, function_name, estimator_name, distribution_name, parameters):
        return "from funclp.modules.Function_LP._functions.gaussians._gaussians import gausfunc"

    def _cpu_assembly_model_setup_source(self, model_params, parameters):
        return '''model_mu = mu[model]
        model_sig = sig[model]
        model_amp = amp[model]
        model_offset = offset[model]
        model_pix = pix[model]
        model_nsig = nsig[model]

        safe_sig = model_sig
        if abs(safe_sig) < 1e-12:
            safe_sig = 1e-12
        inv_sig2 = 1.0 / (safe_sig * safe_sig)
        inv_sig3 = inv_sig2 / safe_sig'''

    def _gpu_assembly_model_setup_source(self, block_params, parameters):
        return '''model_mu = mu[model]
    model_sig = sig[model]
    model_amp = amp[model]
    model_offset = offset[model]
    model_pix = pix[model]
    model_nsig = nsig[model]

    safe_sig = model_sig
    if abs(safe_sig) < 1e-12:
        safe_sig = 1e-12
    inv_sig2 = 1.0 / (safe_sig * safe_sig)
    inv_sig3 = inv_sig2 / safe_sig'''

    def _cpu_assembly_model_eval_source(self, inputs_scalar):
        return '''            base = gausfunc(point_x, model_mu, safe_sig, 1.0, 0.0, model_pix, model_nsig)
            mod = model_amp * base + model_offset

            dev = deviance_scalar(point_raw_data, mod, point_weight)
            los = loss_scalar(point_raw_data, mod, point_weight)
            fis = fisher_scalar(point_raw_data, mod, point_weight)
            chi_local += dev

            dx = point_x - model_mu'''

    def _gpu_assembly_model_eval_source(self, inputs_threads):
        return '''        base = gausfunc(thread_x, model_mu, safe_sig, 1.0, 0.0, model_pix, model_nsig)
        mod = model_amp * base + model_offset

        dev = deviance_scalar(thread_raw_data, mod, thread_weight)
        los = loss_scalar(thread_raw_data, mod, thread_weight)
        fis = fisher_scalar(thread_raw_data, mod, thread_weight)
        chi_local += dev

        dx = thread_x - model_mu'''

    def _cpu_assembly_derivatives_source(self, parameters, inputs_scalar):
        return '''            if bool2fit[0]:
                jacob_local[count] = model_amp * base * dx * inv_sig2
                count += 1
            if bool2fit[1]:
                jacob_local[count] = model_amp * base * dx * dx * inv_sig3
                count += 1
            if bool2fit[2]:
                jacob_local[count] = base
                count += 1
            if bool2fit[3]:
                jacob_local[count] = 1.0
                count += 1
            if bool2fit[4]:
                jacob_local[count] = d_pix(point_x, model_mu, safe_sig, model_amp, model_offset, model_pix, model_nsig)
                count += 1
            if bool2fit[5]:
                jacob_local[count] = d_nsig(point_x, model_mu, safe_sig, model_amp, model_offset, model_pix, model_nsig)
                count += 1'''

    def _gpu_assembly_derivatives_source(self, parameters, inputs_threads):
        return '''        if bool2fit[0]:
            jacob_local[count] = model_amp * base * dx * inv_sig2
            count += 1
        if bool2fit[1]:
            jacob_local[count] = model_amp * base * dx * dx * inv_sig3
            count += 1
        if bool2fit[2]:
            jacob_local[count] = base
            count += 1
        if bool2fit[3]:
            jacob_local[count] = 1.0
            count += 1
        if bool2fit[4]:
            jacob_local[count] = d_pix(thread_x, model_mu, safe_sig, model_amp, model_offset, model_pix, model_nsig)
            count += 1
        if bool2fit[5]:
            jacob_local[count] = d_nsig(thread_x, model_mu, safe_sig, model_amp, model_offset, model_pix, model_nsig)
            count += 1'''



# %% Test function run
if __name__ == "__main__":
    from corelp import debug
    from funclp import plot
    import numpy as np
    debug_folder = debug(__file__)

    # Inputs
    variables = (
        np.linspace(-1, 1, 1000),
    )
    parameters = dict()

    # Plot function
    instance = Gaussian()
    plot(instance, debug_folder, variables, parameters)
