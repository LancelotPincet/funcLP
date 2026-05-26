#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# Date          : 2026-01-01
# Author        : Lancelot PINCET
# GitHub        : https://github.com/LancelotPincet
# Library       : funcLP



# %% Libraries
import numpy as np
import scipy.special as sc
import math
from funclp import Function, Parameter, ufunc
from corelp import rfrom
gausfunc, get_mean, get_std, get_amp, get_offset, correct_angle = rfrom("._gaussians", "gausfunc", "get_mean", "get_std", "get_amp", "get_offset", "correct_angle")



# %% Parameters

def mux(res, *args) :
    return get_mean(res, args[0])
def muy(res, *args) :
    return get_mean(res, args[1])
def sig(res, *args) :
    return np.sqrt(get_std(res, args[0]) * get_std(res, args[1]))
def amp(res, *args) :
    return get_amp(res)
def offset(res, *args) :
    return get_offset(res)

# %% Function

class IsoGaussian(Function):

    @ufunc(
        variables=["x", "y"],
        parameters=[
            Parameter("mux", 0., estimate=mux),
            Parameter("muy", 0., estimate=muy),
            Parameter("sig", 1/(2*np.pi), estimate=sig, bounds=(1e-12, None)),
            Parameter("amp", 1., estimate=amp),
            Parameter("offset", 0., estimate=offset),
            Parameter("pixx", -1.),
            Parameter("pixy", -1.),
            Parameter("nsig", -1.),
        ],
    )
    def function(x, y, /, mux=0., muy=0., sig=1/(2*np.pi), amp=1., offset=0., pixx=-1., pixy=-1., nsig=-1.) :
        return amp * gausfunc(x, mux, sig, 1, 0, pixx, nsig) * gausfunc(y, muy, sig, 1, 0, pixy, nsig) + offset
    
    

    # Parameters derivatives
    @ufunc()
    def d_mux(x, y, /, mux, muy, sig, amp, offset, pixx, pixy, nsig) :
        if abs(sig) < 1e-12 :
            sig = np.float32(1e-12)
        exx = gausfunc(x, mux, sig, 1, 0, pixx, nsig)
        exy = gausfunc(y, muy, sig, 1, 0, pixy, nsig)
        return amp * exx * exy * (x - mux) / sig**2
    @ufunc()
    def d_muy(x, y, /, mux, muy, sig, amp, offset, pixx, pixy, nsig) :
        if abs(sig) < 1e-12 :
            sig = np.float32(1e-12)
        exx = gausfunc(x, mux, sig, 1, 0, pixx, nsig)
        exy = gausfunc(y, muy, sig, 1, 0, pixy, nsig)
        return amp * exx * exy * (y - muy) / sig**2
    @ufunc()
    def d_sig(x, y, /, mux, muy, sig, amp, offset, pixx, pixy, nsig) :
        if abs(sig) < 1e-12 :
            sig = np.float32(1e-12)
        exx = gausfunc(x, mux, sig, 1, 0, pixx, nsig)
        exy = gausfunc(y, muy, sig, 1, 0, pixy, nsig)
        return amp * exx * exy * ((x - mux)**2 + (y - muy)**2) / sig**3
    @ufunc()
    def d_amp(x, y, /, mux, muy, sig, amp, offset, pixx, pixy, nsig) :
        exx = gausfunc(x, mux, sig, 1, 0, pixx, nsig)
        exy = gausfunc(y, muy, sig, 1, 0, pixy, nsig)
        return exx * exy
    @ufunc()
    def d_offset(x, y, /, mux, muy, sig, amp, offset, pixx, pixy, nsig) :
        return 1



    # Other attributes
    
    @property
    def integ(self) :
        return self.amp * (2 * np.pi) * self.sig**2 / self.pix**2
    @integ.setter
    def integ(self, value) :
        self.amp = value / (2 * np.pi) / self.sig**2 * self.pix**2
    @property
    def proba(self) :
        return np.erf(self.nsig / np.sqrt(2)) **2
    @proba.setter
    def proba(self,value) :
        self.nsig = sc.erfinv(np.sqrt(value))*np.sqrt(2)
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
    @property
    def pix(self) :
        return np.sqrt(self.pixx * self.pixy)
    @pix.setter
    def pix(self, value) :
        self.pixx, self.pixy = value, value

    def _cpu_assembly_extra_imports_source(self, estimator, function_name, estimator_name, distribution_name, parameters):
        return "from funclp.modules.Function_LP._functions.gaussians._gaussians import gausfunc"

    def _gpu_assembly_extra_imports_source(self, estimator, function_name, estimator_name, distribution_name, parameters):
        return "from funclp.modules.Function_LP._functions.gaussians._gaussians import gausfunc"

    def _cpu_assembly_model_setup_source(self, model_params, parameters):
        return model_params + '''

        safe_sig = model_sig
        if abs(safe_sig) < 1e-12:
            safe_sig = 1e-12
        inv_sig2 = 1.0 / (safe_sig * safe_sig)
        inv_sig3 = inv_sig2 / safe_sig'''

    def _gpu_assembly_model_setup_source(self, block_params, parameters):
        return block_params + '''

    safe_sig = block_sig
    if abs(safe_sig) < 1e-12:
        safe_sig = 1e-12
    inv_sig2 = 1.0 / (safe_sig * safe_sig)
    inv_sig3 = inv_sig2 / safe_sig'''

    def _cpu_assembly_model_eval_source(self, inputs_scalar):
        return '''
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
            r2 = dx * dx + dy * dy'''

    def _gpu_assembly_model_eval_source(self, inputs_threads):
        return '''
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
        r2 = dx * dx + dy * dy'''

    def _cpu_assembly_derivatives_source(self, parameters, inputs_scalar):
        return '''            if bool2fit[0]:
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
                count += 1'''

    def _gpu_assembly_derivatives_source(self, parameters, inputs_threads):
        return '''        if bool2fit[0]:
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
            count += 1'''



# %% Test function run
if __name__ == "__main__":
    from corelp import debug
    from funclp import plot
    import numpy as np
    debug_folder = debug(__file__)

    # Inputs

    variables = (
        np.linspace(0, 1, 1000).reshape((1,1000)),
        np.linspace(0, 1.5, 1000).reshape((1000,1)),
    )
    parameters = dict()

    # Plot function
    instance = IsoGaussian()
    plot(instance, debug_folder, variables, parameters)
