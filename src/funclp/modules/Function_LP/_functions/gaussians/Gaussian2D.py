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

def mux(res, *vars) :
    return get_mean(res, vars[0])
def muy(res, *vars) :
    return get_mean(res, vars[1])
def sigx(res, *vars) :
    return get_std(res, vars[0])
def sigy(res, *vars) :
    return get_std(res, vars[1])
def amp(res, *vars) :
    return get_amp(res)
def offset(res, *vars) :
    return get_offset(res)

# %% Function

class Gaussian2D(Function):

    @ufunc(
        variables=["x", "y"],
        parameters=[
            Parameter("mux", 0., estimate=mux),
            Parameter("muy", 0., estimate=muy),
            Parameter("sigx", 1/(2*np.pi), estimate=sigx, bounds=(0, None)),
            Parameter("sigy", 1/(2*np.pi), estimate=sigy, bounds=(0, None)),
            Parameter("amp", 1., estimate=amp),
            Parameter("offset", 0., estimate=offset),
            Parameter("pixx", -1.),
            Parameter("pixy", -1.),
            Parameter("nsig", -1.),
            Parameter("theta", 0.),
        ],
    )
    def function(x, y, /, mux=0., muy=0., sigx=1/(2*np.pi), sigy=1/(2*np.pi), amp=1., offset=0., pixx=-1., pixy=-1., nsig=-1., theta=0.) :
        x, y, mux, muy = correct_angle(theta, x, y, mux, muy)
        return amp * gausfunc(x, mux, sigx, 1, 0, pixx, nsig) * gausfunc(y, muy, sigy, 1, 0, pixy, nsig) + offset
    
    

    # Parameters derivatives
    @ufunc()
    def d_mux(x, y, /, mux, muy, sigx, sigy, amp, offset, pixx, pixy, nsig, theta) :
        x, y, mux, muy = correct_angle(theta, x, y, mux, muy)
        if abs(sigx) < 1e-12 :
            sigx = np.float32(1e-12)
        if abs(sigy) < 1e-12 :
            sigy = np.float32(1e-12)
        exx = gausfunc(x, mux, sigx, 1, 0, pixx, nsig)
        exy = gausfunc(y, muy, sigy, 1, 0, pixy, nsig)
        if theta == 0 :
            return amp * exx * exy * (x - mux) / sigx**2
        theta = theta / 180 * math.pi
        return amp * exx * exy * (math.cos(theta) * x / sigx**2 + math.sin(theta) * y / sigy**2)
    @ufunc()
    def d_muy(x, y, /, mux, muy, sigx, sigy, amp, offset, pixx, pixy, nsig, theta) :
        x, y, mux, muy = correct_angle(theta, x, y, mux, muy)
        if abs(sigx) < 1e-12 :
            sigx = np.float32(1e-12)
        if abs(sigy) < 1e-12 :
            sigy = np.float32(1e-12)
        exx = gausfunc(x, mux, sigx, 1, 0, pixx, nsig)
        exy = gausfunc(y, muy, sigy, 1, 0, pixy, nsig)
        if theta == 0 :
            return amp * exx * exy * (y - muy) / sigy**2
        theta = theta / 180 * math.pi
        return amp * exx * exy * (-math.sin(theta) * x / sigx**2 + math.cos(theta) * y / sigy**2)
    @ufunc()
    def d_sigx(x, y, /, mux, muy, sigx, sigy, amp, offset, pixx, pixy, nsig, theta) :
        x, y, mux, muy = correct_angle(theta, x, y, mux, muy)
        if abs(sigx) < 1e-12 :
            sigx = np.float32(1e-12)
        if abs(sigy) < 1e-12 :
            sigy = np.float32(1e-12)
        exx = gausfunc(x, mux, sigx, 1, 0, pixx, nsig)
        exy = gausfunc(y, muy, sigy, 1, 0, pixy, nsig)
        return amp * exx * exy * (x - mux)**2 / sigx**3
    @ufunc()
    def d_sigy(x, y, /, mux, muy, sigx, sigy, amp, offset, pixx, pixy, nsig, theta) :
        x, y, mux, muy = correct_angle(theta, x, y, mux, muy)
        if abs(sigx) < 1e-12 :
            sigx = np.float32(1e-12)
        if abs(sigy) < 1e-12 :
            sigy = np.float32(1e-12)
        exx = gausfunc(x, mux, sigx, 1, 0, pixx, nsig)
        exy = gausfunc(y, muy, sigy, 1, 0, pixy, nsig)
        return amp * exx * exy * (y - muy)**2 / sigy**3
    @ufunc()
    def d_amp(x, y, /, mux, muy, sigx, sigy, amp, offset, pixx, pixy, nsig, theta) :
        x, y, mux, muy = correct_angle(theta, x, y, mux, muy)
        exx = gausfunc(x, mux, sigx, 1, 0, pixx, nsig)
        exy = gausfunc(y, muy, sigy, 1, 0, pixy, nsig)
        return exx * exy
    @ufunc()
    def d_offset(x, y, /, mux, muy, sigx, sigy, amp, offset, pixx, pixy, nsig, theta) :
        return 1
    @ufunc()
    def d_theta(x, y, /, mux, muy, sigx, sigy, amp, offset, pixx, pixy, nsig, theta) :
        x, y, mux, muy = correct_angle(theta, x, y, mux, muy)
        if abs(sigx) < 1e-12 :
            sigx = np.float32(1e-12)
        if abs(sigy) < 1e-12 :
            sigy = np.float32(1e-12)
        exx = gausfunc(x, mux, sigx, 1, 0, pixx, nsig)
        exy = gausfunc(y, muy, sigy, 1, 0, pixy, nsig)
        return amp * exx * exy * (x - mux) * (y - muy) * (1 / sigx**2 - 1 / sigy**2)



    # Other attributes
    
    @property
    def integ(self) :
        return self.amp * (2 * np.pi) * self.sig**2 / self.pix**2
    @integ.setter
    def integ(self,value) :
        self.amp = value / (2 * np.pi) / self.sig**2 * self.pix**2
    @property
    def proba(self) :
        return np.erf(self.nsig / np.sqrt(2)) **2
    @proba.setter
    def proba(self, value) :
        self.nsig = sc.erfinv(np.sqrt(value)) * np.sqrt(2)
    @property
    def w(self) :
        return 2 * self.sig
    @w.setter
    def w(self,value) :
        self.sig = value / 2
    @property
    def wx(self) :
        return 2 * self.sigx
    @wx.setter
    def wx(self,value) :
        self.sigx = value / 2
    @property
    def wy(self) :
        return 2 * self.sigy
    @wy.setter
    def wy(self,value) :
        self.sigy = value / 2
    @property
    def FWHM(self) :
        return np.sqrt(2 * np.log(2)) * self.w
    @FWHM.setter
    def FWHM(self,value) :
        self.w = value / np.sqrt(2 * np.log(2))
    @property
    def FWHMx(self) :
        return np.sqrt(2 * np.log(2)) * self.wx
    @FWHMx.setter
    def FWHMx(self,value) :
        self.wx = value / np.sqrt(2 * np.log(2))
    @property
    def FWHMy(self) :
        return np.sqrt(2 * np.log(2)) * self.wy
    @FWHMy.setter
    def FWHMy(self,value) :
        self.wy = value / np.sqrt(2 * np.log(2))
    @property
    def pix(self) :
        return np.sqrt(self.pixx * self.pixy)
    @pix.setter
    def pix(self, value) :
        self.pixx, self.pixy = value, value
    @property
    def sig(self) :
        return np.sqrt(self.sigx * self.sigy)
    @sig.setter
    def sig(self, value) :
        self.sigx, self.sigy = value, value
    @property
    def ecc(self) :
        a, b = np.min(np.vstack((self.sigx, self.sigy)), axis=0), np.max(np.vstack((self.sigx, self.sigy)), axis=0)
        return np.sqrt(1 - (a/b)**2)

    def _cpu_assembly_model_setup_source(self, model_params, parameters):
        return '''block_mux = mux[model]
        block_muy = muy[model]
        block_sigx = sigx[model]
        block_sigy = sigy[model]
        block_amp = amp[model]
        block_offset = offset[model]
        block_pixx = pixx[model]
        block_pixy = pixy[model]
        block_nsig = nsig[model]
        block_theta = theta[model]'''

    def _gpu_assembly_model_setup_source(self, block_params, parameters):
        return '''block_mux = mux[model]
    block_muy = muy[model]
    block_sigx = sigx[model]
    block_sigy = sigy[model]
    block_amp = amp[model]
    block_offset = offset[model]
    block_pixx = pixx[model]
    block_pixy = pixy[model]
    block_nsig = nsig[model]
    block_theta = theta[model]'''

    def _cpu_assembly_model_eval_source(self, inputs_scalar):
        return '''            mod = model_scalar(point_x, point_y, block_mux, block_muy, block_sigx, block_sigy, block_amp, block_offset, block_pixx, block_pixy, block_nsig, block_theta)
            dev = deviance_scalar(point_raw_data, mod, point_weight)
            los = loss_scalar(point_raw_data, mod, point_weight)
            fis = fisher_scalar(point_raw_data, mod, point_weight)
            chi_local += dev'''

    def _gpu_assembly_model_eval_source(self, inputs_threads):
        return '''        mod = model_scalar(thread_x, thread_y, block_mux, block_muy, block_sigx, block_sigy, block_amp, block_offset, block_pixx, block_pixy, block_nsig, block_theta)
        dev = deviance_scalar(thread_raw_data, mod, thread_weight)
        los = loss_scalar(thread_raw_data, mod, thread_weight)
        fis = fisher_scalar(thread_raw_data, mod, thread_weight)
        chi_local += dev'''

    def _cpu_assembly_derivatives_source(self, parameters, inputs_scalar):
        return '''            if bool2fit[0]:
                jacob_local[count] = d_mux(point_x, point_y, block_mux, block_muy, block_sigx, block_sigy, block_amp, block_offset, block_pixx, block_pixy, block_nsig, block_theta)
                count += 1
            if bool2fit[1]:
                jacob_local[count] = d_muy(point_x, point_y, block_mux, block_muy, block_sigx, block_sigy, block_amp, block_offset, block_pixx, block_pixy, block_nsig, block_theta)
                count += 1
            if bool2fit[2]:
                jacob_local[count] = d_sigx(point_x, point_y, block_mux, block_muy, block_sigx, block_sigy, block_amp, block_offset, block_pixx, block_pixy, block_nsig, block_theta)
                count += 1
            if bool2fit[3]:
                jacob_local[count] = d_sigy(point_x, point_y, block_mux, block_muy, block_sigx, block_sigy, block_amp, block_offset, block_pixx, block_pixy, block_nsig, block_theta)
                count += 1
            if bool2fit[4]:
                jacob_local[count] = d_amp(point_x, point_y, block_mux, block_muy, block_sigx, block_sigy, block_amp, block_offset, block_pixx, block_pixy, block_nsig, block_theta)
                count += 1
            if bool2fit[5]:
                jacob_local[count] = d_offset(point_x, point_y, block_mux, block_muy, block_sigx, block_sigy, block_amp, block_offset, block_pixx, block_pixy, block_nsig, block_theta)
                count += 1
            if bool2fit[6]:
                jacob_local[count] = d_pixx(point_x, point_y, block_mux, block_muy, block_sigx, block_sigy, block_amp, block_offset, block_pixx, block_pixy, block_nsig, block_theta)
                count += 1
            if bool2fit[7]:
                jacob_local[count] = d_pixy(point_x, point_y, block_mux, block_muy, block_sigx, block_sigy, block_amp, block_offset, block_pixx, block_pixy, block_nsig, block_theta)
                count += 1
            if bool2fit[8]:
                jacob_local[count] = d_nsig(point_x, point_y, block_mux, block_muy, block_sigx, block_sigy, block_amp, block_offset, block_pixx, block_pixy, block_nsig, block_theta)
                count += 1
            if bool2fit[9]:
                jacob_local[count] = d_theta(point_x, point_y, block_mux, block_muy, block_sigx, block_sigy, block_amp, block_offset, block_pixx, block_pixy, block_nsig, block_theta)
                count += 1'''

    def _gpu_assembly_derivatives_source(self, parameters, inputs_threads):
        return '''        if bool2fit[0]:
            jacob_local[count] = d_mux(thread_x, thread_y, block_mux, block_muy, block_sigx, block_sigy, block_amp, block_offset, block_pixx, block_pixy, block_nsig, block_theta)
            count += 1
        if bool2fit[1]:
            jacob_local[count] = d_muy(thread_x, thread_y, block_mux, block_muy, block_sigx, block_sigy, block_amp, block_offset, block_pixx, block_pixy, block_nsig, block_theta)
            count += 1
        if bool2fit[2]:
            jacob_local[count] = d_sigx(thread_x, thread_y, block_mux, block_muy, block_sigx, block_sigy, block_amp, block_offset, block_pixx, block_pixy, block_nsig, block_theta)
            count += 1
        if bool2fit[3]:
            jacob_local[count] = d_sigy(thread_x, thread_y, block_mux, block_muy, block_sigx, block_sigy, block_amp, block_offset, block_pixx, block_pixy, block_nsig, block_theta)
            count += 1
        if bool2fit[4]:
            jacob_local[count] = d_amp(thread_x, thread_y, block_mux, block_muy, block_sigx, block_sigy, block_amp, block_offset, block_pixx, block_pixy, block_nsig, block_theta)
            count += 1
        if bool2fit[5]:
            jacob_local[count] = d_offset(thread_x, thread_y, block_mux, block_muy, block_sigx, block_sigy, block_amp, block_offset, block_pixx, block_pixy, block_nsig, block_theta)
            count += 1
        if bool2fit[6]:
            jacob_local[count] = d_pixx(thread_x, thread_y, block_mux, block_muy, block_sigx, block_sigy, block_amp, block_offset, block_pixx, block_pixy, block_nsig, block_theta)
            count += 1
        if bool2fit[7]:
            jacob_local[count] = d_pixy(thread_x, thread_y, block_mux, block_muy, block_sigx, block_sigy, block_amp, block_offset, block_pixx, block_pixy, block_nsig, block_theta)
            count += 1
        if bool2fit[8]:
            jacob_local[count] = d_nsig(thread_x, thread_y, block_mux, block_muy, block_sigx, block_sigy, block_amp, block_offset, block_pixx, block_pixy, block_nsig, block_theta)
            count += 1
        if bool2fit[9]:
            jacob_local[count] = d_theta(thread_x, thread_y, block_mux, block_muy, block_sigx, block_sigy, block_amp, block_offset, block_pixx, block_pixy, block_nsig, block_theta)
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
    instance = Gaussian2D()
    plot(instance, debug_folder, variables, parameters)
