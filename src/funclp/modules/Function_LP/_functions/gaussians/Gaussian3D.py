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
gausfunc, get_mean, get_std, get_amp, get_offset, correct_angle_3D = rfrom("._gaussians", "gausfunc", "get_mean", "get_std", "get_amp", "get_offset", "correct_angle_3D")



# %% Parameters

def mux(res, *vars) :
    return get_mean(res, vars[0])
def muy(res, *vars) :
    return get_mean(res, vars[1])
def muz(res, *vars) :
    return get_mean(res, vars[2])
def sigx(res, *vars) :
    return get_std(res, vars[0])
def sigy(res, *vars) :
    return get_std(res, vars[1])
def sigz(res, *vars) :
    return get_std(res, vars[2])
def amp(res, *vars) :
    return get_amp(res)
def offset(res, *vars) :
    return get_offset(res)

# %% Function

class Gaussian3D(Function):

    @ufunc(
        variables=["x", "y", "z"],
        parameters=[
            Parameter("mux", 0., estimate=mux),
            Parameter("muy", 0., estimate=muy),
            Parameter("muz", 0., estimate=muz),
            Parameter("sigx", 1/(2*np.pi)**(3/2), estimate=sigx, bounds=(0, None)),
            Parameter("sigy", 1/(2*np.pi)**(3/2), estimate=sigy, bounds=(0, None)),
            Parameter("sigz", 1/(2*np.pi)**(3/2), estimate=sigz, bounds=(0, None)),
            Parameter("amp", 1., estimate=amp),
            Parameter("offset", 0., estimate=offset),
            Parameter("pixx", -1.),
            Parameter("pixy", -1.),
            Parameter("pixz", -1.),
            Parameter("nsig", -1.),
            Parameter("theta", 0.),
            Parameter("phi", 0.),
        ],
    )
    def function(x, y, z, /, mux=0., muy=0., muz=0., sigx=1/(2*np.pi)**(3/2), sigy=1/(2*np.pi)**(3/2), sigz=1/(2*np.pi)**(3/2), amp=1., offset=0., pixx=-1., pixy=-1., pixz=-1., nsig=-1., theta=0., phi=0.) :
        x, y, z, mux, muy, muz = correct_angle_3D(theta, phi, x, y, z, mux, muy, muz)
        return amp * gausfunc(x, mux, sigx, 1, 0, pixx, nsig) * gausfunc(y, muy, sigy, 1, 0, pixy, nsig) * gausfunc(z, muz, sigz, 1, 0, pixz, nsig) + offset

    # Parameters derivatives
    @ufunc()
    def d_mux(x, y, z, /, mux, muy, muz, sigx, sigy, sigz, amp, offset, pixx, pixy, pixz, nsig, theta, phi) :
        eps = np.float32(1e-3 * max(1.0, abs(mux)))
        x1, y1, z1, mux1, muy1, muz1 = correct_angle_3D(theta, phi, x, y, z, mux + eps, muy, muz)
        f_plus = amp * gausfunc(x1, mux1, sigx, 1, 0, pixx, nsig) * gausfunc(y1, muy1, sigy, 1, 0, pixy, nsig) * gausfunc(z1, muz1, sigz, 1, 0, pixz, nsig) + offset
        x2, y2, z2, mux2, muy2, muz2 = correct_angle_3D(theta, phi, x, y, z, mux - eps, muy, muz)
        f_minus = amp * gausfunc(x2, mux2, sigx, 1, 0, pixx, nsig) * gausfunc(y2, muy2, sigy, 1, 0, pixy, nsig) * gausfunc(z2, muz2, sigz, 1, 0, pixz, nsig) + offset
        return (f_plus - f_minus) / (2.0 * eps)

    @ufunc()
    def d_muy(x, y, z, /, mux, muy, muz, sigx, sigy, sigz, amp, offset, pixx, pixy, pixz, nsig, theta, phi) :
        eps = np.float32(1e-3 * max(1.0, abs(muy)))
        x1, y1, z1, mux1, muy1, muz1 = correct_angle_3D(theta, phi, x, y, z, mux, muy + eps, muz)
        f_plus = amp * gausfunc(x1, mux1, sigx, 1, 0, pixx, nsig) * gausfunc(y1, muy1, sigy, 1, 0, pixy, nsig) * gausfunc(z1, muz1, sigz, 1, 0, pixz, nsig) + offset
        x2, y2, z2, mux2, muy2, muz2 = correct_angle_3D(theta, phi, x, y, z, mux, muy - eps, muz)
        f_minus = amp * gausfunc(x2, mux2, sigx, 1, 0, pixx, nsig) * gausfunc(y2, muy2, sigy, 1, 0, pixy, nsig) * gausfunc(z2, muz2, sigz, 1, 0, pixz, nsig) + offset
        return (f_plus - f_minus) / (2.0 * eps)

    @ufunc()
    def d_muz(x, y, z, /, mux, muy, muz, sigx, sigy, sigz, amp, offset, pixx, pixy, pixz, nsig, theta, phi) :
        eps = np.float32(1e-3 * max(1.0, abs(muz)))
        x1, y1, z1, mux1, muy1, muz1 = correct_angle_3D(theta, phi, x, y, z, mux, muy, muz + eps)
        f_plus = amp * gausfunc(x1, mux1, sigx, 1, 0, pixx, nsig) * gausfunc(y1, muy1, sigy, 1, 0, pixy, nsig) * gausfunc(z1, muz1, sigz, 1, 0, pixz, nsig) + offset
        x2, y2, z2, mux2, muy2, muz2 = correct_angle_3D(theta, phi, x, y, z, mux, muy, muz - eps)
        f_minus = amp * gausfunc(x2, mux2, sigx, 1, 0, pixx, nsig) * gausfunc(y2, muy2, sigy, 1, 0, pixy, nsig) * gausfunc(z2, muz2, sigz, 1, 0, pixz, nsig) + offset
        return (f_plus - f_minus) / (2.0 * eps)

    @ufunc()
    def d_sigx(x, y, z, /, mux, muy, muz, sigx, sigy, sigz, amp, offset, pixx, pixy, pixz, nsig, theta, phi) :
        eps = np.float32(1e-3 * max(1.0, abs(sigx)))
        x1, y1, z1, mux1, muy1, muz1 = correct_angle_3D(theta, phi, x, y, z, mux, muy, muz)
        f_plus = amp * gausfunc(x1, mux1, sigx + eps, 1, 0, pixx, nsig) * gausfunc(y1, muy1, sigy, 1, 0, pixy, nsig) * gausfunc(z1, muz1, sigz, 1, 0, pixz, nsig) + offset
        f_minus = amp * gausfunc(x1, mux1, sigx - eps, 1, 0, pixx, nsig) * gausfunc(y1, muy1, sigy, 1, 0, pixy, nsig) * gausfunc(z1, muz1, sigz, 1, 0, pixz, nsig) + offset
        return (f_plus - f_minus) / (2.0 * eps)

    @ufunc()
    def d_sigy(x, y, z, /, mux, muy, muz, sigx, sigy, sigz, amp, offset, pixx, pixy, pixz, nsig, theta, phi) :
        eps = np.float32(1e-3 * max(1.0, abs(sigy)))
        x1, y1, z1, mux1, muy1, muz1 = correct_angle_3D(theta, phi, x, y, z, mux, muy, muz)
        f_plus = amp * gausfunc(x1, mux1, sigx, 1, 0, pixx, nsig) * gausfunc(y1, muy1, sigy + eps, 1, 0, pixy, nsig) * gausfunc(z1, muz1, sigz, 1, 0, pixz, nsig) + offset
        f_minus = amp * gausfunc(x1, mux1, sigx, 1, 0, pixx, nsig) * gausfunc(y1, muy1, sigy - eps, 1, 0, pixy, nsig) * gausfunc(z1, muz1, sigz, 1, 0, pixz, nsig) + offset
        return (f_plus - f_minus) / (2.0 * eps)

    @ufunc()
    def d_sigz(x, y, z, /, mux, muy, muz, sigx, sigy, sigz, amp, offset, pixx, pixy, pixz, nsig, theta, phi) :
        eps = np.float32(1e-3 * max(1.0, abs(sigz)))
        x1, y1, z1, mux1, muy1, muz1 = correct_angle_3D(theta, phi, x, y, z, mux, muy, muz)
        f_plus = amp * gausfunc(x1, mux1, sigx, 1, 0, pixx, nsig) * gausfunc(y1, muy1, sigy, 1, 0, pixy, nsig) * gausfunc(z1, muz1, sigz + eps, 1, 0, pixz, nsig) + offset
        f_minus = amp * gausfunc(x1, mux1, sigx, 1, 0, pixx, nsig) * gausfunc(y1, muy1, sigy, 1, 0, pixy, nsig) * gausfunc(z1, muz1, sigz - eps, 1, 0, pixz, nsig) + offset
        return (f_plus - f_minus) / (2.0 * eps)

    @ufunc()
    def d_amp(x, y, z, /, mux, muy, muz, sigx, sigy, sigz, amp, offset, pixx, pixy, pixz, nsig, theta, phi) :
        x, y, z, mux, muy, muz = correct_angle_3D(theta, phi, x, y, z, mux, muy, muz)
        return gausfunc(x, mux, sigx, 1, 0, pixx, nsig) * gausfunc(y, muy, sigy, 1, 0, pixy, nsig) * gausfunc(z, muz, sigz, 1, 0, pixz, nsig)

    @ufunc()
    def d_offset(x, y, z, /, mux, muy, muz, sigx, sigy, sigz, amp, offset, pixx, pixy, pixz, nsig, theta, phi) :
        return 1
    
    

    # Other attributes
    
    @property
    def integ(self) :
        return self.amp * (2 * np.pi)**(3/2) * self.sig**3 / np.abs(self.pix)**3
    @integ.setter
    def integ(self,value) :
        self.amp = value / (2 * np.pi)**(3/2) / self.sig**2 * np.abs(self.pix)**3
    @property
    def proba(self) :
        return np.erf(self.nsig / np.sqrt(2)) **3
    @proba.setter
    def proba(self, value) :
        self.nsig = sc.erfinv(value**(1/3)) * np.sqrt(2)
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
    def wz(self) :
        return 2 * self.sigz
    @wz.setter
    def wz(self,value) :
        self.sigz = value / 2
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
    def FWHMz(self) :
        return np.sqrt(2 * np.log(2)) * self.wz
    @FWHMz.setter
    def FWHMz(self,value) :
        self.wz = value / np.sqrt(2 * np.log(2))
    @property
    def pix(self) :
        return (self.pixx * self.pixy * self.pixz)**(1/3)
    @pix.setter
    def pix(self, value) :
        self.pixx, self.pixy, self.pixz = value, value, value
    @property
    def sig(self) :
        return (self.sigx * self.sigy * self.sigz)**(1/3)
    @sig.setter
    def sig(self, value) :
        self.sigx, self.sigy, self.sigz = value, value, value

    def _cpu_assembly_model_setup_source(self, model_params, parameters):
        return '''block_mux = mux[model]
        block_muy = muy[model]
        block_muz = muz[model]
        block_sigx = sigx[model]
        block_sigy = sigy[model]
        block_sigz = sigz[model]
        block_amp = amp[model]
        block_offset = offset[model]
        block_pixx = pixx[model]
        block_pixy = pixy[model]
        block_pixz = pixz[model]
        block_nsig = nsig[model]
        block_theta = theta[model]
        block_phi = phi[model]'''

    def _gpu_assembly_model_setup_source(self, block_params, parameters):
        return '''block_mux = mux[model]
    block_muy = muy[model]
    block_muz = muz[model]
    block_sigx = sigx[model]
    block_sigy = sigy[model]
    block_sigz = sigz[model]
    block_amp = amp[model]
    block_offset = offset[model]
    block_pixx = pixx[model]
    block_pixy = pixy[model]
    block_pixz = pixz[model]
    block_nsig = nsig[model]
    block_theta = theta[model]
    block_phi = phi[model]'''

    def _cpu_assembly_model_eval_source(self, inputs_scalar):
        return '''            mod = model_scalar(point_x, point_y, point_z, block_mux, block_muy, block_muz, block_sigx, block_sigy, block_sigz, block_amp, block_offset, block_pixx, block_pixy, block_pixz, block_nsig, block_theta, block_phi)
            dev = deviance_scalar(point_raw_data, mod, point_weight)
            los = loss_scalar(point_raw_data, mod, point_weight)
            fis = fisher_scalar(point_raw_data, mod, point_weight)
            chi_local += dev'''

    def _gpu_assembly_model_eval_source(self, inputs_threads):
        return '''        mod = model_scalar(thread_x, thread_y, thread_z, block_mux, block_muy, block_muz, block_sigx, block_sigy, block_sigz, block_amp, block_offset, block_pixx, block_pixy, block_pixz, block_nsig, block_theta, block_phi)
        dev = deviance_scalar(thread_raw_data, mod, thread_weight)
        los = loss_scalar(thread_raw_data, mod, thread_weight)
        fis = fisher_scalar(thread_raw_data, mod, thread_weight)
        chi_local += dev'''

    def _cpu_assembly_derivatives_source(self, parameters, inputs_scalar):
        return '''            if bool2fit[0]:
                jacob_local[count] = d_mux(point_x, point_y, point_z, block_mux, block_muy, block_muz, block_sigx, block_sigy, block_sigz, block_amp, block_offset, block_pixx, block_pixy, block_pixz, block_nsig, block_theta, block_phi)
                count += 1
            if bool2fit[1]:
                jacob_local[count] = d_muy(point_x, point_y, point_z, block_mux, block_muy, block_muz, block_sigx, block_sigy, block_sigz, block_amp, block_offset, block_pixx, block_pixy, block_pixz, block_nsig, block_theta, block_phi)
                count += 1
            if bool2fit[2]:
                jacob_local[count] = d_muz(point_x, point_y, point_z, block_mux, block_muy, block_muz, block_sigx, block_sigy, block_sigz, block_amp, block_offset, block_pixx, block_pixy, block_pixz, block_nsig, block_theta, block_phi)
                count += 1
            if bool2fit[3]:
                jacob_local[count] = d_sigx(point_x, point_y, point_z, block_mux, block_muy, block_muz, block_sigx, block_sigy, block_sigz, block_amp, block_offset, block_pixx, block_pixy, block_pixz, block_nsig, block_theta, block_phi)
                count += 1
            if bool2fit[4]:
                jacob_local[count] = d_sigy(point_x, point_y, point_z, block_mux, block_muy, block_muz, block_sigx, block_sigy, block_sigz, block_amp, block_offset, block_pixx, block_pixy, block_pixz, block_nsig, block_theta, block_phi)
                count += 1
            if bool2fit[5]:
                jacob_local[count] = d_sigz(point_x, point_y, point_z, block_mux, block_muy, block_muz, block_sigx, block_sigy, block_sigz, block_amp, block_offset, block_pixx, block_pixy, block_pixz, block_nsig, block_theta, block_phi)
                count += 1
            if bool2fit[6]:
                jacob_local[count] = d_amp(point_x, point_y, point_z, block_mux, block_muy, block_muz, block_sigx, block_sigy, block_sigz, block_amp, block_offset, block_pixx, block_pixy, block_pixz, block_nsig, block_theta, block_phi)
                count += 1
            if bool2fit[7]:
                jacob_local[count] = d_offset(point_x, point_y, point_z, block_mux, block_muy, block_muz, block_sigx, block_sigy, block_sigz, block_amp, block_offset, block_pixx, block_pixy, block_pixz, block_nsig, block_theta, block_phi)
                count += 1
            if bool2fit[8]:
                jacob_local[count] = d_pixx(point_x, point_y, point_z, block_mux, block_muy, block_muz, block_sigx, block_sigy, block_sigz, block_amp, block_offset, block_pixx, block_pixy, block_pixz, block_nsig, block_theta, block_phi)
                count += 1
            if bool2fit[9]:
                jacob_local[count] = d_pixy(point_x, point_y, point_z, block_mux, block_muy, block_muz, block_sigx, block_sigy, block_sigz, block_amp, block_offset, block_pixx, block_pixy, block_pixz, block_nsig, block_theta, block_phi)
                count += 1
            if bool2fit[10]:
                jacob_local[count] = d_pixz(point_x, point_y, point_z, block_mux, block_muy, block_muz, block_sigx, block_sigy, block_sigz, block_amp, block_offset, block_pixx, block_pixy, block_pixz, block_nsig, block_theta, block_phi)
                count += 1
            if bool2fit[11]:
                jacob_local[count] = d_nsig(point_x, point_y, point_z, block_mux, block_muy, block_muz, block_sigx, block_sigy, block_sigz, block_amp, block_offset, block_pixx, block_pixy, block_pixz, block_nsig, block_theta, block_phi)
                count += 1
            if bool2fit[12]:
                jacob_local[count] = d_theta(point_x, point_y, point_z, block_mux, block_muy, block_muz, block_sigx, block_sigy, block_sigz, block_amp, block_offset, block_pixx, block_pixy, block_pixz, block_nsig, block_theta, block_phi)
                count += 1
            if bool2fit[13]:
                jacob_local[count] = d_phi(point_x, point_y, point_z, block_mux, block_muy, block_muz, block_sigx, block_sigy, block_sigz, block_amp, block_offset, block_pixx, block_pixy, block_pixz, block_nsig, block_theta, block_phi)
                count += 1'''

    def _gpu_assembly_derivatives_source(self, parameters, inputs_threads):
        return self._cpu_assembly_derivatives_source(parameters, inputs_threads).replace('            ', '        ')



# %% Test function run
if __name__ == "__main__":
    from corelp import debug
    from funclp import plot
    import numpy as np
    debug_folder = debug(__file__)

    # Inputs

    variables = (
        np.linspace(0, 1, 100).reshape((1,1,100)),
        np.linspace(0, 1.5, 100).reshape((1,100,1)),
        np.linspace(0, 1., 100).reshape((100,1,1)),
    )
    parameters = dict()

    # Plot function
    instance = Gaussian3D()
    instance(*variables, **parameters)
