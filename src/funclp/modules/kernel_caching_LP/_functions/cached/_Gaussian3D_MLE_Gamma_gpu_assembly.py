import numba as nb
from numba import cuda
from ._Gaussian3D_gpukernel_function import _Gaussian3D_gpukernel_function as model_scalar
from ._MLE_Gamma_gpukernel_deviance import _MLE_Gamma_gpukernel_deviance as deviance_scalar
from ._MLE_Gamma_gpukernel_loss import _MLE_Gamma_gpukernel_loss as loss_scalar
from ._MLE_Gamma_gpukernel_fisher import _MLE_Gamma_gpukernel_fisher as fisher_scalar
from ._Gaussian3D_gpukernel_d_mux import _Gaussian3D_gpukernel_d_mux as d_mux
from ._Gaussian3D_gpukernel_d_muy import _Gaussian3D_gpukernel_d_muy as d_muy
from ._Gaussian3D_gpukernel_d_muz import _Gaussian3D_gpukernel_d_muz as d_muz
from ._Gaussian3D_gpukernel_d_sigx import _Gaussian3D_gpukernel_d_sigx as d_sigx
from ._Gaussian3D_gpukernel_d_sigy import _Gaussian3D_gpukernel_d_sigy as d_sigy
from ._Gaussian3D_gpukernel_d_sigz import _Gaussian3D_gpukernel_d_sigz as d_sigz
from ._Gaussian3D_gpukernel_d_amp import _Gaussian3D_gpukernel_d_amp as d_amp
from ._Gaussian3D_gpukernel_d_offset import _Gaussian3D_gpukernel_d_offset as d_offset
from ._Gaussian3D_gpukernel_d_pixx import _Gaussian3D_gpukernel_d_pixx as d_pixx
from ._Gaussian3D_gpukernel_d_pixy import _Gaussian3D_gpukernel_d_pixy as d_pixy
from ._Gaussian3D_gpukernel_d_pixz import _Gaussian3D_gpukernel_d_pixz as d_pixz
from ._Gaussian3D_gpukernel_d_nsig import _Gaussian3D_gpukernel_d_nsig as d_nsig
from ._Gaussian3D_gpukernel_d_theta import _Gaussian3D_gpukernel_d_theta as d_theta
from ._Gaussian3D_gpukernel_d_phi import _Gaussian3D_gpukernel_d_phi as d_phi


TPB = 128
MAX_PARAMS = 8
NHESS = int(8 * (8 + 1) // 2)

@nb.cuda.jit(cache=True)
def _Gaussian3D_MLE_Gamma_gpu_assembly(
    raw_data, x, y, z, mux, muy, muz, sigx, sigy, sigz, amp, offset, pixx, pixy, pixz, nsig, theta, phi, weights, chi2, gradient, hessian, bool2fit, ignore
):
    model = nb.cuda.blockIdx.x
    tid = nb.cuda.threadIdx.x
    bdim = nb.cuda.blockDim.x

    nmodels, npoints = raw_data.shape
    nparams = gradient.shape[1]

    if model >= nmodels or ignore[model]:
        return

    chi_local = nb.float32(0.0)
    grad_local = nb.cuda.local.array(MAX_PARAMS, nb.float32)
    hess_local = nb.cuda.local.array(NHESS, nb.float32)
    jacob_local = nb.cuda.local.array(MAX_PARAMS, nb.float32)

    for p in range(MAX_PARAMS):
        grad_local[p] = 0.0
    for idx in range(NHESS):
        hess_local[idx] = 0.0

    block_mux = mux[model]
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
    block_phi = phi[model]

    for point in range(tid, npoints, bdim):
        thread_x = x[point]
        thread_y = y[point]
        thread_z = z[point]
        
        thread_raw_data = raw_data[model, point]
        thread_weight = weights[model, point]

        mod = model_scalar(thread_x, thread_y, thread_z, block_mux, block_muy, block_muz, block_sigx, block_sigy, block_sigz, block_amp, block_offset, block_pixx, block_pixy, block_pixz, block_nsig, block_theta, block_phi)
        dev = deviance_scalar(thread_raw_data, mod, thread_weight)
        los = loss_scalar(thread_raw_data, mod, thread_weight)
        fis = fisher_scalar(thread_raw_data, mod, thread_weight)
        chi_local += dev

        count = 0

        if bool2fit[0]:
            jacob_local[count] = d_mux(thread_x, thread_y, thread_z, block_mux, block_muy, block_muz, block_sigx, block_sigy, block_sigz, block_amp, block_offset, block_pixx, block_pixy, block_pixz, block_nsig, block_theta, block_phi)
            count += 1

        if bool2fit[1]:
            jacob_local[count] = d_muy(thread_x, thread_y, thread_z, block_mux, block_muy, block_muz, block_sigx, block_sigy, block_sigz, block_amp, block_offset, block_pixx, block_pixy, block_pixz, block_nsig, block_theta, block_phi)
            count += 1

        if bool2fit[2]:
            jacob_local[count] = d_muz(thread_x, thread_y, thread_z, block_mux, block_muy, block_muz, block_sigx, block_sigy, block_sigz, block_amp, block_offset, block_pixx, block_pixy, block_pixz, block_nsig, block_theta, block_phi)
            count += 1

        if bool2fit[3]:
            jacob_local[count] = d_sigx(thread_x, thread_y, thread_z, block_mux, block_muy, block_muz, block_sigx, block_sigy, block_sigz, block_amp, block_offset, block_pixx, block_pixy, block_pixz, block_nsig, block_theta, block_phi)
            count += 1

        if bool2fit[4]:
            jacob_local[count] = d_sigy(thread_x, thread_y, thread_z, block_mux, block_muy, block_muz, block_sigx, block_sigy, block_sigz, block_amp, block_offset, block_pixx, block_pixy, block_pixz, block_nsig, block_theta, block_phi)
            count += 1

        if bool2fit[5]:
            jacob_local[count] = d_sigz(thread_x, thread_y, thread_z, block_mux, block_muy, block_muz, block_sigx, block_sigy, block_sigz, block_amp, block_offset, block_pixx, block_pixy, block_pixz, block_nsig, block_theta, block_phi)
            count += 1

        if bool2fit[6]:
            jacob_local[count] = d_amp(thread_x, thread_y, thread_z, block_mux, block_muy, block_muz, block_sigx, block_sigy, block_sigz, block_amp, block_offset, block_pixx, block_pixy, block_pixz, block_nsig, block_theta, block_phi)
            count += 1

        if bool2fit[7]:
            jacob_local[count] = d_offset(thread_x, thread_y, thread_z, block_mux, block_muy, block_muz, block_sigx, block_sigy, block_sigz, block_amp, block_offset, block_pixx, block_pixy, block_pixz, block_nsig, block_theta, block_phi)
            count += 1

        if bool2fit[8]:
            jacob_local[count] = d_pixx(thread_x, thread_y, thread_z, block_mux, block_muy, block_muz, block_sigx, block_sigy, block_sigz, block_amp, block_offset, block_pixx, block_pixy, block_pixz, block_nsig, block_theta, block_phi)
            count += 1

        if bool2fit[9]:
            jacob_local[count] = d_pixy(thread_x, thread_y, thread_z, block_mux, block_muy, block_muz, block_sigx, block_sigy, block_sigz, block_amp, block_offset, block_pixx, block_pixy, block_pixz, block_nsig, block_theta, block_phi)
            count += 1

        if bool2fit[10]:
            jacob_local[count] = d_pixz(thread_x, thread_y, thread_z, block_mux, block_muy, block_muz, block_sigx, block_sigy, block_sigz, block_amp, block_offset, block_pixx, block_pixy, block_pixz, block_nsig, block_theta, block_phi)
            count += 1

        if bool2fit[11]:
            jacob_local[count] = d_nsig(thread_x, thread_y, thread_z, block_mux, block_muy, block_muz, block_sigx, block_sigy, block_sigz, block_amp, block_offset, block_pixx, block_pixy, block_pixz, block_nsig, block_theta, block_phi)
            count += 1

        if bool2fit[12]:
            jacob_local[count] = d_theta(thread_x, thread_y, thread_z, block_mux, block_muy, block_muz, block_sigx, block_sigy, block_sigz, block_amp, block_offset, block_pixx, block_pixy, block_pixz, block_nsig, block_theta, block_phi)
            count += 1

        if bool2fit[13]:
            jacob_local[count] = d_phi(thread_x, thread_y, thread_z, block_mux, block_muy, block_muz, block_sigx, block_sigy, block_sigz, block_amp, block_offset, block_pixx, block_pixy, block_pixz, block_nsig, block_theta, block_phi)
            count += 1


        for p in range(nparams):
            Jp = jacob_local[p]
            grad_local[p] += Jp * los
            for q in range(p, nparams):
                idx = p * MAX_PARAMS - (p * (p - 1)) // 2 + (q - p)
                hess_local[idx] += Jp * jacob_local[q] * fis

    s_chi = nb.cuda.shared.array(TPB, nb.float32)
    s_grad = nb.cuda.shared.array((TPB, MAX_PARAMS), nb.float32)
    s_hess = nb.cuda.shared.array((TPB, NHESS), nb.float32)

    s_chi[tid] = chi_local
    for p in range(MAX_PARAMS):
        s_grad[tid, p] = grad_local[p]
    for idx in range(NHESS):
        s_hess[tid, idx] = hess_local[idx]

    nb.cuda.syncthreads()

    stride = bdim // 2
    while stride > 0:
        if tid < stride:
            s_chi[tid] += s_chi[tid + stride]
            for p in range(nparams):
                s_grad[tid, p] += s_grad[tid + stride, p]
            for p in range(nparams):
                for q in range(p, nparams):
                    idx = p * MAX_PARAMS - (p * (p - 1)) // 2 + (q - p)
                    s_hess[tid, idx] += s_hess[tid + stride, idx]
        nb.cuda.syncthreads()
        stride //= 2

    if tid == 0:
        chi2[model] = s_chi[0]
        for p in range(nparams):
            gradient[model, p] = s_grad[0, p]
        for p in range(nparams):
            for q in range(p, nparams):
                idx = p * MAX_PARAMS - (p * (p - 1)) // 2 + (q - p)
                v = s_hess[0, idx]
                hessian[model, p, q] = v
                hessian[model, q, p] = v
