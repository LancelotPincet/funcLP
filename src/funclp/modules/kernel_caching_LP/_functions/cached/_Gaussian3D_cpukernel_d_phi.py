
from funclp import ufunc
import numba as nb
_Gaussian3D_cpukernel_d_phi = nb.njit(nogil=True, cache=True)(ufunc.main_functions["Gaussian3D_d_phi"])
