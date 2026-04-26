
from funclp import ufunc
import numba as nb
_GaussianBeam_cpukernel_d_z0 = nb.njit(nogil=True, cache=True)(ufunc.main_functions["GaussianBeam_d_z0"])
