
from funclp import ufunc
import numba as nb
_IsoGaussian_cpukernel_d_muy = nb.njit(nogil=True, cache=True)(ufunc.main_functions["IsoGaussian_d_muy"])
