
from funclp import ufunc
import numba as nb
_IsoGaussian_cpukernel_d_mux = nb.njit(nogil=True, cache=True)(ufunc.main_functions["IsoGaussian_d_mux"])
