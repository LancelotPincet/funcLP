
from funclp import ufunc
import numba as nb
_Rectangle_cpukernel_d_mux = nb.njit(nogil=True, cache=True)(ufunc.main_functions["Rectangle_d_mux"])
