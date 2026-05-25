
from funclp import ufunc
import numba as nb
_Gamma_cpukernel_fisher = nb.njit(nogil=True, cache=True)(ufunc.main_functions["Gamma_fisher"])
