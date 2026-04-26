
from funclp import ufunc
import numba as nb
_Gamma_cpukernel_loglikelihood = nb.njit(nogil=True, cache=True)(ufunc.main_functions["Gamma_loglikelihood"])
