
from funclp import ufunc
import numba as nb
_Gamma_cpukernel_d2loglikelihood = nb.njit(nogil=True, cache=True)(ufunc.main_functions["Gamma_d2loglikelihood"])
