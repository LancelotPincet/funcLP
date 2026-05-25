
from funclp import ufunc
import numba as nb
_Binomial_cpukernel_d2loglikelihood = nb.njit(nogil=True, cache=True)(ufunc.main_functions["Binomial_d2loglikelihood"])
