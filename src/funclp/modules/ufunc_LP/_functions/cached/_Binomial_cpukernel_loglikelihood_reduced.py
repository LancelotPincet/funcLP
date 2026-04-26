
from funclp import ufunc
import numba as nb
_Binomial_cpukernel_loglikelihood_reduced = nb.njit(nogil=True, cache=True)(ufunc.main_functions["Binomial_loglikelihood_reduced"])
