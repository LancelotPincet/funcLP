
from funclp import ufunc
import numba as nb
_Poisson_cpukernel_loglikelihood_reduced = nb.njit(nogil=True, cache=True)(ufunc.main_functions["Poisson_loglikelihood_reduced"])
