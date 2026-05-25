
from funclp import ufunc
import numba as nb
_Gamma_cpukernel_loglikelihood_reduced = nb.njit(nogil=True, cache=True)(ufunc.main_functions["Gamma_loglikelihood_reduced"])
