
from funclp import ufunc
import numba as nb
_Normal_cpukernel_loglikelihood_reduced = nb.njit(nogil=True, cache=True)(ufunc.main_functions["Normal_loglikelihood_reduced"])
