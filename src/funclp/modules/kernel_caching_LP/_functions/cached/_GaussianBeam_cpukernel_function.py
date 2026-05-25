
from funclp import ufunc
import numba as nb
_GaussianBeam_cpukernel_function = nb.njit(nogil=True, cache=True)(ufunc.main_functions["GaussianBeam_function"])
