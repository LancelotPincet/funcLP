
from funclp import ufunc
import numba as nb
_MyClass_cpukernel_otherfunc = nb.njit(nogil=True, cache=True)(ufunc.main_functions["MyClass_otherfunc"])
