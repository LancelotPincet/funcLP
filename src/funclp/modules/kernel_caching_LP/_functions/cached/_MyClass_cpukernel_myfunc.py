
from funclp import ufunc
import numba as nb
_MyClass_cpukernel_myfunc = nb.njit(nogil=True, cache=True)(ufunc.main_functions["MyClass_myfunc"])
