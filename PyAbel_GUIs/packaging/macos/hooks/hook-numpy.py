"""NumPy hook for packaging base-Conda packages from an isolated venv.

The upstream hook assumes that a PyInstaller venv layered on Conda also has a
local ``conda-meta`` directory. Our deliberately isolated packager does not,
so collect NumPy's package libraries and let normal Mach-O dependency analysis
collect the linked BLAS/Fortran libraries from the base environment.
"""

from PyInstaller.utils.hooks import collect_dynamic_libs


binaries = collect_dynamic_libs("numpy")
datas = []
hiddenimports = ["numpy.core._dtype_ctypes", "numpy.core._multiarray_tests"]
excludedimports = [
    "distutils",
    "f2py",
    "nose",
    "numpy.distutils",
    "pytest",
    "setuptools",
]
