
from distutils.core import setup
from distutils.extension import Extension
from Cython.Distutils import build_ext
import numpy as np                           # <---- New line


ext_modules = [Extension("gnl.gslrand",
    sources = ["gnl/gslrand.pyx"],
    libraries = ['gsl', 'gslcblas'],
    include_dirs = ["gnl"] # This line is crucial
    ),
    Extension("gnl.pdes.tadmor.tadmor",
    sources = ["gnl/pdes/tadmor/tadmor.pyx"],
    libraries = [],
    include_dirs = [np.get_include()] # This line is crucial
    ),
    Extension("gnl.pdes.petsc.kernel",
    sources = ["gnl/pdes/petsc/kernel.pyx"],
    libraries = [],
    include_dirs = [np.get_include()] # This line is crucial
    ),
    ]

setup(
    name = 'gnl',
    cmdclass = {'build_ext': build_ext},
    # include_dirs = [np.get_include()],         # <---- New line
    ext_modules = ext_modules,
    packages    = ['gnl', 'gnl.filter'],
    # package_dir = {'': ''},
    package_data = {'gnl':['*.pxd',]}

)
# -lmkl_intel_lp64 -lmkl_sequential -lmkl_core -lpthread -lm
