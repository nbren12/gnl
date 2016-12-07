
from distutils.core import setup
from distutils.extension import Extension
from Cython.Distutils import build_ext
import numpy as np                           # <---- New line


import os
os.environ['CC'] = 'gcc'
cython_kw = dict(extra_compile_args=['-fopenmp'], extra_link_args=['-fopenmp'])
ext_modules = [Extension("gnl.gslrand",
                         sources = ["gnl/gslrand.pyx"],
                         libraries = ['gsl', 'gslcblas'],
                         include_dirs = ["gnl"],
                         **cython_kw
),
               Extension("gnl.pdes.tadmor.tadmor",
                         sources = ["gnl/pdes/tadmor/tadmor.pyx"],
                         libraries = [],
                         include_dirs = ["gnl"],
                         **cython_kw
               ),
               Extension("gnl.pdes.petsc.kernel",
                         sources = ["gnl/pdes/petsc/kernel.pyx"],
                         libraries = [],
                         include_dirs = ["gnl"],
                         **cython_kw
               ),
               Extension("gnl.pdes.barotropic.swe",
                         sources = ["gnl/pdes/barotropic/swe.pyx"],
                         libraries = [],
                         include_dirs = ["gnl"],
                         **cython_kw
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
