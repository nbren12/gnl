
from distutils.core import setup
from distutils.extension import Extension
from Cython.Distutils import build_ext
import numpy as np                           # <---- New line

ext_modules = [Extension("gslrand",
    sources = ["gslrand.pyx"],
    libraries = ['gsl', 'cblas']
    )]

setup(
    name = 'gnl',
    cmdclass = {'build_ext': build_ext},
    # include_dirs = [np.get_include()],         # <---- New line
    ext_modules = ext_modules
)
# -lmkl_intel_lp64 -lmkl_sequential -lmkl_core -lpthread -lm
