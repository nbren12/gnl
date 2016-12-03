from distutils.extension import Extension
from distutils.core import setup
import numpy
from Cython.Build import cythonize

headers = [numpy.get_include()]
print("headers", headers)
extensions = [
    Extension("tadmor", ["*.pyx"],
        include_dirs = headers,
        # libraries = [...],
        # library_dirs = [...])
    )
]
setup(
    name = "funfunfun",
    ext_modules = cythonize(extensions),  # accepts a glob pattern
)
