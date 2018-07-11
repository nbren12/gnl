from setuptools import setup


def get_ext_modules():
    """Get the extension modules for the gnl library"""
    from distutils.extension import Extension
    import os
    os.environ['CC'] = 'gcc-7'
    cython_kw = dict(
        extra_compile_args=['-fopenmp'], extra_link_args=['-fopenmp'])
    ext_modules = [
        Extension(
            "gnl.gslrand",
            sources=["gnl/gslrand.pyx"],
            libraries=['gsl', 'gslcblas'],
            include_dirs=["gnl"],
            **cython_kw),
        Extension(
            "gnl.pdes.tadmor.tadmor",
            sources=["gnl/pdes/tadmor/tadmor.pyx"],
            libraries=[],
            include_dirs=["gnl"],
            **cython_kw),
        Extension(
            "gnl.pdes.petsc.kernel",
            sources=["gnl/pdes/petsc/kernel.pyx"],
            libraries=[],
            include_dirs=["gnl"],
            **cython_kw),
        Extension(
            "gnl.pdes.barotropic.swe",
            sources=["gnl/pdes/barotropic/swe.pyx"],
            libraries=[],
            include_dirs=["gnl"],
            **cython_kw),
    ]

    return ext_modules


setup(
    name='xnoah',
    version='0.0dev',
    packages=['gnl.xarray', 'gnl.xarray.sam', 'gnl', 'gnl.filter'],
    license='Creative Commons Attribution-Noncommercial-Share Alike license')
