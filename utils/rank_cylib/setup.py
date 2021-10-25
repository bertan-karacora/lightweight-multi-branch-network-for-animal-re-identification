from distutils.core import setup, Extension
from Cython.Build import cythonize
import numpy as np

setup(ext_modules = cythonize(
    Extension(
        'utils.rank_cylib.rank_cy',
        sources=['utils/rank_cylib/rank_cy.pyx'],
        language='c',
        include_dirs=[np.get_include()],
        library_dirs=[],
        libraries=[],
        extra_compile_args=[],
        extra_link_args=[],
        define_macros=[('NPY_NO_DEPRECATED_API', 'NPY_1_7_API_VERSION')]
    ),
    language_level = "3"
))
