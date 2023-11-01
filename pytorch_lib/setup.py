from setuptools import setup, find_packages
from torch.utils.cpp_extension import BuildExtension, CUDAExtension


setup(
    name='pan_lib',
    version='v1.0',
    description='pytorch third-party library',
    author='gang.zhang',
    author_email='zhanggang11021136@gmail.com',
    ext_modules=[
        CUDAExtension(name = 'pan_lib.cuda_kernel',
                    sources = ['src/pt_lib_cuda.cpp', 'src/pt_lib_cuda_kernel.cu'],
                    include_dirs = ['src']),
    ],
    cmdclass={'build_ext': BuildExtension},
    packages=find_packages()
)