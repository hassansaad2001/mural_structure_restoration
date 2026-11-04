from setuptools import setup
from torch.utils.cpp_extension import BuildExtension, CUDAExtension

setup(
    name='upfirdn2d',
    ext_modules=[
        CUDAExtension(
            'upfirdn2d',
            ['upfirdn2d.cpp', 'upfirdn2d.cu'],
            extra_compile_args={
                'cxx': ['-O2', '-fPIC'],
                'nvcc': ['-O2', '-arch=sm_75', '-arch=sm_80', '-arch=sm_89']
            }
        )
    ],
    cmdclass={'build_ext': BuildExtension}
)