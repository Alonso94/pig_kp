from setuptools import setup
from torch.utils import cpp_extension
import os

# export cuda path
os.environ["CUDA_HOME"] = "~/miniconda3/envs/pig/"

setup(name='entropy_layer',
      ext_modules=[
            cpp_extension.CUDAExtension(name='entropy_layer',
                  sources=['src/pig/entropy_layer/entropy_cuda.cpp',
                   'src/pig/entropy_layer/entropy_cuda_kernel.cu'],
                  )
            ],
      cmdclass={'build_ext': cpp_extension.BuildExtension} )

setup(name='joint_entropy',
      ext_modules=[
            cpp_extension.CUDAExtension(name='joint_entropy',
                  sources=['src/pig/joint_entropy/joint_entropy_cuda.cpp',
                   'src/pig/joint_entropy/joint_entropy_cuda_kernel.cu'],
                  )
            ],
      cmdclass={'build_ext': cpp_extension.BuildExtension} )