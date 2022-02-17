from torch.utils.cpp_extension import load
lltm_cuda = load(
    'joint_entropy', ['src/joint_entropy/joint_entropy_cuda.cpp',
                   'src/joint_entropy/joint_entropy_cuda_kernel.cu'], verbose=True)
help(lltm_cuda)