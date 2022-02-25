from torch.utils.cpp_extension import load
lltm_cuda = load(
    'histogram_layer', ['src/histogram_layer/histogram_cuda.cpp',
                   'src/histogram_layer/histogram_cuda_kernel.cu'], verbose=True)
help(lltm_cuda)