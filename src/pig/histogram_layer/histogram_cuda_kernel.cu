# include <torch/types.h>
  
#include <ATen/ATen.h>

# include <cuda.h>
# include <cuda_runtime.h>

namespace{
    // the sigmoid function
    template<typename scalar_t>
    __device__ __forceinline__ scalar_t sigmoid(scalar_t z) {
        return 1.0 / (1.0 + exp(-z));
    }
    // the derivative of the sigmoid function
    template<typename scalar_t>
    __device__ __forceinline__ scalar_t d_sigmoid(scalar_t z) {
        return sigmoid(z) * (1 - sigmoid(z));
    }
    // the kernel function to compute the histogram
    template<typename scalar_t>
    __device__ __forceinline__ scalar_t kernel(scalar_t d, scalar_t L, scalar_t B) {
        return sigmoid((d+L/2)/B) - sigmoid((d-L/2)/B);
    }
    // the derivative of the kernel function
    template<typename scalar_t>
    __device__ __forceinline__ scalar_t d_kernel(scalar_t d, scalar_t L, scalar_t B) {
        return 1/B * (d_sigmoid((d+L/2)/B) - d_sigmoid((d-L/2)/B));
    }
    // the dimensions of the input image as a constant memory
    __constant__ int d_H, d_W;
    // the kernel function to compute the histogram
    template<typename scalar_t>
    __global__ void histogram_cuda_forward_kernel(torch::PackedTensorAccessor32<scalar_t,3,torch::RestrictPtrTraits> input,
                                                        torch::PackedTensorAccessor32<scalar_t,2,torch::RestrictPtrTraits> histogram_output,
                                                        float L,
                                                        float B
                                                        ){
        // the image index
        int n = blockIdx.x;
        // the index of the pixel
        int x = blockIdx.y;
        int y = blockIdx.z;
        // the thread index
        int t = threadIdx.x;
        histogram_output[n][t] += kernel(input[n][x][y] - t, L, B);
    }
    // the kernel function to compute the histogram
    template<typename scalar_t>
    __global__ void histogram_cuda_backward_kernel(torch::PackedTensorAccessor32<scalar_t,3,torch::RestrictPtrTraits> input,
                                                torch::PackedTensorAccessor32<scalar_t,2,torch::RestrictPtrTraits> d_histogram,
                                                torch::PackedTensorAccessor32<scalar_t,3,torch::RestrictPtrTraits> grad_out,
                                                float L,
                                                float B                                                
                                                ){
        // the image index
        int n = blockIdx.x;
        // the index of the pixel
        int x = blockIdx.y;
        int y = blockIdx.z;
        // the thread index
        int t = threadIdx.x;
        // the gradient of the output
        grad_out[n][x][y] += d_kernel(input[n][x][y] - t, L, B) * d_histogram[n][t];
    }
} // namespace

// the forward pass of the histogram layer
// x: input tensor
// region_size: the size of the region
// bandwidth: the bandwidth of the kernel
torch::Tensor histogram_cuda_forward(torch::Tensor x, float bandwidth){
    // printf("histogram_cuda_forward\n");
    // cudaError_t cudaStatus;
	// cudaEvent_t start, stop;
	// cudaEventCreate(&start);
	// cudaEventCreate(&stop);
    // the parameters for the kernel function
    const float L=1.0/255.0;
    const float B=bandwidth;
    // get the shape of the input tensor
    // N x H x W
    int N = x.size(0);
    int H = x.size(1);
    int W = x.size(2);
    // copy C,H,W to constant memory
    cudaMemcpyToSymbol(d_H,&H,sizeof(int));
    cudaMemcpyToSymbol(d_W,&W,sizeof(int));
    // block size
    dim3 threads(256);
    // grid size
    dim3 grid(N,H,W);
    // define the output tensor
    // N x C x H x W
    auto histogram_output = torch::zeros({N,256}).to(x.device());
    // printf("grid: %d %d %d\n",grid.x,grid.y,grid.z);
    // printf("threads: %d %d\n",threads.x,threads.y);
    // cudaEventRecord(start,0);
    // call the kernel
    AT_DISPATCH_FLOATING_TYPES(x.type(),"histogram_cuda_forward",([&]{
        histogram_cuda_forward_kernel<float><<<grid,threads>>>(
            x.packed_accessor32<float,3,torch::RestrictPtrTraits>(),
            histogram_output.packed_accessor32<float,2,torch::RestrictPtrTraits>(),
             L,  B);
    }));
    cudaDeviceSynchronize();
    // cudaEventRecord(stop, 0);
	// cudaEventSynchronize(stop);
    // // Check for any errors launching the kernel
	// cudaStatus = cudaGetLastError();
	// if (cudaStatus != cudaSuccess) {
	// 	fprintf(stderr, "kernel launch failed: %s\n", cudaGetErrorString(cudaStatus));
	// }

	// // cudaDeviceSynchronize waits for the kernel to finish, and returns
	// // any errors encountered during the launch.
	// cudaStatus = cudaDeviceSynchronize();
	// if (cudaStatus != cudaSuccess) {
	// 	fprintf(stderr, "cudaDeviceSynchronize returned error code %d after launching addKernel!\n", cudaStatus);
	// }

	// float milliseconds = 0;
	// cudaEventElapsedTime(&milliseconds, start, stop);

	// std::cout << "GPU rendering required " << milliseconds/1000.0f << "s." << std::endl;

    // return the output
    return histogram_output;
}

// the backward pass of the histogram layer
// x: input tensor
// region_size: the size of the region
// bandwidth: the bandwidth of the kernel
torch::Tensor histogram_cuda_backward(torch::Tensor x,
                            torch::Tensor d_histogram,
                            float bandwidth){
    // printf("histogram_cuda_forward\n");
    cudaError_t cudaStatus;
	cudaEvent_t start, stop;
	cudaEventCreate(&start);
	cudaEventCreate(&stop);
    // the parameters for the kernel function
    const float L=1.0/255.0;
    const float B=bandwidth;
    // get the shape of the input tensor
    // N x H x W
    int N = x.size(0);
    int H = x.size(1);
    int W = x.size(2);
    // copy C,H,W to constant memory
    cudaMemcpyToSymbol(d_H,&H,sizeof(int));
    cudaMemcpyToSymbol(d_W,&W,sizeof(int));
    // block size
    dim3 threads(256);
    // grid size
    dim3 grid(N,H,W);
    // define the output tensor (the gradient)
    // N x H x W 
    auto grad_out = torch::zeros({N,H,W},x.options());
    // cudaEventRecord(start,0);
    // call the kernel
    AT_DISPATCH_FLOATING_TYPES(x.type(),"histogram_cuda_backward",([&]{
        histogram_cuda_backward_kernel<float><<<grid,threads>>>(
            x.packed_accessor32<float,3,torch::RestrictPtrTraits>(),
            d_histogram.packed_accessor32<float,2,torch::RestrictPtrTraits>(),
            grad_out.packed_accessor32<float,3,torch::RestrictPtrTraits>(),
             L,  B);
    }));
    cudaDeviceSynchronize();

    // cudaEventRecord(stop, 0);
	// cudaEventSynchronize(stop);
    // // Check for any errors launching the kernel
	// cudaStatus = cudaGetLastError();
	// if (cudaStatus != cudaSuccess) {
	// 	fprintf(stderr, "kernel launch failed: %s\n", cudaGetErrorString(cudaStatus));
	// }

	// // cudaDeviceSynchronize waits for the kernel to finish, and returns
	// // any errors encountered during the launch.
	// cudaStatus = cudaDeviceSynchronize();
	// if (cudaStatus != cudaSuccess) {
	// 	fprintf(stderr, "cudaDeviceSynchronize returned error code %d after launching addKernel!\n", cudaStatus);
	// }

	// float milliseconds = 0;
	// cudaEventElapsedTime(&milliseconds, start, stop);

	// std::cout << "GPU rendering required " << milliseconds/1000.0f << "s." << std::endl;

    // return the output
    return grad_out;
}