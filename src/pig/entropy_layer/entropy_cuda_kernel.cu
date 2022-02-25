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
    // the entropy function
    template<typename scalar_t>
    __device__ __forceinline__ scalar_t entropy(scalar_t z) {
        return -z * log(z+1e-8);
    }
    // the derivative of the entropy function
    template<typename scalar_t>
    __device__ __forceinline__ scalar_t d_entropy(scalar_t z) {
        return -1.0 / log(10.0) * (1.0+log(z+1e-8));
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
    // the kernel function to compute the histogram
    template<typename scalar_t>
    __global__ void entropy_cuda_forward_kernel(torch::PackedTensorAccessor32<scalar_t,4,torch::RestrictPtrTraits> input,
                                                        torch::PackedTensorAccessor32<scalar_t,3,torch::RestrictPtrTraits> entropy_output,
                                                        float L,
                                                        float B,
                                                        int patch_size                                                       
                                                        ){
        // the image index
        int n = blockIdx.y;
        // the index of the first patch
        int p = blockIdx.x;
        // the thread index
        int t = threadIdx.x;
        float prob = 0, depth_prob=0;
        #pragma unroll
        for(int i=0;i<patch_size;i++){
            prob += kernel(input[n][p][i][0]-t,L,B)/patch_size;
            prob += kernel(input[n][p][i][1]-t,L,B)/patch_size;
            prob += kernel(input[n][p][i][2]-t,L,B)/patch_size;
            depth_prob += kernel(input[n][p][i][3]-t,L,B)/patch_size;
        }
        // update the output
        atomicAdd(&entropy_output[n][0][p],entropy(prob));
        atomicAdd(&entropy_output[n][1][p],entropy(depth_prob));
    }
    // the kernel function to compute the histogram
    template<typename scalar_t>
    __global__ void entropy_cuda_backward_kernel(torch::PackedTensorAccessor32<scalar_t,4,torch::RestrictPtrTraits> input,
                                                torch::PackedTensorAccessor32<scalar_t,3,torch::RestrictPtrTraits> d_entropy_out,
                                                torch::PackedTensorAccessor32<scalar_t,4,torch::RestrictPtrTraits> grad_out,
                                                float L,
                                                float B,
                                                int patch_size
                                                ){
        // the image index
        int n = blockIdx.y;
        // the index of the patch
        int p = blockIdx.x;
        // the thread index
        int t = threadIdx.x;
        float prob = 0, depth_prob=0;
        #pragma unroll
        for(int i=0;i<patch_size;i++){
            prob += kernel(input[n][p][i][0]-t,L,B)/patch_size;
            prob += kernel(input[n][p][i][1]-t,L,B)/patch_size;
            prob += kernel(input[n][p][i][2]-t,L,B)/patch_size;
            depth_prob += kernel(input[n][p][i][3]-t,L,B)/patch_size;
        }
        // the derivative of the entropy function
        float d_prob = d_entropy_out[n][0][p] * d_entropy(prob);
        float d_depth_prob = d_entropy_out[n][1][p] * d_entropy(depth_prob);
        prob = 0;
        depth_prob=0;
        // compute the gradient
        #pragma unroll
        for(int i=0;i<patch_size;i++){
            prob += d_kernel(input[n][p][i][0]-t,L,B)/patch_size;
            prob += d_kernel(input[n][p][i][1]-t,L,B)/patch_size;
            prob += d_kernel(input[n][p][i][2]-t,L,B)/patch_size;
            grad_out[n][0][p][i] += d_prob * prob;
            depth_prob += d_kernel(input[n][p][i][3]-t,L,B)/patch_size;
            grad_out[n][1][p][i] += d_depth_prob * depth_prob;
        }
    }
} // namespace

// the forward pass of the entropy layer
// x: input tensor
// region_size: the size of the region
// bandwidth: the bandwidth of the kernel
torch::Tensor entropy_cuda_forward(torch::Tensor x, float bandwidth){
    printf("conditional_entropy_cuda_forward\n");
    cudaError_t cudaStatus;
	cudaEvent_t start, stop;
	cudaEventCreate(&start);
	cudaEventCreate(&stop);
    // the parameters for the kernel function
    const float L=1.0/255.0;
    const float B=bandwidth;
    // get the shape of the input tensor
    // N x P x R x C
    int N = x.size(0);
    int P = x.size(1);
    int R = x.size(2);
    // block size
    dim3 threads(256);
    // grid size
    dim3 grid(P,N);
    // define the output tensor
    // N x 2 x P
    auto entropy_output = torch::zeros({N,2,P}).to(x.device());
    int blockSize;
	int minGridSize;
	cudaOccupancyMaxPotentialBlockSize( &minGridSize, &blockSize, entropy_cuda_forward_kernel<float>); 
	printf("BlockSize: %d\n", blockSize);
	printf("MinGridSize: %d\n", minGridSize);

    printf("input tensor shape: %d %d %d\n",N, P, R);
    printf("grid: %d %d %d\n",grid.x,grid.y,grid.z);
    printf("threads: %d %d\n",threads.x,threads.y);
    cudaEventRecord(start,0);
    cudaFuncSetAttribute(entropy_cuda_forward_kernel<float>,cudaFuncAttributeMaxDynamicSharedMemorySize,65536);
    cudaFuncSetCacheConfig(entropy_cuda_forward_kernel<float>,cudaFuncCachePreferL1);
    // call the kernel
    AT_DISPATCH_FLOATING_TYPES(x.type(),"entropy_cuda_forward",([&]{
        entropy_cuda_forward_kernel<float><<<grid,threads>>>(
            x.packed_accessor32<float,4,torch::RestrictPtrTraits>(),
            entropy_output.packed_accessor32<float,3,torch::RestrictPtrTraits>(),
                L,  B,  R);
    }));
    cudaDeviceSynchronize();

    cudaEventRecord(stop, 0);
	cudaEventSynchronize(stop);
    // Check for any errors launching the kernel
	cudaStatus = cudaGetLastError();
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "kernel launch failed: %s\n", cudaGetErrorString(cudaStatus));
	}

	// cudaDeviceSynchronize waits for the kernel to finish, and returns
	// any errors encountered during the launch.
	cudaStatus = cudaDeviceSynchronize();
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaDeviceSynchronize returned error code %d after launching addKernel!\n", cudaStatus);
	}

	float milliseconds = 0;
	cudaEventElapsedTime(&milliseconds, start, stop);

	std::cout << "GPU rendering required " << milliseconds/1000.0f << "s." << std::endl;

    // return the output
    return entropy_output;
}

// the backward pass of the entropy layer
// x: input tensor
// region_size: the size of the region
// bandwidth: the bandwidth of the kernel
torch::Tensor entropy_cuda_backward(torch::Tensor x,
                            torch::Tensor d_entropy_out,
                            float bandwidth){
    printf("conditional_entropy_cuda_forward\n");
    cudaError_t cudaStatus;
	cudaEvent_t start, stop;
	cudaEventCreate(&start);
	cudaEventCreate(&stop);
    // the parameters for the kernel function
    const float L=1.0/255.0;
    const float B=bandwidth;
    // get the shape of the input tensor
    // N x C x H x W x R 
    int N = x.size(0);
    int P = x.size(1);
    int R = x.size(2);
    // block size
    dim3 threads(256);
    // grid size
    dim3 grid(P,N);
    // define the output tensor (the gradient)
    // N x 2 x P x R 
    auto grad_out = torch::zeros({N,2,P,R}).to(x.device());
    cudaEventRecord(start,0);
    // call the kernel
    AT_DISPATCH_FLOATING_TYPES(x.type(),"entropy_cuda_backward",([&]{
        entropy_cuda_backward_kernel<float><<<grid,threads>>>(
            x.packed_accessor32<float,4,torch::RestrictPtrTraits>(),
            d_entropy_out.packed_accessor32<float,3,torch::RestrictPtrTraits>(),
            grad_out.packed_accessor32<float,4,torch::RestrictPtrTraits>(),
             L,  B,  R);
    }));
    cudaDeviceSynchronize();

    cudaEventRecord(stop, 0);
	cudaEventSynchronize(stop);
    // Check for any errors launching the kernel
	cudaStatus = cudaGetLastError();
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "kernel launch failed: %s\n", cudaGetErrorString(cudaStatus));
	}

	// cudaDeviceSynchronize waits for the kernel to finish, and returns
	// any errors encountered during the launch.
	cudaStatus = cudaDeviceSynchronize();
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaDeviceSynchronize returned error code %d after launching addKernel!\n", cudaStatus);
	}

	float milliseconds = 0;
	cudaEventElapsedTime(&milliseconds, start, stop);

	std::cout << "GPU rendering required " << milliseconds/1000.0f << "s." << std::endl;

    // return the output
    return grad_out;
}