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
    // initialize the bins as constant memory using linespace
    __constant__ float bins[255];
    // the dimensions of the input image as a constant memory
    __constant__ int d_C, d_H, d_W;
    // the kernel function to compute the histogram
    template<typename scalar_t>
    __global__ void histogram_cuda_forward_kernel(torch::PackedTensorAccessor32<scalar_t,5,torch::RestrictPtrTraits> input,
                                                        torch::PackedTensorAccessor32<scalar_t,5,torch::RestrictPtrTraits> histogram_output,
                                                        float L,
                                                        float B,
                                                        float region_size
                                                        ){
        // the image index
        int n = blockIdx.x;
        // the channel index
        int c = blockIdx.y;
        // the bin index
        int idx = blockIdx.z;
        // the local thread index
        int x = threadIdx.x;
        int y = threadIdx.y;
        // iterate over the region and update the histogram
        for(int i=0; i<d_W; i+=blockDim.x)
            for(int j=0; j<d_H; j+=j+blockDim.y)
                for(int r=0; r<region_size;r++){
                    // the difference between the bin and the pixel value
                    float d = input[n][c][x+i][y+j][r]/256.0-bins[idx];
                    // the kernel value for the marginal histogram
                    float k_val = kernel(d,L,B);
                    // update the marginal histogram with atomicadd
                    atomicAdd(&histogram_output[n][c][x+i][y+j][idx],k_val);
                }
        __syncthreads();
    }
    // the kernel function to compute the histogram
    template<typename scalar_t>
    __global__ void histogram_cuda_backward_kernel(torch::PackedTensorAccessor32<scalar_t,5,torch::RestrictPtrTraits> input,
                                                torch::PackedTensorAccessor32<scalar_t,5,torch::RestrictPtrTraits> d_histogram,
                                                torch::PackedTensorAccessor32<scalar_t,5,torch::RestrictPtrTraits> grad_out,
                                                float L,
                                                float B,
                                                int region_size
                                                ){
        // the image index
        int n = blockIdx.x;
        // the channel index
        int c = blockIdx.y;
        // the bin index
        int idx = blockIdx.z;
        // the local thread index
        int x = threadIdx.x;
        int y = threadIdx.y;
        // iterate over the region and update the histogram
        for(int i=0; i<d_W; i+=blockDim.x)
            for(int j=0; j<d_H; j+=j+blockDim.y)
                for(int r=0; r<region_size;r++){
                    // the difference between the bin and the pixel value
                    float d = input[n][c][x+i][y+j][r]/256.0-bins[idx];
                    // the kernel value for the marginal histogram
                    float d_k_val = d_kernel(d,L,B);
                    // update the marginal histogram with atomicadd
                    atomicAdd(&grad_out[n][c][x+i][y+j][r],d_histogram[n][c][x+i][y+j][idx]*d_k_val);
                }
        __syncthreads();
    }
} // namespace

// the forward pass of the histogram layer
// x: input tensor
// region_size: the size of the region
// bandwidth: the bandwidth of the kernel
torch::Tensor histogram_cuda_forward(torch::Tensor x, float bandwidth){
    printf("conditional_histogram_cuda_forward\n");
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
    int C = x.size(1);
    int H = x.size(2);
    int W = x.size(3);
    int R = x.size(4);
    // copy C,H,W to constant memory
    cudaMemcpyToSymbol(d_C,&C,sizeof(int));
    cudaMemcpyToSymbol(d_H,&H,sizeof(int));
    cudaMemcpyToSymbol(d_W,&W,sizeof(int));
    // block size
    dim3 threads(32,32);
    // grid size
    dim3 grid(N,C,256);
    // define the output tensor
    // N x C x H x W
    auto histogram_output = torch::zeros({N,C,H,W,256}).to(x.device());
    // initialize the bins
    float h_bins[256];
    for(int i=0;i<256;i++)
        h_bins[i]=i/256.0;
    // copy the bins to constant memory
    cudaMemcpyToSymbol(bins,h_bins,256*sizeof(float),0,cudaMemcpyHostToDevice);
    printf("grid: %d %d %d\n",grid.x,grid.y,grid.z);
    printf("threads: %d %d\n",threads.x,threads.y);
    cudaEventRecord(start,0);
    // call the kernel
    AT_DISPATCH_FLOATING_TYPES(x.type(),"histogram_cuda_forward",([&]{
        histogram_cuda_forward_kernel<scalar_t><<<grid,threads>>>(
            x.packed_accessor32<scalar_t,5,torch::RestrictPtrTraits>(),
            histogram_output.packed_accessor32<scalar_t,5,torch::RestrictPtrTraits>(),
             L,  B,  R );
    }));
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
    return histogram_output;
}

// the backward pass of the histogram layer
// x: input tensor
// region_size: the size of the region
// bandwidth: the bandwidth of the kernel
torch::Tensor histogram_cuda_backward(torch::Tensor x,
                            torch::Tensor d_histogram,
                            float bandwidth){
    // the parameters for the kernel function
    const float L=1.0/255.0;
    const float B=bandwidth;
    // get the shape of the input tensor
    // N x C x H x W x R 
    int N = x.size(0);
    int C = x.size(1);
    int H = x.size(2);
    int W = x.size(3);
    int R = x.size(4);
    // copy C,H,W to constant memory
    cudaMemcpyToSymbol(d_C,&C,sizeof(int));
    cudaMemcpyToSymbol(d_H,&H,sizeof(int));
    cudaMemcpyToSymbol(d_W,&W,sizeof(int));
    // ( 255 x C ) threads per block
    const dim3 block_size(255,C);
    // ( N*SF x H x W) blocks
    const dim3 grid_size(N,H,W);
    // define the output tensor (the gradient)
    // N*SF x C x H x W 
    auto grad_out = torch::zeros({N,C,H,W},x.options());
    // ershape the histogram gradient
    // N*SF x C x H x W
    d_histogram = d_histogram.reshape({N,C,H,W});
    // initialize the bins
    float h_bins[255];
    for(int i=0;i<255;i++)
        h_bins[i]=i/255.0;
    // copy the bins to constant memory
    cudaMemcpyToSymbol(bins,h_bins,255*sizeof(float),0,cudaMemcpyHostToDevice);
    // call the kernel
    AT_DISPATCH_FLOATING_TYPES(x.type(),"histogram_cuda_backward",([&]{
        histogram_cuda_backward_kernel<scalar_t><<<grid_size,block_size>>>(
            x.packed_accessor32<scalar_t,5,torch::RestrictPtrTraits>(),
            d_histogram.packed_accessor32<scalar_t,5,torch::RestrictPtrTraits>(),
            grad_out.packed_accessor32<scalar_t,5,torch::RestrictPtrTraits>(),
             L,  B,  R);
    }));
    // return the output
    return grad_out;
}