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
    // initialize the bins as constant memory using linespace
    __constant__ float bins[255];
    // the dimensions of the input image as a constant memory
    __constant__ int d_P, d_R, d_N, d_C, d_H, d_W;
    // the kernel function to compute the histogram
    template<typename scalar_t>
    __global__ void entropy_cuda_forward_kernel(torch::PackedTensorAccessor32<scalar_t,3,torch::RestrictPtrTraits> input,
                                                        torch::PackedTensorAccessor32<scalar_t,2,torch::RestrictPtrTraits> entropy_output,
                                                        float L,
                                                        float B,
                                                        int patch_size,
                                                        int num_patches
                                                        ){
        // the image index
        int n = blockIdx.y;
        // the index of the group of patches
        int p = blockIdx.x;
        // the thread index
        int tx = threadIdx.x;
        int ty = threadIdx.y;
        // global index
        int gt = ty*256+tx;
        // the shared memory for the histogram
        extern __shared__ float hist[];
        // the array to store the entropy_vals
        float* entropy_vals = (float*)&hist[num_patches*256];
        // the array to store the data
        float* data = (float*)&entropy_vals[num_patches];
        // load the data into the shared memory
        if(gt<num_patches*patch_size){
            data[gt] = input[n][p*num_patches+gt/patch_size][gt%patch_size];
        }
        if(gt<num_patches) entropy_vals[gt]=0;
        __syncthreads();
        for(int i=0;i<num_patches;i++){
            hist[gt] = 0;
            for(int j=0;j<patch_size;++j){
                // the difference between the pixel value and the bin
                float d = data[i*patch_size+j]-bins[tx];
                // update the histogram
                hist[gt]+=kernel(d,L,B)/patch_size;
            }
            // print the time
            
            atomicAdd(&entropy_vals[i],entropy(hist[gt]));
        }
        __syncthreads();
        // update the entropy output
        if(gt<num_patches){
            entropy_output[n][p*num_patches+gt]=entropy_vals[gt];
        }
    }
    // the kernel function to compute the histogram
    template<typename scalar_t>
    __global__ void entropy_cuda_backward_kernel(torch::PackedTensorAccessor32<scalar_t,5,torch::RestrictPtrTraits> input,
                                                torch::PackedTensorAccessor32<scalar_t,4,torch::RestrictPtrTraits> d_entropy,
                                                torch::PackedTensorAccessor32<scalar_t,4,torch::RestrictPtrTraits> grad_out,
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
        // define the shared memory for the histogram
        __shared__ scalar_t hist[6][256];
        // initialize the shared memory with 0
        hist[c][idx]=0;
        __syncthreads();
        // iterate over the region and update the histogram
        for(int i=0; i<d_H; i+=blockDim.x)
            for(int j=0; j<d_W; j+=j+blockDim.y)
                for(int r=0; r<region_size;r++){
                    // the difference between the bin and the pixel value
                    float d = input[n][c][x+i][y+j][r]/256.0-bins[idx];
                    // the kernel value for the marginal histogram
                    float k_val = kernel(d,L,B);
                    // update the marginal histogram with atomicadd
                    atomicAdd(&hist[c][idx],k_val);
                }
        __syncthreads();
        // compute the probability of each pixel value
        float p=hist[c][idx]/(region_size);
        float entropy_val = entropy(p);
        // update the entropy using atomic add
        atomicAdd(&d_entropy[n][c][x][y],entropy_val);
        __syncthreads();
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
    // N x C x H x W x R 
    int N = x.size(0);
    int P = x.size(1);
    int R = x.size(2);
    // move P to the constant memory
    cudaMemcpyToSymbol(d_P,&P, sizeof(int));
    cudaMemcpyToSymbol(d_R,&R, sizeof(int));
    // block size
    dim3 threads(256,1);
    // number of patches in each sweep
    int num_patches = (256)/R;
    // grid size
    dim3 grid((P+num_patches-1)/num_patches,N);
    // shared memory size
    size_t shared_mem = (R+258)*num_patches*sizeof(float);
    // define the output tensor
    // N x C x H x W
    auto entropy_output = torch::zeros({N,P}).to(x.device());
    // initialize the bins
    float h_bins[256];
    for(int i=0;i<256;i++)
        h_bins[i]=i;
    // copy the bins to constant memory
    cudaMemcpyToSymbol(bins,h_bins,256*sizeof(float));
    int blockSize;
	int minGridSize;
	cudaOccupancyMaxPotentialBlockSize( &minGridSize, &blockSize, entropy_cuda_forward_kernel<float>,shared_mem); 
	printf("BlockSize: %d\n", blockSize);
	printf("MinGridSize: %d\n", minGridSize);

    printf("input tensor shape: %d %d %d\n",N, P, R);
    printf("grid: %d %d %d\n",grid.x,grid.y,grid.z);
    printf("threads: %d %d\n",threads.x,threads.y);
    printf("shared_mem: %d\n",shared_mem);
    printf("num_patches: %d\n",num_patches);
    printf("entropy vals: %d\n",num_patches);
    printf("histogram size: %d\n",256*num_patches);
    printf("patch array size: %d\n",num_patches*R);
    cudaEventRecord(start,0);
    cudaFuncSetAttribute(entropy_cuda_forward_kernel<float>,cudaFuncAttributeMaxDynamicSharedMemorySize,98304);
    // call the kernel
    AT_DISPATCH_FLOATING_TYPES(x.type(),"entropy_cuda_forward",([&]{
        entropy_cuda_forward_kernel<scalar_t><<<grid,threads, shared_mem>>>(
            x.packed_accessor32<scalar_t,3,torch::RestrictPtrTraits>(),
            entropy_output.packed_accessor32<scalar_t,2,torch::RestrictPtrTraits>(),
             L,  B,  R , num_patches);
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
                            torch::Tensor d_entropy,
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
    // reshape the input tensor
    // N*SF x C x H x W
    auto x_flat = x.reshape({N,C,H,W});
    // ( 255 x C ) threads per block
    const dim3 block_size(255,C);
    // ( N*SF x H x W) blocks
    const dim3 grid_size(N,H,W);
    // define the output tensor (the gradient)
    // N*SF x C x H x W 
    auto grad_out = torch::zeros({N,C,H,W},x.options());
    // ershape the entropy gradient
    // N*SF x C x H x W
    d_entropy = d_entropy.reshape({N,C,H,W});
    // initialize the bins
    float h_bins[255];
    for(int i=0;i<255;i++)
        h_bins[i]=i/255.0;
    // copy the bins to constant memory
    cudaMemcpyToSymbol(bins,h_bins,255*sizeof(float),0,cudaMemcpyHostToDevice);
    // call the kernel
    AT_DISPATCH_FLOATING_TYPES(x.type(),"entropy_cuda_backward",([&]{
        entropy_cuda_backward_kernel<scalar_t><<<grid_size,block_size>>>(
            x.packed_accessor32<scalar_t,5,torch::RestrictPtrTraits>(),
            d_entropy.packed_accessor32<scalar_t,4,torch::RestrictPtrTraits>(),
            grad_out.packed_accessor32<scalar_t,4,torch::RestrictPtrTraits>(),
             L,  B,  R);
    }));
    // return the output
    return grad_out;
}