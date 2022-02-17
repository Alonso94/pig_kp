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
    __constant__ int d_C, d_H, d_W;
    // the kernel function to compute the histogram
    template<typename scalar_t>
    __global__ void entropy_cuda_forward_kernel(torch::PackedTensorAccessor32<scalar_t,4,torch::RestrictPtrTraits> input,
                                                torch::PackedTensorAccessor32<scalar_t,4,torch::RestrictPtrTraits> entropy_out,
                                                float L,
                                                float B,
                                                int region_size
                                                ){
        // the image index
        int n = blockIdx.x;
        // x and y coordinates of the center of the patch
        int x_coord = blockIdx.y;
        int y_coord = blockIdx.z;
        // the channel index
        int c = threadIdx.y;
        // the index of the thread
        int t = threadIdx.x;
        // initialize the histogram as zero
        __shared__ scalar_t hist[3][256];
        if (t<255 and c<d_C){
            hist[c][t]=0;
        }
        __syncthreads();
        // the start and the end, row and column of the patch
        int start_row = max(x_coord-region_size/2,0);
        int start_col = max(y_coord-region_size/2,0);
        int end_row = min(x_coord+region_size/2,d_H-1);
        int end_col = min(y_coord+region_size/2,d_W-1);
        // iterate over the patch and compute the histogram
        for (int i=start_row;i<=end_row;i++)
            for (int j=start_col;j<=end_col;j++){
                // the difference between the bin and the pixel value
                float d = input[n][c][i][j]/255.0-bins[t];
                // the kernel value
                float k_val = kernel(d,L,B);
                // update the histogram using atomic add
                atomicAdd(&hist[c][t],k_val);
            }
        __syncthreads();
        // compute the probability of each pixel value
        float p=hist[c][t]/(region_size*region_size);
        float entropy_val = entropy(p);
        // update the entropy using atomic add
        atomicAdd(&entropy_out[n][c][x_coord][y_coord],entropy_val);
        __syncthreads();
    }
    // the kernel function to compute the histogram
    template<typename scalar_t>
    __global__ void entropy_cuda_backward_kernel(torch::PackedTensorAccessor32<scalar_t,4,torch::RestrictPtrTraits> input,
                                                torch::PackedTensorAccessor32<scalar_t,4,torch::RestrictPtrTraits> d_entropy,
                                                torch::PackedTensorAccessor32<scalar_t,4,torch::RestrictPtrTraits> grad_out,
                                                float L,
                                                float B,
                                                int region_size
                                                ){
        // the image index
        int n = blockIdx.x;
        // x and y coordinates of the center of the patch
        int x = blockIdx.y;
        int y = blockIdx.z;
        // the channel index
        int c = threadIdx.y;
        // the index of the thread
        int t = threadIdx.x;
        // initialize the histogram as zero
        __shared__ scalar_t hist[3][256];
        if (t<255 and c<d_C){
            hist[c][t]=0;
        }
        __syncthreads();
        /// the start and the end, row and column of the patch
        int start_row = max(x-region_size/2,0);
        int start_col = max(y-region_size/2,0);
        int end_row = min(x+region_size/2,d_H-1);
        int end_col = min(y+region_size/2,d_W-1);
        // iterate over the patch and compute the histogram
        for (int i=start_row;i<end_row;i++)
            for (int j=start_col;j<end_col;j++){
                // the difference between the bin and the pixel value
                float d = input[n][c][i][j]/255.0-bins[t];
                // the kernel value
                float k_val = kernel(d,L,B);
                // update the histogram using atomic add
                atomicAdd(&hist[c][t],k_val);
            }
        __syncthreads();
        // compute the entropy
        float p=hist[c][t]/(region_size*region_size);
        // define shared tmp gradient value
        __shared__ float tmp_grad;
        tmp_grad=0;
        // compute the gradient of the entropy w.r.t the probability
        float de_p =  -1.0 / log(10.0) * (1.0+log(p));
        // compute the gradient of the probability w.r.t the histogram
        float dp_hist = 1.0/(region_size*region_size);
        // compute the gradient of the histogram w.r.t the pixel value
        // iterate over the pixels in the patch
        for (int i=start_row;i<end_row;i++)
            for (int j=start_col;j<end_col;j++){
                // the difference between the bin and the pixel value
                float d = input[n][c][i][j]/255.0-bins[t];
                // compute the gradient of the kernel w.r.t the pixel value
                float dh_x = d_kernel(d,L,B);
                // update the gradient using atomic add
                atomicAdd(&tmp_grad,de_p*dp_hist*dh_x);
                // update the gradient of the input image using atomic add
                atomicAdd(&grad_out[n][c][i][j],tmp_grad*d_entropy[n][x][y][c]);
            }
        __syncthreads();
    }
} // namespace

// the forward pass of the entropy layer
// x: input tensor
// region_size: the size of the region
// bandwidth: the bandwidth of the kernel
torch::Tensor entropy_cuda_forward(torch::Tensor x, int region_size, float bandwidth){
    // the parameters for the kernel function
    const float L=1.0/255.0;
    const float B=bandwidth;
    const int R = region_size;
    // get the shape of the input tensor
    // N x SF x C x H x W
    int N = x.size(0);
    int SF = x.size(1);
    int C = x.size(2);
    int H = x.size(3);
    int W = x.size(4);
    // copy C,H,W to constant memory
    cudaMemcpyToSymbol(d_C,&C,sizeof(int));
    cudaMemcpyToSymbol(d_H,&H,sizeof(int));
    cudaMemcpyToSymbol(d_W,&W,sizeof(int));
    // reshape the input tensor
    // N*SF x H x W x C
    x = x.reshape({N*SF,C,H,W});
    // ( 255 x C ) threads per block
    const dim3 block_size(255,C);
    // ( N*SF x H x W) blocks per grid
    const dim3 grid_size(N*SF,H,W);
    // define the output tensor (the entropy)
    // N*SF x C x H x W
    auto entropy_out = torch::zeros({N*SF,C,H,W},x.options());
    // initialize the bins
    float h_bins[255];
    for(int i=0;i<255;i++)
        h_bins[i]=i/255.0;
    // copy the bins to constant memory
    cudaMemcpyToSymbol(bins,h_bins,255*sizeof(float),0,cudaMemcpyHostToDevice);
    // call the kernel
    AT_DISPATCH_FLOATING_TYPES(x.type(),"entropy_cuda_forward",([&]{
        entropy_cuda_forward_kernel<scalar_t><<<grid_size,block_size>>>(
            x.packed_accessor32<scalar_t,4,torch::RestrictPtrTraits>(),
            entropy_out.packed_accessor32<scalar_t,4,torch::RestrictPtrTraits>(),
             L,  B,  R );
    }));
    // reshape the output tensor
    // N x SF x C x H x W
    entropy_out = entropy_out.reshape({N,SF,C,H,W});
    // return the output
    return entropy_out;
}

// the backward pass of the entropy layer
// x: input tensor
// region_size: the size of the region
// bandwidth: the bandwidth of the kernel
torch::Tensor entropy_cuda_backward(torch::Tensor x,
                            torch::Tensor d_entropy,
                            int region_size,
                            float bandwidth){
    // the parameters for the kernel function
    const float L=1.0/255.0;
    const float B=bandwidth;
    const int R = region_size;
    // get the shape of the input tensor
    // N x SF x C x H x W
    int N = x.size(0);
    int SF = x.size(1);
    int C = x.size(2);
    int H = x.size(3);
    int W = x.size(4);
    // copy C,H,W to constant memory
    cudaMemcpyToSymbol(d_C,&C,sizeof(int));
    cudaMemcpyToSymbol(d_H,&H,sizeof(int));
    cudaMemcpyToSymbol(d_W,&W,sizeof(int));
    // reshape the input tensor
    // N*SF x C x H x W
    auto x_flat = x.reshape({N*SF,C,H,W});
    // ( 255 x C ) threads per block
    const dim3 block_size(255,C);
    // ( N*SF x H x W) blocks
    const dim3 grid_size(N*SF,H,W);
    // define the output tensor (the gradient)
    // N*SF x C x H x W 
    auto grad_out = torch::zeros({N*SF,C,H,W},x.options());
    // ershape the entropy gradient
    // N*SF x C x H x W
    d_entropy = d_entropy.reshape({N*SF,C,H,W});
    // initialize the bins
    float h_bins[255];
    for(int i=0;i<255;i++)
        h_bins[i]=i/255.0;
    // copy the bins to constant memory
    cudaMemcpyToSymbol(bins,h_bins,255*sizeof(float),0,cudaMemcpyHostToDevice);
    // call the kernel
    AT_DISPATCH_FLOATING_TYPES(x.type(),"entropy_cuda_backward",([&]{
        entropy_cuda_backward_kernel<scalar_t><<<grid_size,block_size>>>(
            x.packed_accessor32<scalar_t,4,torch::RestrictPtrTraits>(),
            d_entropy.packed_accessor32<scalar_t,4,torch::RestrictPtrTraits>(),
            grad_out.packed_accessor32<scalar_t,4,torch::RestrictPtrTraits>(),
             L,  B,  R);
    }));
    // reshape the output tensor
    // N x SF x C x H x W
    grad_out = grad_out.reshape({N,SF,C,H,W});
    // return the output
    return grad_out;
}