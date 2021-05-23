/**
Authors: Christian Henn, Qianli Liao
**/

#include <torch/types.h>

#include <vector>
#include <math.h>
#include <stdio.h>
#include <iostream>
#include <string.h>

// define for error checking
// #define CUDA_ERROR_CHECK

#define CudaCheckError() __cudaCheckError( __FILE__, __LINE__ )
inline void __cudaCheckError( const char *file, const int line )
{
#ifdef CUDA_ERROR_CHECK
    do{
        cudaError err = cudaGetLastError();
        if ( cudaSuccess != err )
        {
            fprintf( stderr, "cudaCheckError() failed at %s:%i : %s\n",
                     file, line, cudaGetErrorString( err ) );
            exit( -1 );
        }

        err = cudaDeviceSynchronize();
        if( cudaSuccess != err )
        {
            fprintf( stderr, "cudaCheckError() with sync failed at %s:%i : %s\n",
                     file, line, cudaGetErrorString( err ) );
            exit( -1 );
        }
    } while(0);
#endif

    return;
}


template <typename scalar_t>
__global__ void stack_kernel(

        const scalar_t* data,
        const int data_size0,
        const int data_size1,

        scalar_t* data_stacked,
        const int data_stacked_size0,
        const int data_stacked_size1,

        const long* ids,

        int* col_write_counts

    ){

    for (int row_in = blockIdx.x * blockDim.x + threadIdx.x; row_in < data_size0; row_in += blockDim.x * gridDim.x)
    {
        auto row_out = ids[row_in];

        auto write_col = atomicAdd(&col_write_counts[row_out], data_size1);

        if (write_col+data_size1 >= data_stacked_size1) continue;

        for (int col_in = 0; col_in < data_size0; col_in++){
            data_stacked[row_out * data_stacked_size1 + write_col + col_in] = data[row_in * data_size1 + col_in];
        }
    }
}







/////////////////////////////////////////////////////
// cpu entry point for frnn python extension.
__host__ std::vector<torch::Tensor> stack_call(
    torch::Tensor data,
    torch::Tensor ids,

    long out_size
) {


    using namespace torch::indexing;
    auto device_id = data.get_device();
    cudaSetDevice(device_id);

    auto i_options = torch::TensorOptions()
            .dtype(torch::kI32)
            .layout(torch::kStrided)
            .device(torch::kCUDA, device_id)
            .requires_grad(false);

    auto out_options = torch::TensorOptions()
            .dtype(data.scalar_type())
            .layout(torch::kStrided)
            .device(torch::kCUDA, device_id)
            .requires_grad(false);

    const int max_items_per_col = 70;
    auto data_stacked = torch::zeros({out_size, max_items_per_col * data.size(1)}, out_options);

    auto col_write_counts = torch::zeros(out_size, i_options);

    cudaDeviceProp deviceProp;
    cudaGetDeviceProperties(&deviceProp, device_id);
    int n_threads = 256;
    int sms = deviceProp.multiProcessorCount;
    int full_cover = (data.size(0)-1) / n_threads + 1;
    int n_blocks = min(full_cover, 2 * sms);

    AT_DISPATCH_ALL_TYPES(data.scalar_type(), "stack_kernel", ([&] {
    stack_kernel<<<n_blocks, n_threads>>>
    (
        data.data_ptr<scalar_t>(),
        data.size(0),
        data.size(1),

        data_stacked.data_ptr<scalar_t>(),
        data_stacked.size(0),
        data_stacked.size(1),

        ids.data_ptr<long>(),

        col_write_counts.data_ptr<int>()

    ); }));

    return {data_stacked};
}



