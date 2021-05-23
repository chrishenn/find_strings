/**
Authors: Christian Henn, Qianli Liao

Implements the Fixed-Radius Nearest-Neighbor (frnn) algorithm [Qianli Liao and David Walter]. Parallel exclusive scan code adapted from [Matt Dean - 1422434 - mxd434].

This file relies on Pytorch calls from the Pytorch C++ API.

 NOTE:
 cuda __global__ kernels can be included from a .cu file by adding a declaration in this file, where the call is. The kernels will be compiled and found no
 problem (as long as the names are unique?). HOWEVER, templated __global__ kernels CANNOT be included in this way - the kernel call with the AT_DISPATCH macro
 and the kernel it's calling MUST be in the same file, or else python will throw an "unknown symbol" error.
**/

#include <torch/types.h>

//#include <cuda.h>
//#include <cuda_runtime.h>
//#include <assert.h>

#include <vector>
#include <math.h>
#include <stdio.h>
#include <iostream>
#include <string.h>

// define for error checking
 #define CUDA_ERROR_CHECK

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

// timing helper
double get_nanos() {
	struct timespec ts;
	timespec_get(&ts, TIME_UTC);
	return (double)ts.tv_nsec;
}



template <typename scalar_t>
__global__ void frnn_brute_lin_kernel(
        const scalar_t* pts,
        const int       pts_size0,
        const int       pts_size1,

        const int* lookup_table,
        const int   lookup_size0,
        const int   lookup_size1,
        const int* col_counts,

              long* edges,
              int*  glob_count,

        const float lin_radius,
        const float scale_radius
){
    int imid = blockIdx.x;
    int row_end = col_counts[imid];

    for (int row_a = threadIdx.x; row_a < row_end-1; row_a += blockDim.x)
    {
        for (int row_b = row_a + 1; row_b < row_end; row_b++)
        {
            int ptid_a = lookup_table[row_a * lookup_size1 + imid];
            int ptid_b = lookup_table[row_b * lookup_size1 + imid];

            float ay = float( pts[ptid_a * pts_size1 + 0] );
            float ax = float( pts[ptid_a * pts_size1 + 1] );

            float by = float( pts[ptid_b * pts_size1 + 0] );
            float bx = float( pts[ptid_b * pts_size1 + 1] );

            float diffy = by - ay;
            float diffx = bx - ax;

            float dist_cc = sqrtf( diffx*diffx + diffy*diffy );
//            float dist = dist_cc - (as / 2) - (bs / 2);

//            bool check = (dist < lin_radius);
            bool check = (dist_cc < lin_radius);

            if (check)
            {
                int thread_i = atomicAdd(glob_count, 2);
                edges[thread_i + 0] = long(ptid_a);
                edges[thread_i + 1] = long(ptid_b);
            }
        }
    }
}

__global__ void build_lookup(
        const long* imgid,
        const int   imgid_size0,

              int*  lookup_table,
        const int   lookup_size0,
        const int   lookup_size1,

              int*  col_counts
){
    for (int o_row = blockIdx.x*blockDim.x + threadIdx.x; o_row < imgid_size0; o_row += blockDim.x * gridDim.x)
    {
        auto im_id = imgid[o_row];

        auto lookup_row = atomicAdd( &col_counts[im_id], 1);

        lookup_table[lookup_row * lookup_size1 + im_id] = o_row;
    }
}





/////////////////////////////////////////////////////
// cpu entry point for frnn python extension.
__host__ std::vector<torch::Tensor> frnn_ts_call(
    torch::Tensor pts,
    torch::Tensor imgid,

    torch::Tensor lin_radius,
    torch::Tensor scale_radius,

    torch::Tensor batch_size
) {

    using namespace torch::indexing;
    auto device_id = pts.get_device();
    cudaSetDevice(device_id);

    // Device data options
    auto i_options = torch::TensorOptions()
            .dtype(torch::kI32)
            .layout(torch::kStrided)
            .device(torch::kCUDA, device_id)
            .requires_grad(false);

    auto l_options = torch::TensorOptions()
            .dtype(torch::kI64)
            .layout(torch::kStrided)
            .device(torch::kCUDA, device_id)
            .requires_grad(false);

    // TODO: add dynamic block allocation per sms present: unroll on block work in kernels
    // Setup device sizes
    const dim3 blocks(batch_size.item<int>());
    const dim3 threads(256);

    // calculate output size
    auto intermed = (lin_radius * 4 + 1).to(torch::kInt32);
    auto area = intermed.pow(2).to(torch::kFloat32);
    auto n_average = torch::true_divide(torch::full({1}, pts.size(0), i_options), batch_size);
    auto hyp = (area * n_average * batch_size * 21 * scale_radius).to(torch::kI32);
    auto sq_max = (n_average.pow(2) * batch_size * 1.8).to(torch::kI32);

    auto edges_size0 = torch::min(hyp, sq_max);
    auto edges_size1 = 2;

    auto edges = torch::empty({edges_size0.item<int>(), edges_size1}, l_options);
    auto glob_count = torch::zeros({1}, i_options);

    auto im_counts = imgid.bincount();
    auto lookup_size0 = im_counts.max().item<int>();

    // lookup table: each col gives an image's worth of ptids
    auto lookup_table = torch::empty({lookup_size0, batch_size.item<int>()}, i_options);
    auto col_counts = torch::zeros(batch_size.item<int>(), i_options);

    build_lookup<<<blocks, threads>>>(
            imgid.data_ptr<long>(),
            imgid.size(0),

            lookup_table.data_ptr<int>(),
            lookup_size0,
            lookup_table.size(1),

            col_counts.data_ptr<int>()
    ); CudaCheckError();

    AT_DISPATCH_FLOATING_TYPES_AND(torch::ScalarType::Half, pts.scalar_type(), "frnn_brute_kernel", ([&] {
        frnn_brute_lin_kernel<<<blocks, threads>>>(
            pts.data_ptr<scalar_t>(),
            pts.size(0),
            pts.size(1),

            lookup_table.data_ptr<int>(),
            lookup_size0,
            lookup_table.size(1),
            col_counts.data_ptr<int>(),

            edges.data_ptr<long>(),
            glob_count.data_ptr<int>(),

            lin_radius.item<float>(),
            scale_radius.item<float>()

    ); })); CudaCheckError();

    auto edge_count = glob_count.floor_divide(2);
    if ( edge_count.gt(edges_size0).item<int>() ){

        fprintf (stderr, "ERROR frnn_driver.cu: FRNN_MAIN_KERNEL ATTEMPTED TO WRITE MORE EDGES THAN SPACE WAS ALLOCATED FOR\n");
        fprintf (stderr, "attemped: %i edges; allocated: %i edges\n", edge_count.item<int>(), edges_size0.item<int>());

        std::string err_mode;
        if (sq_max.item<int>() < hyp.item<int>())
            err_mode = "sq_max";
        else err_mode = "hyp";
        fprintf (stderr, "error mode: %s\n", err_mode.c_str());
        exit(EXIT_FAILURE);
    }

    edges = edges.narrow(0, 0, edge_count.item<int>());
    return {edges};
}



