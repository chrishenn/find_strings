#include <torch/script.h>

#include <vector>
#include <iostream>
#include <string>

std::vector<torch::Tensor> frnn_ts_call(
        torch::Tensor pts,
        torch::Tensor imgid,

        torch::Tensor lin_radius,
        torch::Tensor scale_radius,

        torch::Tensor batch_size
    );

#define CHECK_CUDA(x) AT_ASSERTM(x.is_cuda(), #x " must be a CUDA tensor")
#define CHECK_CONTIGUOUS(x) AT_ASSERTM(x.is_contiguous(), #x " must be contiguous")
#define CHECK_INPUT(x) CHECK_CONTIGUOUS(x)

std::vector<torch::Tensor> frnn_ts(
        torch::Tensor pts,
        torch::Tensor imgid,

        torch::Tensor lin_radius,
        torch::Tensor scale_radius,

        torch::Tensor batch_size
    )
{

    CHECK_INPUT(pts);
    CHECK_INPUT(imgid);
    CHECK_INPUT(batch_size);
    CHECK_INPUT(lin_radius);
    CHECK_INPUT(scale_radius);

    return frnn_ts_call(
        pts,
        imgid,

        lin_radius,
        scale_radius,

        batch_size
    );
}

TORCH_LIBRARY(my_ops, m) {
    m.def("frnn_ts_kernel", frnn_ts);
}