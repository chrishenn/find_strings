#include <torch/script.h>

#include <vector>
#include <iostream>
#include <string>

#define CHECK_CUDA(x) AT_ASSERTM(x.is_cuda(), #x " must be a CUDA tensor")
#define CHECK_CONTIGUOUS(x) AT_ASSERTM(x.is_contiguous(), #x " must be contiguous")
#define CHECK_INPUT(x) CHECK_CONTIGUOUS(x)


std::vector<torch::Tensor> frnn_ts_call(
        torch::Tensor pts,
        torch::Tensor imgid,

        torch::Tensor lin_radius,
        torch::Tensor scale_radius,

        torch::Tensor pair_ids,
        torch::Tensor tree_table,

        torch::Tensor batch_size
    );


std::vector<torch::Tensor> fbt(
        torch::Tensor pts,
        torch::Tensor imgid,

        torch::Tensor lin_radius,
        torch::Tensor scale_radius,

        torch::Tensor pair_ids,
        torch::Tensor tree_table,

        torch::Tensor batch_size
    )
{

    CHECK_INPUT(pts);
    CHECK_INPUT(imgid);

    CHECK_INPUT(lin_radius);
    CHECK_INPUT(scale_radius);

    CHECK_INPUT(pair_ids);

    CHECK_INPUT(batch_size);

    return frnn_ts_call(
            pts,
            imgid,

            lin_radius,
            scale_radius,

            pair_ids,
            tree_table,

            batch_size
    );
}

TORCH_LIBRARY(fbt_op, m) {
    m.def("fbt_kern", fbt);
}