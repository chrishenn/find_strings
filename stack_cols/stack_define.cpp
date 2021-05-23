#include <torch/script.h>

#include <vector>
#include <iostream>
#include <string>

#define CHECK_CUDA(x) AT_ASSERTM(x.is_cuda(), #x " must be a CUDA tensor")
#define CHECK_CONTIGUOUS(x) AT_ASSERTM(x.is_contiguous(), #x " must be contiguous")
#define CHECK_INPUT(x) CHECK_CUDA(x); CHECK_CONTIGUOUS(x)


std::vector<torch::Tensor> stack_call(
        torch::Tensor data,
        torch::Tensor ids,

        long out_size
    );

std::vector<torch::Tensor> stack_cols(
        torch::Tensor data,
        torch::Tensor ids,

        long out_size
    )
{
    CHECK_INPUT(data);
    CHECK_INPUT(ids);

    return stack_call(
        data,
        ids,

        out_size
    );
}

TORCH_LIBRARY(stack_op, m) {
    m.def("stack_bind", stack_cols);
}