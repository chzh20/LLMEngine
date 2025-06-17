#pragma once
#include<cuda_runtime.h>
#include<cuda.h>
#include"src/kernels/cublas_util.h"
#include"src/weights/base_weights.h"
#include"src/utils/tensor.h"
#include"src/utils/macro.h"



template<typename T>
void launchLinearGemm(
    Tensor<T> *input,
    BaseWeight<T> *weight,
    Tensor<T> *output,
    cublasWrapper *cublas_wrapper,
    bool trans_a = false,
    bool trans_b = false
);


template<typename T>
void launchLinearStridedBatchedGemm(
    Tensor<T> *input,
    BaseWeight<T> *weight,
    Tensor<T> *output,
    cublasWrapper *cublas_wrapper,
    bool trans_a = false,
    bool trans_b = false
);






