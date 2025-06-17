#include"linear.h"
#include "cublas_util.h"
#include <cuda_fp16.h>

template<typename T>
void launchLinearGemm(
    Tensor<T> *input,
    BaseWeight<T> *weight,
    Tensor<T> *output,
    cublasWrapper *cublas_wrapper,
    bool trans_a = false,
    bool trans_b = false
){
    //y=x*W^T
    // 获取输入输出维度
    // input: [m,k]
    // weight: [k,n]
    // output: [m,n]
    //trans_b = true 表示转置权重矩阵 y^T = W*X^T -> y = x*W^T
    //trans_b = false 表示不转置权重矩阵 y^T = W^T*X^T -> y = x*W
    int m = input->shape[0];  // batch size[m,k]
    int k = input->shape[1];  // input dimension
    int n = weight->shape[1]; // output dimension
    
    // 设置矩阵乘法参数
    float alpha = 1.0f;
    float beta = 0.0f;
    
    // 对于行主序数据，需要转置权重矩阵
    // 输入矩阵: [m, k] (row major)
    // 权重矩阵: [k, n] (row major)
    // 输出矩阵: [m, n] (row major)
    // 在cuBLAS中计算: C = A * B^T
    if constexpr (std::is_same_v<T, float>) {
        cublas_wrapper->sgemm(
            CUBLAS_OP_N,  // 不转置输入矩阵
            CUBLAS_OP_T,  // 转置权重矩阵
            m, n, k,
            alpha,
            input->data, k,    // lda = k (输入矩阵的列数)
            weight->data, k,   // ldb = k (权重矩阵的列数)
            beta,
            output->data, n    // ldc = n (输出矩阵的列数)
        );
    }
    else if constexpr (std::is_same_v<T, half>) {
        cublas_wrapper->hgemm(
            CUBLAS_OP_N,  // 不转置输入矩阵
            CUBLAS_OP_T,  // 转置权重矩阵
            m, n, k,
            input->data, k,    // lda = k
            weight->data, k,   // ldb = k
            output->data, n,   // ldc = n
            alpha, beta
        );
    }
    
}