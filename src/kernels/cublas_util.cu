#include"linear.h"
#include <cuda_fp16.h>


cublasWrapper::cublasWrapper(cublasHandle_t handle)
{
    cublasCreate(&handle_);
}

void cublasWrapper::sgemm(cublasOperation_t transa, cublasOperation_t transb, int m, int n, int k, float alpha, const float *A, int lda, const float *B, int ldb, float beta, float *C, int ldc)
{
    cublasSgemm(handle_, transa, transb, m, n, k, &alpha, A, lda, B, ldb, &beta, C, ldc);
}

void cublasWrapper::hgemm(cublasOperation_t transa, cublasOperation_t transb, int m, int n, int k, 
                              const __half *A, int lda, const __half *B, int ldb, 
                               __half *C, int ldc, float alpha, float beta) {
    cublasGemmEx(handle_, transa, transb, m, n, k, 
                 &alpha, A, CUDA_R_16F, lda, 
                 B, CUDA_R_16F, ldb, 
                 &beta, C, CUDA_R_16F, ldc, 
                 CUDA_R_16F, CUBLAS_GEMM_DEFAULT);
}

void cublasWrapper::sgemm_strided_batched(
    cublasOperation_t transa, cublasOperation_t transb,
    int m, int n, int k,
    const float* A, int lda, long long int strideA,
    const float* B, int ldb, long long int strideB,
    float* C, int ldc, long long int strideC,
    float alpha, float beta, int batchCount)
{
    cublasGemmStridedBatchedEx(
        handle_, transa, transb,
        m, n, k,
        &alpha,
        A, CUDA_R_32F, lda, strideA,
        B, CUDA_R_32F, ldb, strideB,
        &beta,
        C, CUDA_R_32F, ldc, strideC,
        batchCount,
        CUDA_R_32F,
        CUBLAS_GEMM_DEFAULT);
}

void cublasWrapper::hgemm_strided_batched(
    cublasOperation_t transa, cublasOperation_t transb,
    int m, int n, int k,
    const __half* A, int lda, long long int strideA,
    const __half* B, int ldb, long long int strideB,
    __half* C, int ldc, long long int strideC,
    float alpha, float beta, int batchCount)
{
    cublasGemmStridedBatchedEx(
        handle_, transa, transb,
        m, n, k,
        &alpha,
        A, CUDA_R_16F, lda, strideA,
        B, CUDA_R_16F, ldb, strideB,
        &beta,
        C, CUDA_R_16F, ldc, strideC,
        batchCount,
        CUDA_R_16F,
        CUBLAS_GEMM_DEFAULT);
}

cublasWrapper::~cublasWrapper()
{
    cublasDestroy(handle_);
}



