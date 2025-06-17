#pragma once
#include<cublas_v2.h>
#include<cuda_runtime.h>
#include<cuda_fp16.h>

class cublasWrapper{
public:
    cublasWrapper(cublasHandle_t handle);
    ~cublasWrapper();
    cublasWrapper(const cublasWrapper&) = delete;
    cublasWrapper& operator=(const cublasWrapper&) = delete;

    void sgemm(cublasOperation_t transa, cublasOperation_t transb, int m, int n, int k, float alpha, const float *A, int lda, const float *B, int ldb, float beta, float *C, int ldc);
    void hgemm(cublasOperation_t transa, cublasOperation_t transb, int m, int n, int k, 
               const __half *A, int lda, const __half *B, int ldb, 
                __half *C, int ldc, float alpha, float beta);

    void sgemm_strided_batched(cublasOperation_t transa, cublasOperation_t transb, int m, int n, int k,
                               const float* A, int lda, long long int strideA,
                               const float* B, int ldb, long long int strideB,
                               float* C, int ldc, long long int strideC,
                               float alpha, float beta, int batchCount);
    void hgemm_strided_batched(cublasOperation_t transa, cublasOperation_t transb, int m, int n, int k,
                               const __half* A, int lda, long long int strideA,
                               const __half* B, int ldb, long long int strideB,
                               __half* C, int ldc, long long int strideC,
                               float alpha, float beta, int batchCount);
                               
private:
    cublasHandle_t handle_;
};


