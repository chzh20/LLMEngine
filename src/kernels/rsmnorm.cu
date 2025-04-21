#include"cuda_runtime.h"
#include"rsmnorm.h"
#include"src/utils/vec_utils.h"


template<typename T>
__device__ T warpReduceSum(T val)
{
    for(int i = 32/2;i>0; i>>=1)
    {
        val +=__shfl_xor_sync(0xffffffff,val,i);
    }
    return val;
}

template<typename T>
__device__ T blockReduceSum(T val)
{
    int tid = threadIdx.x;
    int wid = tid/32;
    int laneid = tid %32;
    int warpnums = (blockDim.x+32-1)/32;
    static __shared__ T warpSems[32];
    val = warpReduceSum<T>(val);
    if(laneid == 0)
        warpSems[wid] = val;
    __syncthreads();
    T sum = tid<warpnums? warpSems[tid]:(T)0;
    sum = warpReduceSum(sum);
    return sum;
}
template<typename T>
T Pow2(T val)
{
    return val*val;
}

template<typename T>
__global__ void RMSNorm(
    T* decoder_out,//[num tokens, q_hidden_units]
    T* decoder_residual,
    T* scale,//[q_hidden_units]
    float esp,
    int num_tokens,
    int hidden_size
)
{
    // 1. vectorize loading
    int vec_size = Vec<T>::size;
    using VecType = typename Vec<T>::type;
    // each block process one token which have q_hidden_unitsc cols
    VecType* d_out = reinterpret_cast<VecType*>(decoder_out+blockIdx.x * hidden_size );
    // rsm = sum(x*x)/n+ esp
    // y = x/rsm(x)
    VecType * rsd;
    rsd = reinterpret_cast<VecType*>(decoder_residual + blockIdx.x * hidden_size);
    float thread_sum = 0.0f;
    for(int idx= threadIdx.x ; idx<hidden_size/vec_size; idx+= blockDim.x)
    {
        VecType vec = d_out[idx];
        rsd[idx] = vec;
        thread_sum += Pow2(vec.x);
        thread_sum += Pow2(vec.y);
        thread_sum += Pow2(vec.z);
        thread_sum += Pow2(vec.w);
    }
    thread_sum = blockReduceSum<float>(thread_sum);
    __shared__  float inv_mean;
    if(threadIdx.x == 0){
        inv_mean = rsqrtf((float)thread_sum/hidden_size+eps);
    }
    __syncthreads();
    VecType* s = reinterpret_cast<VecType*>(scale);
    for(int idx = threadIdx.x; idx < hidden_size/vec_size; idx += blockDim.x) {
        VecType out = d_out[idx];
        d_out[idx].x = out.x* inv_mean + s[idx].x;
        d_out[idx].y = out.y* inv_mean + s[idx].y;
        d_out[idx].z = out.z* inv_mean + s[idx].z;
        d_out[idx].w = out.w* inv_mean + s[idx].w;
    }






}


template<>
__global__ void RMSNorm<half>(
    half* decoder_out,//[num tokens, q_hidden_units]
    half* decoder_residual,
    half* scale,//[q_hidden_units]
    float eps,
    int num_tokens,
    int hidden_size
)
{
    int vec_size = Vec<half>::size;
    using VecType = Vec<half>::type;
    VecType* d_out = reinterpret_cast<VecType*>(decoder_out+blockIdx.x * hidden_size );
    // rsm = sum(x*x)/n+ esp
    // y = x/rsm(x)
    VecType * rsd;
    rsd = reinterpret_cast<VecType*>(decoder_residual + blockIdx.x * hidden_size);
    float thread_sum = 0.0f;
    for(int idx= threadIdx.x ; idx<hidden_size/vec_size; idx+= blockDim.x)
    {
        VecType vec = d_out[idx];
        rsd[idx] = vec;
        thread_sum += __half2float(vec.x) * __half2float(vec.x);
        thread_sum += __half2float(vec.y) * __half2float(vec.y);
    }

    thread_sum = blockReduceSum<float>(thread_sum);
    __shared__  float inv_mean;
    if(threadIdx.x == 0){
        inv_mean = rsqrtf(float(thread_sum/hidden_size)+eps);
    }
    __syncthreads();
    VecType* s = reinterpret_cast<VecType*>(scale);
    for(int idx = threadIdx.x; idx < hidden_size/vec_size; idx += blockDim.x) {
        VecType out = d_out[idx];
        d_out[idx].x = __float2half(__half2float(out.x)*inv_mean) * s[idx].x;
        d_out[idx].y = __float2half(__half2float(out.y)*inv_mean) * s[idx].y;
    }
}



template<typename T>
void lauchRMSNorm(
    Tensor<T>* decoder_out,
    Tensor<T>* decoder_residual,
    LayerNormWeight<T>& attn_norm_weight,
    float eps,
    bool is_last = false
)
{
    int num_tokens = decoder_out->shape[0];
    int hidden_size = decoder_out->shape[1];
    int vec_size = Vec<T>::size;
    int num_threads = hidden_size/vec_size;
    T* rsd = decoder_residual->data;
    dim3 gird(num_tokens);
    dim3 block(num_threads);
    RMSNorm<<<gird,block>>> (decoder_out->data,
                            rsd,
                            attn_norm_weight.gamma,
                            eps,
                            num_tokens,
                            hidden_size);

}


template<>
void lauchRMSNorm(
    Tensor<float>* decoder_out,
    Tensor<float>* decoder_residual,
    LayerNormWeight<float>& attn_norm_weight,
    float eps,
    bool is_last
);


template<> 
void lauchRMSNorm(
    Tensor<half>* decoder_out,
    Tensor<half>* decoder_residual,
    LayerNormWeight<half>& attn_norm_weight,
    float eps,
    bool is_last
);



