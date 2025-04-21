#pragma onnce
#include"src/utils/tensor.h"
#include"src/weights/llama/norm_weight.h"

template<typename T>
void lauchRMSNorm(
    Tensor<T>* decoder_out,
    Tensor<T>* decoder_residual,
    LayerNormWeight<T>& attn_norm_weight,
    float eps,
    bool is_last = false
);
