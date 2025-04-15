#pragma once
#include<vector>
#include<cstdint>
#include<cuda_fp16.h>

enum class WeightType
{
    FP32_W = 0,
    FP16_W = 1,
    INT8_W = 2,
    INT4_W = 3,
    BF16_W = 4,
    UNDEFINED_W = 5,
};

template<typename T>
inline WeightType getWeightType()
{
    if constexpr (std::is_same_v<T, float> || std::is_same_v<T, const float>)
    {
        return WeightType::FP32_W;
    }
    else if constexpr (std::is_same_v<T, half> || std::is_same_v<T, const half>)
    {
        return WeightType::FP16_W;
    }
    else if constexpr (std::is_same_v<T, __half> || std::is_same_v<T, const __half>)
    {
        return WeightType::FP16_W;
    }
    else if constexpr (std::is_same_v<T, int8_t> || std::is_same_v<T const int8_t>)
    {
        return WeightType::INT8_W;
    }
    else if constexpr (std::is_same_v<T, int4_t> || std::is_same_v<T,const int4_t>)
    {
        return WeightType::INT4_W;
    }
    else if constexpr (std::is_same_v<T, __nv_bfloat16>)
    {
        return WeightType::BF16_W;
    }
    else
    {
        return WeightType::UNDEFINED_W;
    }
}

template<typename T>
struct BaseWeight
{
    WeightType type;
    std::vector<int> shape;
    T* data = nullptr;
    T* bias = nullptr;
};