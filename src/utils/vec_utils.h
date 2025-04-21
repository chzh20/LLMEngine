#pragma once
#include"cuda_runtime.h"
#include"cuda_fp16.h"



template<typename T>
struct Vec
{
    using type= T;
    static constexpr int size = 0;
};

template<>
struct Vec<float>
{
    using type = float4;
    static constexpr int size = 4;
};

template<>
struct Vec<half>
{
    using type = half2;
    static constexpr int size = 2;
};

