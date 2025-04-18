#pragma once
#include <iostream>
#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#define CUDA_CHECK(call) { \
    cudaError_t err = call; \
    if (err != cudaSuccess) { \
        std::cerr << "CUDA error in " << __FILE__ << " at line " << __LINE__ << ": " \
                  << cudaGetErrorString(err) << std::endl; \
        exit(err); \
    } \
} \

[[noreturn]] inline void throwRuntimeError(const char* file, int line, const char* message) {
    std::cerr << "Runtime error in " << file << " at line " << line << ": " << message << std::endl;
    exit(EXIT_FAILURE);
}

inline void llmAssert(bool result, const char* file,int line, const char* message)
{
    if(!result)
    {
        throwRuntimeError(file, line, message);
    }
}
#define LLM_ASSERT(result, message)  \
    do {\
        bool res = (result); \
        if (!res) { \
            throwRuntimeError(__FILE__, __LINE__, message); \
        } \
    }while(0)\
