#include <algorithm>   // std::fill_n
#include <iostream>    // snprintf
#include <math.h>      // expf, log
#include <stdlib.h>    // rand
#include <string>      // std::string
#include <vector>      // std::vector
#include <random>

#include <cuda.h>
#include <cuda_fp16.h>
#include <cuda_runtime.h>

#include "src/kernels/input_embedding.h"
// (RussWong)note:
// there is no embedding cpu kernel implementation now
// `./embedding` to test fp16 GPU kernel
// `./embedding 1` to test fp32 GPU kernel

void embedding_cpu(int*input, float* embedding_table, float* output, int max_context_token_num,  int vocab_size, int hidden_size) 
{
    for(int i = 0; i< max_context_token_num * hidden_size; ++i)
    {
        int token_id = input[i/hidden_size];
        output[i] = embedding_table[token_id*hidden_size+ i%hidden_size];
    }
}


bool checkResults(float* h_output, float* d_output, const int output_size) {
    float* d_output_cpu = (float*) malloc(output_size * sizeof(float)); // prepare for cpu check
    CUDA_CHECK(cudaMemcpy(d_output_cpu, d_output, output_size * sizeof(float), cudaMemcpyDeviceToHost));
    for (int i = 0; i < output_size; ++i) {
        if (fabs(d_output_cpu[i] - h_output[i]) > 1e5) {
            std::cout << "Dev : ";
            for (int j = max(0, i - 10); j < min(output_size, i + 10); ++j) {
                std::cout << d_output_cpu[i];
            }
            std::cout << std::endl;
            std::cout << "Cpu : ";
            for (int j = max(0, i - 10); j < min(output_size, i + 10); ++j) {
                std::cout << h_output[i];
            }
            std::cout << std::endl;
            free(d_output_cpu);
            return false;
        }
    }
    std::cout<<"cpu result is same as gpu result\n";
    free(d_output_cpu);
    return true;
}

int main(int argc, char *argv[]) {
    const int max_context_token_num = 64;
    const int hidden_size = 4096;
    const int vocab_size = 30000;
    const int input_size = max_context_token_num;
    const int table_size = vocab_size * hidden_size;
    const int output_size = max_context_token_num * hidden_size;

    //prepare data on cpu
    int *h_input = (int*)malloc(sizeof(int)* max_context_token_num);
    float* h_embed_table = (float*)malloc(sizeof(float)*vocab_size* hidden_size);
    float* h_output = (float*)malloc(sizeof(float)*max_context_token_num* hidden_size);

    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_int_distribution<>  dis_int(0,vocab_size-1);
    std::uniform_real_distribution<> dis_float(1.0,2.0);
    for(int i = 0; i<max_context_token_num;++i)
        h_input[i]=dis_int(gen);
    for(int i= 0; i<vocab_size;++i)
    {
       for(int j =0; j<hidden_size;++j)
       {
            h_embed_table[i*hidden_size+j] = dis_float(gen);
       }
    }

    // copy to cuda
    int* d_input;
    float* d_embed_table, *d_out_put;
    cudaMalloc((void**)&d_input,sizeof(int)*input_size);
    cudaMalloc((void**)&d_embed_table,sizeof(float)*table_size);
    cudaMalloc((void**)&d_out_put,sizeof(float)*output_size);
    cudaMemcpy(d_input,h_input,sizeof(int)*input_size,cudaMemcpyHostToDevice);
    cudaMemcpy(d_embed_table,h_embed_table,sizeof(float)*table_size,cudaMemcpyHostToDevice);

    DataType type_float = getTensorType<float>();
    DataType type_int = getTensorType<int>();
    
    Tensor<int>* input_tensor = new Tensor<int>(Device::GPU,type_int,{max_context_token_num},d_input);
    Tensor<float>* output_tensor = new Tensor<float>(Device::GPU,type_float,{max_context_token_num,hidden_size},d_out_put);
    EmbeddingWeight<float> embedding_table;
    embedding_table.data = d_embed_table;
    launchInputEmbedding(input_tensor,output_tensor,&embedding_table);

    // process result;
    //cudaMemcpy(h_output,d_out_put,sizeof(float)*output_size,cudaMemcpyDeviceToHost);

    // cpu
    float* h_out_cpu = (float*)malloc(sizeof(float)*output_size);
    embedding_cpu(h_input,h_embed_table,h_out_cpu,max_context_token_num,vocab_size,hidden_size);
    checkResults(h_out_cpu,d_out_put,10);
    cudaFree(d_embed_table);
    cudaFree(d_input);
    cudaFree(d_out_put);
    free(h_embed_table);
    free(h_input);
    free(h_output);
    free(h_out_cpu);
    return 0;
}