#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include "input_embedding.h"
#include "src/utils/macro.h"
#include "src/utils/tensor.h"

/*
    embeddingFunctor is a kernel function that performs the embedding lookup
    for a given set of input IDs. It takes the input IDs and retrieves the
    corresponding embeddings from the embedding table. The output is stored
    in the output tensor.
    input_ids: [token_num]
    embedding_table: [vocab_size, hidden_size]
    output: [token_num, hidden_size]

    output[i] = embedding_table[input_ids[i]]
*/

/*
# LLM引擎的输入嵌入（Input Embedding）机制解析

输入嵌入（Input
Embedding）是大语言模型处理文本的第一个关键步骤，它负责将离散的token
ID转换为连续的向量表示。这个`input_embedding.cu`文件实现了这一过程的CUDA加速版本。

## 核心功能

这段代码的主要功能是通过GPU并行计算，高效地完成大量token的嵌入查找操作。简单来说，它做的是：
1. 接收一系列token ID（整数）
2. 从预训练的嵌入表中查找每个ID对应的向量
3. 将这些向量组合成一个输出矩阵

## 代码结构分析

### 1. CUDA内核函数 `embeddingFunctor`
这是在GPU上执行的核心函数，它通过并行计算快速完成嵌入查找：
- 使用模板参数`<typename T>`支持不同的数据类型（如float或half）
- 每个GPU线程负责处理一个或多个输出元素
- 通过`index / hidden_size`计算当前处理的是哪个token
- 通过`token_id * hidden_size + index % hidden_size`定位嵌入表中的具体位置

### 2. 主机端函数 `launchInputEmbeddingKernel`
这个函数负责配置和启动CUDA内核：
- 设置线程块大小（blockSize = 256）和网格大小（gridSize = 2048）
- 从输入张量中提取维度信息
- 进行参数检查，确保输入维度匹配
- 使用`<<<...>>>`语法启动GPU内核
- 使用`CUDA_CHECK`检查执行是否成功

### 3. 模板实例化
代码最后提供了两种数据类型的显式实例化：
- `float`类型：用于全精度（32位）计算
- `half`类型：用于半精度（16位）计算，可节省内存和提高速度

## 技术要点

1. **内存访问模式**：代码中的内存访问模式经过优化，以提高GPU的缓存命中率
2. **线程分配**：每个线程处理多个元素，通过`index += blockDim.x * gridDim.x`跳转
3. **数据并行**：利用GPU的大规模并行能力同时处理多个嵌入向量
4. **模板编程**：使用C++模板提供类型灵活性，支持不同精度的计算

这种实现方式使得LLM模型能够高效地处理长序列输入，为后续的注意力层和前馈网络层提供基础特征表示。
*/

template <typename T>
__global__ void embeddingFunctor(const int *input_ids, T *output, const T *embedding_table,
                                 const int max_context_token_num, const int hidden_size)
{
    int index = blockIdx.x * blockDim.x + threadIdx.x;
    while (index < max_context_token_num * hidden_size) {
        int token_id = input_ids[ index / hidden_size ];
        output[ index ] = embedding_table[ token_id * hidden_size + index % hidden_size ];
        index += blockDim.x * gridDim.x;
    }
}

template <typename T>
void launchInputEmbeddingKernel(Tensor<int> *input_ids,             // int [token_num]
                                Tensor<T> *output,                  // FP32 [token_num, hidden_size] = [token_num, 4096]
                                EmbeddingWeight<T> *embedding_table // FP32 [vocab_size, hidden_size]
)
{
    const int blockSize = 256;
    const int max_context_token_num = output->shape[ 0 ];
    const int hidden_size = output->shape[ 1 ];
    const int gridSize = 2048;
    LLM_ASSERT(max_context_token_num == input_ids->shape[ 0 ], "input_ids and output "
                                                               "must have the same first dimension");
    embeddingFunctor<<<gridSize, blockSize>>>(input_ids->data, output->data, embedding_table->data,
                                              max_context_token_num, hidden_size);

    CUDA_CHECK(cudaGetLastError());
}

template void launchInputEmbeddingKernel(Tensor<int> *input_ids, Tensor<float> *output,
                                         EmbeddingWeight<float> *embedding_table);
template void launchInputEmbeddingKernel(Tensor<int> *input_ids, Tensor<half> *output,
                                         EmbeddingWeight<half> *embedding_table);