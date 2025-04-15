#pragma  once
#include "src/weights/llama/embedding_weights.h"

#include "src/utils/string_utils.h"
#include "src/utils/tensor.h"


/*
    * @brief Input embedding layer for Llama model
    * 
    * This class handles the input embedding layer of the Llama model. It manages the weights and biases
    * for the embedding layer, as well as the input tensor. The class provides methods to initialize,
    * forward, and get the output tensor.
    * 
    * @tparam T Type of the data (e.g., float, half)
    *
*/

template <typename T>

void launchInputEmbeddingKernel(
    Tensor<int> *input_ids, 
    Tensor<T> * output,
    EmbeddingWeight<T> *embedding_table
);