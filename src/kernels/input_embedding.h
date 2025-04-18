#pragma  once
#include "src/weights/llama/embedding_weights.h"
#include "src/utils/string_utils.h"
#include "src/utils/tensor.h"


/*
    this function is used to launch the input embedding kernel.
    launchInputEmbedding is a template function that takes input_ids, output, and embedding_table as parameters.
    input_ids is a pointer to a Tensor of integers, 
    output is a pointer to a Tensor of type T, and embedding_table is a pointer to an EmbeddingWeight of type T.
    The function is designed to perform the input embedding operation, 
    which typically involves looking up the embeddings for the input IDs in the embedding table 
    and storing the result in the output tensor.
*/

template <typename T>

void launchInputEmbedding(
    Tensor<int> *input_ids,  // input_ids is a pointer to a Tensor of integers the shape is [max_context_token_num]
    Tensor<T> * output, // output is a pointer to a Tensor of type T the shape is [max_context_token_num, hidden_size]
    EmbeddingWeight<T> *embedding_table // embedding_table is a pointer to an EmbeddingWeight of type T.the shape is [vocab_size, hidden_size]
);