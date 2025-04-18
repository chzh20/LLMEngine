cmake_minimum_required(VERSION 3.22)

add_executable(embedding_test inputEmbedding.cu)
link_directories(${CMAKE_BINARY_DIR}/lib)
target_link_libraries(
    embedding_test PUBLIC
    embeddingFunctor
    cudart
    cublas
)