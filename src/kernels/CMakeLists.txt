add_library(embeddingFunctor STATIC input_embedding.cu)
set_property(TARGET embeddingFunctor PROPERTY CUDA_SEPARABLE_COMPILATION   ON)
set_property(TARGET embeddingFunctor PROPERTY POSITION_INDEPENDENT_CODE  ON)
set_property(TARGET embeddingFunctor PROPERTY CUDA_RESOLVE_DEVICE_SYMBOLS  ON)

add_library(cal_paddingoffset STATIC cal_paddingoffset.cu)
set_property(TARGET cal_paddingoffset PROPERTY CUDA_SEPARABLE_COMPILATION   ON)
set_property(TARGET cal_paddingoffset PROPERTY POSITION_INDEPENDENT_CODE  ON)
set_property(TARGET cal_paddingoffset PROPERTY CUDA_RESOLVE_DEVICE_SYMBOLS  ON)

add_library(rsmnorm STATIC rsmnorm.cu)
set_property(TARGET rsmnorm PROPERTY CUDA_SEPARABLE_COMPILATION   ON)
set_property(TARGET rsmnorm PROPERTY POSITION_INDEPENDENT_CODE  ON)
set_property(TARGET rsmnorm PROPERTY CUDA_RESOLVE_DEVICE_SYMBOLS  ON)

add_library(cublasWrapper STATIC cublas_util.cu)
set_property(TARGET cublasWrapper PROPERTY CUDA_SEPARABLE_COMPILATION   ON)
set_property(TARGET cublasWrapper PROPERTY POSITION_INDEPENDENT_CODE  ON)
set_property(TARGET cublasWrapper PROPERTY CUDA_RESOLVE_DEVICE_SYMBOLS  ON)

add_library(linear STATIC linear.cu)
set_property(TARGET linear PROPERTY CUDA_SEPARABLE_COMPILATION   ON)
set_property(TARGET linear PROPERTY POSITION_INDEPENDENT_CODE  ON)
set_property(TARGET linear PROPERTY CUDA_RESOLVE_DEVICE_SYMBOLS  ON)