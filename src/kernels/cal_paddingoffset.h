#pragma once
#include"src/utils/tensor.h"


void launchCalPaddingOffset(
    Tensor<int>* inputSentence,
    Tensor<int>* outputPaddingOffset,
    Tensor<int>* scanSentenceLen
);