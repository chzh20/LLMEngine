#include"cal_paddingoffset.h"
#include"src/utils/macro.h"
#include "cuda_runtime.h"
#include "device_launch_parameters.h"




/*



11110
11100
11000

maxQlen = 5;
bathSize = 3;

inputSentence ={4,3,2}
scanSentenceLen ={0,4,7,9}
outputPaddingOffset ={
  00000 
  11111
  33333
}
 

*/

__global__ void calPaddingOffset(int* outputPaddingOffset,// shape[bacthSize* maxQLen]
                                 int* scanSentenceLen, //shape[batchSize+1]
                                 int* inputSentence, //shape[batchSize]
                                 int batchSize,
                                 int maxQLen) // max lenth of all sentences 
{
    int index = 0;
    int scan_offset = 0; 
    int total_seqlen = 0;
    for(int bs = 0; bs < batchSize; ++bs)
    {
        int seqlen = inputSentence[bs];
        
        scanSentenceLen[bs] = total_seqlen;
        
        for(int i = 0; i < seqlen;++i)
        {
            outputPaddingOffset[index] = scan_offset;
            index++;
        }

        scan_offset += maxQLen-seqlen; // paddinsg offset
        total_seqlen += seqlen;
    }
    scanSentenceLen[batchSize] = total_seqlen;
    
}






void launchCalPaddingOffset(
    Tensor<int>* inputSentence,
    Tensor<int>* outputPaddingOffset,
    Tensor<int>* scanSentenceLen
)
{
    const int batchSize = outputPaddingOffset->shape[0];
    const int maxQLen = outputPaddingOffset->shape[1]; // max sequence length
    LLM_ASSERT(batchSize == inputSentence->shape[0],"input lengh should equal to padding offset bs dim");
    LLM_ASSERT(batchSize == scanSentenceLen->shape[0]-1,"scan Sentence len should equal to padding offset bs dim+1");


}