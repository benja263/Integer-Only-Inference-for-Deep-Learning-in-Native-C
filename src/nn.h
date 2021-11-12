#ifndef NN.H
#define NN.H

#include "ml_math.h"

void linear_layer(const float *x, const int8_t *w, const int8_t *b, float *output, const float x_amax_quant,
                  const float *x_w_amax_dequant, const float b_amax_dequant, 
                  const unsigned int N, const unsigned int M, const unsigned int K, const unsigned int is_output);


#endif 