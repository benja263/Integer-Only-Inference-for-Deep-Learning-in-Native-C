#ifndef NN_H
#define NN_H

#include "nn_params.h"

void linear_layer(const float *x, const int8_t *w, float *output, const float x_amax_quant,
                  const float *x_w_amax_dequant,
                  const unsigned int N, const unsigned int M, const unsigned int K, const unsigned int is_output);

void fxp_linear_layer(const int *x, const int8_t *w, int *output, const int x_amax_quant,
                  const int *x_w_amax_dequant,
                  const unsigned int N, const unsigned int M, const unsigned int K, const unsigned int is_output);


#endif 
