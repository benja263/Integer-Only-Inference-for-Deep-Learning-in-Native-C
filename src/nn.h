#ifndef NN_H
#define NN_H

#include "nn_params.h"

void linear_layer(const int *x, const int8_t *w, int *output, const int x_amax_quant,
                  const int *x_w_amax_dequant,
                  const unsigned int N,  const unsigned int K, const unsigned int M, const unsigned int not_final);

void run_mlp(int *x, const unsigned int N, unsigned int *image_class);

#endif 
