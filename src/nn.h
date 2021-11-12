#ifndef NN.H
#define NN.H

#include "ml_math.h"

void linear_layer(float *x, int8_t *w, int8_t *b, float *output, float x_amax_quant,
                  float *x_w_amax_dequant, float b_amax_dequant, 
                  unsigned int N, unsigned int M, unsigned int K)
{
    int8_t x_q[N * K];
    quantize(x, x_q, x_amax_quant, N*K);

    int tmp_result[N*M];
    mat_mult(x_q, w, tmp_result, N, K, M);

    float b_deq[M];
    dequantize_per_row(tmp_result, output, x_w_amax_dequant, N, M);
    dequantize_bias(b, b_deq, b_amax_dequant, M);

    _mat_vec_add(output, b_deq, N, M);

    relu(output, N*M);
}


#endif 