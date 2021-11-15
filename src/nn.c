
#include "nn.h"
#include "ml_math.h"

#include <stdio.h>

void linear_layer(const float *x, const int8_t *w, const int8_t *b, float *output, const float x_amax_quant,
                  const float *x_w_amax_dequant, const float b_amax_dequant, 
                  const unsigned int N, const unsigned int M, const unsigned int K, const unsigned int is_output)
{
    int8_t x_q[N * K];
    
    quantize(x, x_q, x_amax_quant, N*K);


    int tmp_result[N*M];
    mat_mult(x_q, w, tmp_result, N, K, M);

    float b_deq[M];
    dequantize_per_row(tmp_result, output, x_w_amax_dequant, N, M);
    dequantize_bias(b, b_deq, b_amax_dequant, M);

    _broadcast_mat_vec_add(output, b_deq, N, M);

    if (is_output)
        relu(output, N*M);
}

void fxp_linear_layer(const int *x, const int8_t *w, const int8_t *b, int *output, const int x_amax_quant,
                  const int *x_w_amax_dequant, const int b_amax_dequant, 
                  const unsigned int N, const unsigned int M, const unsigned int K, const unsigned int is_output)
{
    int8_t x_q[N * K];
    fxp_quantize(x, x_q, x_amax_quant, N*K);

    mat_mult(x_q, w, output, N, K, M);

    int b_deq[M];
    fxp_dequantize_per_row(output, x_w_amax_dequant, N, M);
    fxp_dequantize_bias(b, b_deq, b_amax_dequant, M);

    _fxp_broadcast_mat_vec_add(output, b_deq, N, M);

    if (is_output)
        fxp_relu(output, N*M);
}
