
#include "mlp.h"
#include "nn_math.h"

#include <stdio.h>
#include <stdlib.h>

void linear_layer(const int *x, const int8_t *w, int *output, const int x_amax_quant, const int *w_amax_dequant, const int x_amax_dequant,
                  const unsigned int  N, const unsigned int  K, const unsigned int  M, const unsigned int  not_output)
{
    int8_t x_q[N * K];
    quantize(x, x_q, x_amax_quant, x_amax_dequant,  N*K);

    mat_mult(x_q, w, output, N, K, M);

    dequantize_per_row(output, w_amax_dequant, x_amax_dequant, N, M);

    if (not_output)
        relu(output, N*M);
    
}


void run_mlp(int *x, const unsigned int N, unsigned int *class_indices)
{
    int out_input[N*H1];
    linear_layer(x, layer_1_weight, out_input, layer_1_s_x,
                  layer_1_s_w_inv, layer_1_s_x_inv,
                  N, INPUT_DIM, H1, 1);
    int out_h1[N*H2];
    linear_layer(out_input, layer_2_weight, out_h1, layer_2_s_x,
                  layer_2_s_w_inv, layer_2_s_x_inv,
                  N, H1, H2, 1);
    int output[N*OUTPUT_DIM];
    linear_layer(out_h1, layer_3_weight, output, layer_3_s_x,
                  layer_3_s_w_inv, layer_3_s_x_inv,
                  N, H2, OUTPUT_DIM, 0);
    // get argmax
    argmax_over_cols(output, class_indices, N, OUTPUT_DIM);
    // for (unsigned int i = 0; i < N; i++)
    //     printf("%d ", class_indices[i]);
    // printf("\n");
}

