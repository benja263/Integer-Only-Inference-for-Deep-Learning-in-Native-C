
#include "nn.h"
#include "nn_math.h"

void linear_layer(const int *x, const int8_t *w,int *output, const int x_amax_quant, const int *x_w_amax_dequant,
                  const unsigned int N, const unsigned int K, const unsigned int M, const unsigned int not_final)
{
    int8_t x_q[N * K];
    quantize(x, x_q, x_amax_quant, N*K);

    mat_mult(x_q, w, output, N, K, M);

    dequantize_per_row(output, x_w_amax_dequant, N, M);

    if (not_final)
        relu(output, N*M);
}


void run_mlp(int *x, const unsigned int N, unsigned int *image_class)
{
    float out_input[N*HIDDEN_1];
    linear_layer(x, net_0_weight, out_input, net_0_input,
                  net_0_wx_scale,
                  N, INPUT_DIM, HIDDEN_1, 1);
    float out_h1[N*HIDDEN_2];
    linear_layer(out_input, net_2_weight, out_h1, net_2_input,
                  net_2_wx_scale,
                  N, HIDDEN_1, HIDDEN_2, 1);
    float output[N*OUTPUT_DIM];
    linear_layer(out_h1, net_4_weight, output, net_4_input,
                  net_4_wx_scale,
                  N, HIDDEN_2, OUTPUT_DIM, 0);
    // get argmax
    argmax_over_cols(output, image_class, N, OUTPUT_DIM);
}