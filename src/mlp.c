
#include "mlp_params.h"
#include "nn.h"
#include "nn_math.h"

void run_mlp(const int *x, const unsigned int N, unsigned int *class_indices)
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
}

