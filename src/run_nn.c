// #include "run_nn.h"

#include "ml_math.h"
#include "nn_params.h"
#include "nn.h"

#include <stdio.h>


void run_mlp(float *x, const unsigned int N, unsigned int *image_class)
{
    float out_input[N*HIDDEN_1];
    linear_layer(x, net_0_weight, out_input, net_0_input,
                  net_0_wx_scale,
                  N, HIDDEN_1, INPUT_DIM, 1);
    float out_h1[N*HIDDEN_2];
    linear_layer(out_input, net_2_weight, out_h1, net_2_input,
                  net_2_wx_scale,
                  N, HIDDEN_2, HIDDEN_1, 1);
    float output[N*OUTPUT_DIM];
    linear_layer(out_h1, net_4_weight, output, net_4_input,
                  net_4_wx_scale,
                  N, OUTPUT_DIM, HIDDEN_2, 0);
    // get argmax
    argmax_over_cols(output, image_class, N, OUTPUT_DIM);
}


void run_fxp_mlp(int *x, const unsigned int N, unsigned int *image_class)
{
 
    int out_input[N*HIDDEN_1];
    fxp_linear_layer(x, net_0_weight, out_input, fxp_net_0_input,
                  fxp_net_0_wx_scale,
                  N, HIDDEN_1, INPUT_DIM, 1);
    int out_h1[N*HIDDEN_2];
    fxp_linear_layer(out_input, net_2_weight, out_h1, fxp_net_2_input,
                  fxp_net_2_wx_scale,
                  N, HIDDEN_2, HIDDEN_1, 1);
    int output[N*OUTPUT_DIM];
    fxp_linear_layer(out_h1, net_4_weight, output, fxp_net_4_input,
                  fxp_net_4_wx_scale,
                  N, OUTPUT_DIM, HIDDEN_2, 0);
    
    // get argmax
    fxp_argmax_over_cols(output, image_class, N, OUTPUT_DIM);
}
