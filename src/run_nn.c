// #include "run_nn.h"

#include "ml_math.h"
#include "nn_params.h"
#include "nn.h"

#include <stdio.h>


void run_mlp(float *x, const unsigned int N, unsigned int *image_class)
{
    float out_input[N*64];
    linear_layer(x, net_0_weight, net_0_bias, out_input, net_0_input,
                  net_0_wx_scale, net_0_bias_scale, 
                  N, 64, INPUT_DIM, 1);
    float out_h1[N*32];
    linear_layer(out_input, net_2_weight, net_2_bias, out_h1, net_2_input,
                  net_2_wx_scale, net_2_bias_scale, 
                  N, 32, 64, 1);
    float output[N*10];
    linear_layer(out_h1, net_4_weight, net_4_bias, output, net_4_input,
                  net_4_wx_scale, net_4_bias_scale, 
                  N, 10, 32, 0);
    
    // get argmax
    argmax_over_cols(output, image_class, N, 10);
}


void run_fxp_mlp(int *x, const unsigned int N, unsigned int *image_class)
{
 
    int out_input[N*64];
    fxp_linear_layer(x, net_0_weight, net_0_bias, out_input, fxp_net_0_input,
                  fxp_net_0_wx_scale, fxp_net_0_bias_scale, 
                  N, 64, INPUT_DIM, 1);
    int out_h1[N*32];
    fxp_linear_layer(out_input, net_2_weight, net_2_bias, out_h1, fxp_net_2_input,
                  fxp_net_2_wx_scale, fxp_net_2_bias_scale, 
                  N, 32, 64, 1);
    int output[N*10];
    fxp_linear_layer(out_h1, net_4_weight, net_4_bias, output, fxp_net_4_input,
                  fxp_net_4_wx_scale, fxp_net_4_bias_scale, 
                  N, 10, 32, 0);
    
    // get argmax
    fxp_argmax_over_cols(output, image_class, N, 10);
}
