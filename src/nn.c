
#include "nn.h"
#include "nn_math.h"

#include <stdio.h>

void linear_layer(const int *x, const int8_t *w, int *output, const int x_amax_quant, const int *w_amax_dequant, const int x_amax_dequant,
                  const unsigned int  N, const unsigned int  K, const unsigned int  M, const unsigned int  not_output)
{
    int8_t x_q[N * K];
    quantize(x, x_q, x_amax_quant, N*K);

    mat_mult(x_q, w, output, N, K, M);

    dequantize_per_row(output, w_amax_dequant, x_amax_dequant, N, M);

    if (not_output)
        relu(output, N*M);
    
}

void conv2d_layer(const int *x, const int8_t *w,int *output, const int x_amax_quant, const int *w_amax_dequant, const int x_amax_dequant,
                  const unsigned int N, const unsigned int C_in, const unsigned int C_out, const int H, const int W,
                  int H_new, int W_new, int k_size_h, int k_size_w,  int stride_h, int stride_w, int padding_h,
                  int padding_w, int dilation_h, int dilation_w)
{
    int8_t x_q[N*C_in*H*W];
    // printf("before quant\n");
    quantize(x, x_q, x_amax_quant, N*C_in*H*W);
    // printf("quantized\n");
    // for (int i = 0; i < N * C_in*H*W; i++)
    //     printf("%d ", x_q[i]);
    // printf("\n");

    conv2d(x_q, w, output, N, C_in, C_out, H, W, H_new, W_new,
            k_size_h, k_size_w,  stride_h, stride_w, padding_h,
            padding_w, dilation_h, dilation_w);
    // printf("conv\n");

    dequantize_per_channel(output, w_amax_dequant, x_amax_dequant, N, C_out, H_new*W_new);
    // printf("dequantized\n");
    // printf("dequantized\n");
    // for (int i = 0; i < N*C_out*H_new*W_new; i++)
    //     printf("%d ", output[i]);
    // printf("\n");

    relu(output, N*C_out*H_new*W_new);
    // printf("relu\n");
}


void run_mlp(int *x, const unsigned int N, unsigned int *class_indices)
{
    int out_input[N*H_MLP1];
    linear_layer(x, layer_1_weight, out_input, layer_1_s_x,
                  layer_1_s_w_inv, layer_1_s_x_inv,
                  N, INPUT_DIM, H_MLP1, 1);
    int out_h1[N*H_MLP2];
    linear_layer(out_input, layer_2_weight, out_h1, layer_2_s_x,
                  layer_2_s_w_inv, layer_2_s_x_inv,
                  N, H_MLP1, H_MLP2, 1);
    int output[N*OUTPUT_DIM];
    linear_layer(out_h1, layer_3_weight, output, layer_3_s_x,
                  layer_3_s_w_inv, layer_3_s_x_inv,
                  N, H_MLP2, OUTPUT_DIM, 0);
    // get argmax
    argmax_over_cols(output, class_indices, N, OUTPUT_DIM);
    // for (unsigned int i = 0; i < N; i++)
    //     printf("%d ", class_indices[i]);
    // printf("\n");
}

void run_convnet(int *x, const unsigned int N, unsigned int *class_indices)
{
    
    // printf("entered convnet\n");
    int H2 = get_output_dim(H1, 0, 3, 1, 1);
    int W2 = get_output_dim(W1, 0, 3, 1, 1);
    // printf("N: %d\n", N);
    // printf("%d\n", N*C2*H2*W2);
    int out_conv1[N*C2*H2*W2];
    // printf("finished conv1\n");
    conv2d_layer(x, layer_1_weight, out_conv1, layer_1_s_x, layer_1_s_w_inv, layer_1_s_x_inv,
                 N, C1, C2, H1, W1, H2, W2,
                 3, 3,  1, 1, 0, 0, 1, 1);
    // printf("finished conv1\n");
    int H2_pool = get_output_dim(H2, 0, 2, 2, 1);
    int W2_pool = get_output_dim(W2, 0, 2, 2, 1);

    int out_conv1_pool[N*C2*H2_pool*W2_pool];
    
    pooling2d(out_conv1, out_conv1_pool, N, C2, H2, W2, H2_pool, W2_pool, 2, 2,  2, 2, 0, 0, 1, 1);
    // printf("finished pool1\n");
    

    int H3 = get_output_dim(H2_pool, 0, 3, 1, 1);
    int W3 = get_output_dim(W2_pool, 0, 3, 1, 1);
    int out_conv2[N*C3*H3*W3];

    conv2d_layer(out_conv1_pool, layer_2_weight, out_conv2, layer_2_s_x, layer_2_s_w_inv, layer_2_s_x_inv,
                 N, C2, C3, H2_pool, W2_pool, H3, W3,
                 3, 3,  1, 1, 0, 0, 1, 1);
    // printf("finished conv2\n");

    int H3_pool = get_output_dim(H3, 0, 2, 2, 1);
    int W3_pool = get_output_dim(W3, 0, 2, 2, 1);

    int out_conv2_pool[N*C3*H3_pool*W3_pool];
    pooling2d(out_conv2, out_conv2_pool, N, C3, H3, W3, H3_pool, W3_pool, 2, 2,  2, 2, 0, 0, 1, 1);

    // printf("finished pool2\n");

    int output[N*OUTPUT_DIM];
    linear_layer(out_conv2_pool, layer_3_weight, output, layer_3_s_x,
                  layer_3_s_w_inv, layer_3_s_x_inv,
                  N, C3*H3_pool*W3_pool, OUTPUT_DIM, 0);
    // printf("finished output\n");
    // get argmax
    argmax_over_cols(output, class_indices, N, OUTPUT_DIM);
}

/*
    
*/