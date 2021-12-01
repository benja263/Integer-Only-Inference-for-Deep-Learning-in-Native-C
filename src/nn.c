
#include "nn.h"
#include "nn_math.h"

void linear_layer(const int *x, const int8_t *w,int *output, const int x_amax_quant, const int *x_w_amax_dequant,
                  const unsigned int N, const unsigned int K, const unsigned int M, const unsigned int not_output)
{
    int8_t x_q[N * K];
    quantize(x, x_q, x_amax_quant, N*K);

    mat_mult(x_q, w, output, N, K, M);

    dequantize_per_row(output, x_w_amax_dequant, N, M);

    if (not_output)
        relu(output, N*M);
}

void conv2d_layer(const int *x, const int8_t *w,int *output, const int x_amax_quant, const int *x_w_amax_dequant,
                  const unsigned int N, const unsigned int C_in, const unsigned int C_out, const int H, const int W,
                  int H_new, int W_new, int k_size_h, int k_size_w,  int stride_h, int stride_w, int padding_h,
                  int padding_w, int dilation_h, int dilation_w)
{
    int8_t x_q[N * C_in*H*W];

    quantize(x, x_q, x_amax_quant, N*C_in*H*W);

    conv2d(x_q, w, output, N, C_in, C_out, H, W, H_new, W_new,
            k_size_h, k_size_w,  stride_h, stride_w, padding_h,
            padding_w, dilation_h, dilation_w);

    dequantize_per_channel(output, x_w_amax_dequant, N, C_out, H_new*W_new);

    relu(output, N*C_out*H_new*W_new);
}


void run_mlp(int *x, const unsigned int N, unsigned int *class_indices)
{
    int out_input[N*HIDDEN_1];
    linear_layer(x, net_0_weight, out_input, net_0_s_x,
                  net_0_s_wx_inv,
                  N, INPUT_DIM, HIDDEN_1, 1);
    int out_h1[N*HIDDEN_2];
    linear_layer(out_input, net_2_weight, out_h1, net_2_s_x,
                  net_2_s_wx_inv,
                  N, HIDDEN_1, HIDDEN_2, 1);
    int output[N*OUTPUT_DIM];
    linear_layer(out_h1, net_4_weight, output, net_4_s_x,
                  net_4_s_wx_inv,
                  N, HIDDEN_2, OUTPUT_DIM, 0);
    // get argmax
    argmax_over_cols(output, class_indices, N, OUTPUT_DIM);
}

void run_convnet(int *x, const unsigned int N, unsigned int *class_indices)
{
    int out_conv1[N*C2*H2*W2];

    int H2 = get_output_dim(H1, 0, 3, 1, 1);
    int W2 = get_output_dim(W1, 0, 3, 1, 1);

    conv2d_layer(x, net_0_weight, out_conv1, net_0_s_x, net_0_s_wx_inv,
                 N, C_1, C_2, H_1, W_1, H2, W2,
                 3, 3,  1, 1, 0, 0, 1, 1);

    int H2_pool = get_output_dim(H2, 0, 2, 2, 1);
    int W2_pool = get_output_dim(W2, 0, 2, 2, 1);

    int out_conv1_pool[N*C2*H2_pool*W2_pool];
    
    pooling2d(out_conv1, out_conv1_pool, N, C2, H2, W2, H2_pool, W2_pool, 2, 2,  2, 2, 0, 0, 1, 1);

    int out_conv2[N*C3*H3*W3];

    int H3 = get_output_dim(H2_pool, 0, 3, 1, 1);
    int W3 = get_output_dim(W2_pool, 0, 3, 1, 1);

    conv2d_layer(out_conv1_pool, net_3_weight, out_conv2, net_3_s_x, net_3_s_wx_inv,
                 N, C_2, C_3, H_2_pool, W_2_pool, H3, W3,
                 3, 3,  1, 1, 0, 0, 1, 1);

    int H3_pool = get_output_dim(H3, 0, 2, 2, 1);
    int W3_pool = get_output_dim(W3, 0, 2, 2, 1);

    int out_conv2_pool[N*C3*H3_pool*W3_pool];
    pooling2d(out_conv2, out_conv2_pool, N, C3, H3, W3, H3_pool, W3_pool, 2, 2,  2, 2, 0, 0, 1, 1);


    int output[N*OUTPUT_DIM];
    linear_layer(out_conv2_pool, net_7_weight, output, net_7_s_x,
                  net_7_s_wx_inv,
                  N, C3*H3_pool*W3_pool, OUTPUT_DIM, 0);
    // get argmax
    argmax_over_cols(output, class_indices, N, OUTPUT_DIM);
}

/*
    
*/