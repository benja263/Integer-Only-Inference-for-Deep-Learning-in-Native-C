#include "nn.h"
#include "nn_math.h"

void linear_layer(const int *x, const int8_t *w, int *output, const int x_scale_factor,
                  const int *w_scale_factor_inv, const int x_scale_factor_inv,
                  const unsigned int  N, const unsigned int  K, const unsigned int  M,
                  const unsigned int  hidden_layer)
{
    int8_t x_q[N * K];
    quantize(x, x_q, x_scale_factor, x_scale_factor_inv,  N*K);

    mat_mult(x_q, w, output, N, K, M);

    dequantize_per_row(output, w_scale_factor_inv, x_scale_factor_inv, N, M);

    if (hidden_layer)
        relu(output, N*M);
    
}

void conv2d_layer(const int *x, const int8_t *w,int *output, const int x_scale_factor, const int *w_scale_factor_inv, const int x_scale_factor_inv,
                  const unsigned int N, const unsigned int C_in, const unsigned int C_out, const int H, const int W,
                  const int H_conv, const int W_conv, const int k_size_h, const int k_size_w,  const int stride_h, const int stride_w)
{
    int8_t x_q[N*C_in*H*W];

    quantize(x, x_q, x_scale_factor, x_scale_factor_inv, N*C_in*H*W);

    conv2d(x_q, w, output, N, C_in, C_out, H, W, H_conv, W_conv,
            k_size_h, k_size_w,  stride_h, stride_w);
    
    dequantize_per_channel(output, w_scale_factor_inv, x_scale_factor_inv, N, C_out, H_conv*W_conv);

    relu(output, N*C_out*H_conv*W_conv);


}

