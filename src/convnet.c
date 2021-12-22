
#include "convnet.h"
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

void linear_layerf(const float *x, const int8_t *w, float *output, const float x_amax_quant, const float *w_amax_dequant, const float x_amax_dequant,
                  const unsigned int  N, const unsigned int  K, const unsigned int  M, const unsigned int  not_output)
{
    int8_t x_q[N * K];

    quantizef(x, x_q, x_amax_quant,  N*K);

    int tmp_output[N*M];
    mat_mult(x_q, w, tmp_output, N, K, M);

    dequantize_per_rowf(tmp_output, output, w_amax_dequant, x_amax_dequant, N, M);


    if (not_output)
        reluf(output, N*M);
    
}

void conv2d_layer(const int *x, const int8_t *w,int *output, const int x_amax_quant, const int *w_amax_dequant, const int x_amax_dequant,
                  const unsigned int N, const unsigned int C_in, const unsigned int C_out, const int H, const int W,
                  int H_conv, int W_conv, int H_pool, int W_pool, int k_size_h, int k_size_w,  int stride_h, int stride_w)
{
    int8_t x_q[N*C_in*H*W];
    // printf("before quant\n");
    quantize(x, x_q, x_amax_quant, x_amax_dequant, N*C_in*H*W);

    // printf("after quant\n");
    int out_conv[N*C_out*H_conv*W_conv];
    conv2d(x_q, w, out_conv, N, C_in, C_out, H, W, H_conv, W_conv,
            k_size_h, k_size_w,  stride_h, stride_w);
    
    // printf("conv\n");
    // for (int i = 0; i <N*C_out*H_conv*W_conv; i++)
    // {
    //     printf("%d ", out_conv[i]);
    // }
    // printf("\n");

    // printf("after conv\n");
    // int output_tmp[N*C_out*H_conv*W_conv];
    dequantize_per_channel(out_conv, w_amax_dequant, x_amax_dequant, N, C_out, H_conv*W_conv);

    pooling2d(out_conv, output, N, C_out, H_conv, W_conv, H_pool, W_pool, 2, 2,  2, 2);
    // printf("after pool\n");

    // printf("after dequant\n");
    relu(output, N*C_out*H_pool*W_pool);

    // for (int i = 0; i < N*C_out*H_pool*W_pool; i++)
    // {
    //     printf("%d ", output[i]);
    // }
    // printf("\nend_conv_layer\n");
    // exit(1);


    // printf("after rellu\n");

}

void conv2d_layerf(const float *x, const int8_t *w, float *output, const float x_amax_quant, const float *w_amax_dequant, const float x_amax_dequant,
                  const unsigned int N, const unsigned int C_in, const unsigned int C_out, const int H, const int W,
                  int H_conv, int W_conv, int H_pool, int W_pool, int k_size_h, int k_size_w,  int stride_h, int stride_w)
{   

    // printf("input conv\n");
    // for (int i = 0; i <N*C_in*H*W; i++)
    // {
    //     printf("%f ",x[i]);
    // }
    // printf("\n");
    int8_t x_q[N*C_in*H*W];

    quantizef(x, x_q, x_amax_quant, N*C_in*H*W);

    int out_conv[N*C_out*H_conv*W_conv];
    conv2d(x_q, w, out_conv, N, C_in, C_out, H, W, H_conv, W_conv,
            k_size_h, k_size_w,  stride_h, stride_w);

    
    float output_tmp[N*C_out*H_conv*W_conv];
    dequantize_per_channelf(out_conv, output_tmp, w_amax_dequant, x_amax_dequant, N, C_out, H_conv*W_conv);
    // for (int i = 0; i < N*C_out*H_conv*W_conv; i++)
    // {
    //     printf("%f ", output_tmp[i]);
    // }
    // printf("\nend_conv_layer\n");
    // // printf("sizes N:%d, C_out %d H_conv%d W_conv%d\n", N, C_out, H_conv, W_conv);
    // exit(1);

    pooling2df(output_tmp, output, N, C_out, H_conv, W_conv, H_pool, W_pool, 2, 2,  2, 2);

    reluf(output, N*C_out*H_pool*W_pool);


}


void run_convnet(int *x, unsigned int *class_indices)
{
    
    // printf("entered convnet\n");
    // int H2 = get_output_dim(H1, 3, 1);
    // int W2 = get_output_dim(W1, 3, 1);
    // int H2_pool = get_output_dim(H2, 2, 2);
    // int W2_pool = get_output_dim(W2, 2, 2);

    static int out_conv1[BATCH_SIZE*C1*H1_pool*W1_pool];

    conv2d_layer(x, layer_1_weight, out_conv1, layer_1_s_x, layer_1_s_w_inv, layer_1_s_x_inv,
                 BATCH_SIZE, C0, C1, H1, W1, H1_conv, W1_conv, H1_pool, W1_pool,
                 3, 3,  1, 1);
    
    // printf("finish/ed pool1\n");
    

    // int H3 = get_output_dim(H2_pool, 3, 1);
    // int W3 = get_output_dim(W2_pool, 3, 1);
    // int H3_pool = get_output_dim(H3, 2, 2);
    // int W3_pool = get_output_dim(W3, 2, 2);

    static int out_conv2[BATCH_SIZE*C2*H2_pool*W2_pool];
    conv2d_layer(out_conv1, layer_2_weight, out_conv2, layer_2_s_x, layer_2_s_w_inv, layer_2_s_x_inv,
                 BATCH_SIZE, C1, C2, H1_pool, W1_pool, H2_conv, W2_conv, H2_pool, W2_pool,
                 3, 3,  1, 1);

    static int output[BATCH_SIZE*OUTPUT_DIM];
    linear_layer(out_conv2, layer_3_weight, output, layer_3_s_x,
                  layer_3_s_w_inv, layer_3_s_x_inv,
                  BATCH_SIZE, C2*H2_pool*W2_pool, OUTPUT_DIM, 0);
    // printf("finished output\n");
    // get argmax
    argmax_over_cols(output, class_indices, BATCH_SIZE, OUTPUT_DIM);
}

void run_convnetf(float *x, unsigned int *class_indices)
{
    
    // printf("entered convnetf\n");
    // int H2 = get_output_dim(H1, 3, 1);
    // int W2 = get_output_dim(W1, 3, 1);
    // int H2_pool = get_output_dim(H2, 2, 2);
    // int W2_pool = get_output_dim(W2, 2, 2);

    static float out_conv1[BATCH_SIZE*C1*H1_pool*W1_pool];

    conv2d_layerf(x, layer_1_weight, out_conv1, layer_1_s_x_f, layer_1_s_w_inv_f, layer_1_s_x_inv_f,
                 BATCH_SIZE, C0, C1, H1, W1, H1_conv, W1_conv, H1_pool, W1_pool,
                 3, 3,  1, 1);
    

    // int H3 = get_output_dim(H2_pool, 3, 1);
    // int W3 = get_output_dim(W2_pool, 3, 1);
    // int H3_pool = get_output_dim(H3, 2, 2);
    // int W3_pool = get_output_dim(W3, 2, 2);

    static float out_conv2[BATCH_SIZE*C2*H2_pool*W2_pool];
    conv2d_layerf(out_conv1, layer_2_weight, out_conv2, layer_2_s_x_f, layer_2_s_w_inv_f, layer_2_s_x_inv_f,
                 BATCH_SIZE, C1, C2, H1_pool, W1_pool, H2_conv, W2_conv, H2_pool, W2_pool,
                 3, 3,  1, 1);

    static float output[BATCH_SIZE*OUTPUT_DIM];
    linear_layerf(out_conv2, layer_3_weight, output, layer_3_s_x_f,
                  layer_3_s_w_inv_f, layer_3_s_x_inv_f,
                  BATCH_SIZE, C2*H2_pool*W2_pool, OUTPUT_DIM, 0);

    // for (int i = 0; i < N; i++)
    // {
    //     printf("%f ", output[i]);
    // }
    // printf("\n");

    argmax_over_colsf(output, class_indices, BATCH_SIZE, OUTPUT_DIM);

}

/*
    
*/