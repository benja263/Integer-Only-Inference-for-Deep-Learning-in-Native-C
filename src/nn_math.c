
#include "nn_math.h"

#include <stdio.h>

#include <math.h>
#include <stdlib.h>

void mat_mult(const int8_t *mat_l, const int8_t *mat_r, int *result, const unsigned int N, const unsigned int K, const unsigned int M)
{       
    unsigned int n, k, m;
    unsigned int row, col;
    int sum_;

    for (m = 0; m < M; m++)
    {
        for (n = 0; n < N; n++)
        {
            row = n*K;
            sum_ = 0;
            for (k = 0; k < K; k++)
            {
                col = k*M;
                sum_ += mat_l[row + k] * mat_r[col + m];
            }
            result[n*M + m] = sum_;
        }
    }
}


int get_output_dim(int input_dim, int kernel_size, int stride)
{
    int output_dim = (input_dim -(kernel_size-1) - 1) / stride;
    return output_dim + 1;
}


void conv2d(const int8_t *x, const int8_t *w, int *y, int N, int C_in, int C_out, int H, int W, int H_new, int W_new,
            int k_size_h, int k_size_w,  int stride_h, int stride_w)
{
    int n_i, c_out_j, c_in_i; /* sample and channels*/
    int n, m; /* kernel iterations */
    int i, j; /* output image iteration*/
        
    for (n_i = 0; n_i < N; n_i++)
    {
        int N_idx_y = n_i*C_out*H_new*W_new;
        int N_idx_x = n_i*C_in*H*W;
        
        for (c_out_j = 0; c_out_j < C_out; c_out_j++)
        {
            int C_out_idx_y = c_out_j*H_new*W_new;
            int C_out_idx_kernel = c_out_j*C_in*k_size_h*k_size_w;
                        
            for (i = 0; i < H_new; i++)
            {
                for (j = 0; j < W_new; j++)
                {
                    int output_idx_y = i*W_new + j;
                    int output_idx_x = i*stride_h*W + j*stride_w;
                    int sum = 0;
                    for (c_in_i = 0; c_in_i < C_in; c_in_i++)
                    {
                        int C_in_idx_x = c_in_i*H*W;
                        int C_in_idx_kernel = c_in_i*k_size_h*k_size_w;
                        for (n = 0; n < k_size_h; n++)
                        {
                            for (m = 0; m < k_size_w; m++)
                            {
                                int kernel_idx = n*k_size_w + m;
                                int kernel_idx_x = n*W + m;
                                int x_value = (int)x[N_idx_x + C_in_idx_x + kernel_idx_x + output_idx_x];
                                int w_value = (int)w[C_out_idx_kernel + C_in_idx_kernel + kernel_idx];
                                sum += x_value*w_value;
                            }
                        }
                    }
                    y[N_idx_y + C_out_idx_y + output_idx_y] = sum;
                }
                
            }
        }
    }
}


void pooling2d(int *x, int *y, int N, int C_out, int H, int W, int H_new, int W_new,
            int k_size_h, int k_size_w,  int stride_h, int stride_w)
{
    int n_i, c_out_j; /* sample and channels*/
    int n, m; /* kernel iterations */
    int i, j; /* output image iteration*/
    
    for (n_i = 0; n_i < N; n_i++)
    {
        int N_idx_y = n_i*C_out*H_new*W_new;
        int N_idx_x = n_i*C_out*H*W;
        
        for (c_out_j = 0; c_out_j < C_out; c_out_j++)
        {
            int C_out_idx_y = c_out_j*H_new*W_new;
            int C_out_idx_x = c_out_j*H*W;

            for (i = 0; i < H_new; i++)
            {
                for (j = 0; j < W_new; j++)
                {
                    int output_idx_y = i*W_new + j;
                    int output_idx_x = i*stride_h*W + j*stride_w;
                    
                    int max = x[N_idx_x+ C_out_idx_x + output_idx_x];
                    for (n = 0; n < k_size_w; n++)
                    {
                        for (m = 0; m < k_size_h; m++)
                        {
                            int kernel_idx = n*W + m;
                            
                            int value = x[N_idx_x+ C_out_idx_x + kernel_idx + output_idx_x];
                            if (value > max)
                                max = value;
                        }
                    }
                    y[N_idx_y + C_out_idx_y + output_idx_y] = max;
                }
                
            }
        }
    }
}


void relu(int *tensor, const unsigned int size)
{
    unsigned int i;
    for (i = 0; i < size; i++)
        tensor[i] = MAX(tensor[i], 0);
}

void quantize(const int *tensor_in, int8_t *tensor_q, const int amax, const int amax_inv, const unsigned int size)
{
    unsigned int i;

    int rounded_value, tensor_int, tensor_frac;
    // separation to integer and fraction parts
    int amax_int = (amax + ROUND_CONST) >> FXP_VALUE;
    int amax_frac = amax - (amax_int << FXP_VALUE);

    for (i = 0; i < size; i++)
    {

        tensor_int = (tensor_in[i] + ROUND_CONST) >> FXP_VALUE;
        if (tensor_int > INT8_MAX_VALUE*amax_inv)
            tensor_q[i] = (int8_t)INT8_MAX_VALUE;
        else if (tensor_int < -INT8_MAX_VALUE*amax_inv)
            tensor_q[i] = -(int8_t)INT8_MAX_VALUE;
        else
        {
            tensor_frac = tensor_in[i] - (tensor_int << FXP_VALUE);

            rounded_value = tensor_int*amax_frac + amax_int*tensor_frac; /* int * fxp = normal multiplication with result is in fxp */
            rounded_value += (tensor_frac*amax_frac + ROUND_CONST) >> FXP_VALUE; /* fxp * fxp = fix-point multiplication with result is in fxp */

            rounded_value = ((rounded_value + ROUND_CONST) >> FXP_VALUE) + tensor_int*amax_int; /* convert fxp to int and add to integer parts as final value should be a rounded integer */

            tensor_q[i] = (int8_t)rounded_value; /* store quantized value in output matrix */
        }
    }
}


void dequantize_per_row(int *mat_in, const int *amax_w, const int amax_x, const unsigned int  N, const unsigned int  M)
{
    unsigned int  k, n;

    int out_value;


    for (n = 0; n < N; n++)
    {
        for (k = 0; k < M; k++)
        {
            
            out_value = amax_w[k] *amax_x;
            if (out_value > (1 << FXP_VALUE))
                mat_in[n*M + k]  *= ((out_value + ROUND_CONST) >> FXP_VALUE);
            else
                mat_in[n*M + k] = (out_value*mat_in[n*M + k] + ROUND_CONST) >> FXP_VALUE;
        }
    }
}



void dequantize_per_channel(int *tensor_in, const int *amax_w, const int amax_x, const unsigned int N, const unsigned int C, const unsigned int K)
{
    unsigned int k, n, c;

    int out_value;

    for (n = 0; n < N; n++)
    {
        for (c = 0; c < C; c++)
        {
            for (k =0; k < K; k++)
            {
                out_value = amax_w[c] *amax_x;
                if (out_value > (1 << FXP_VALUE))
                    tensor_in[n*C + c*K + k]  *= ((out_value + ROUND_CONST) >> FXP_VALUE);
                else
                    tensor_in[n*C + c*K + k] = (out_value*tensor_in[n*C + c*K + k] + ROUND_CONST) >> FXP_VALUE;
            }
        }
            
    }
}


void argmax_over_cols(const int *mat_in, unsigned int *indices, const unsigned int N, const unsigned int M)
{

    // calculate max of each row
    unsigned int n, m, max_idx;
    int row_max, value;
    for (n = 0; n < N; n++)
    {
        row_max = mat_in[n*M];
        max_idx = 0;
        for (m = 0; m < M; m++)
        {
            value = mat_in[n*M + m];
            if (value > row_max)
            {
                row_max = value;
                max_idx = m; // return column
            }
        }
        indices[n] = max_idx;
    }
}
