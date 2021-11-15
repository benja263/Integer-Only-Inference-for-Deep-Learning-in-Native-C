
#include "ml_math.h"

#include "nn_params.h"

#include <math.h>
#include <stdio.h>


void mat_mult(const int8_t *mat_l, const int8_t *mat_r, int *result, const unsigned int N, const unsigned int K, const unsigned int M)
{       
    unsigned int n, k, m;
    unsigned int row, col;
    int sum_;
    for (m = 0; m < M; m++)
    {
        for (n = 0; n < N; n++)
        {
            sum_ = 0;
            for (k = 0; k < K; k++)
            {
                row = n*K + k;
                col = k*M + m;
                sum_ += mat_l[row] * mat_r[col];
            }
        result[n*M + m] = sum_;
        }
    }
}

void _broadcast_mat_vec_add(float *mat, const float *vec, const unsigned int N, const unsigned int M)
{
    /*
    in place addition of an 1xM row vector and an matrix NxM matrix (broadcasting per row)
    */ 
    unsigned int k, n;
    for (n = 0; n < N; n++)
    {
        for (k = 0; k < M; k++)
            mat[n*M + k] += vec[k];
    }
}

void _fxp_broadcast_mat_vec_add(int *mat, const int *vec, const unsigned int N, const unsigned int M)
{
    /*
    in place addition of an 1xM row vector and an matrix NxM matrix (broadcasting per row)
    */ 
    unsigned int k, n;
    for (n = 0; n < N; n++)
    {
        for (k = 0; k < M; k++)
            mat[n*M + k] += vec[k];
    }
}

void _broadcast_mat_vec_mult(float *mat, const float *vec, const unsigned int N, const unsigned int M)
{
    /*
    in place element-wise multplication of an 1xM row vector and an matrix NxM matrix , such that 
    every element of column m in the matrix is multiplied by element of the m-th column in the vector
    */ 
    unsigned int k, n;
    for (n = 0; n < N; n++)
    {
        for (k = 0; k < M; k++)
            mat[n*M + k] *= vec[k];
    }
}

void _fxp_broadcast_mat_vec_mult(int *mat, const int *vec, const unsigned int N, const unsigned int M)
{
    /*
    in place element-wise multplication of an 1xM row vector and an matrix NxM matrix , such that 
    every element of column m in the matrix is multiplied by element of the m-th column in the vector
    */ 
    unsigned int k, n;
    for (n = 0; n < N; n++)
    {
        for (k = 0; k < M; k++)
            mat[n*M + k] *= vec[k];
    }
}

void relu(float *mat, const unsigned int size)
{
    unsigned int i;
    for (i = 0; i < size; i++)
        mat[i] = MAX(mat[i], 0);
}

void fxp_relu(int *mat, const unsigned int size)
{
    unsigned int i;
    for (i = 0; i < size; i++)
        mat[i] = MAX(mat[i], 0);
}

void quantize(const float *mat, int8_t *mat_q, const float amax, const unsigned int size)
{
    unsigned int i;
    for (i = 0; i < size; i++)
        mat_q[i] = (int8_t)MIN(MAX(roundf((float)mat[i] * amax), _INT8_MIN), _INT8_MAX);
}


void fxp_quantize(const int *mat, int8_t *mat_q, const int amax, const unsigned int size)
{
    unsigned int i;
    for (i = 0; i < size; i++)
    {
        int value;
        // avoiding overflow by separating between integer parts and fractions parts
        // extract integer part and fraction part (in fixed-point)
        int amax_int = amax >> FXP_VALUE;
        int amax_frac = amax - (amax_int << FXP_VALUE);

        int mat_int = mat[i] >> FXP_VALUE;
        int mat_frac = mat[i] - (mat_int << FXP_VALUE);

        // int * fxp = normal multiplicaation with result is in fxp
        value = mat_int*amax_frac + amax_int*mat_frac;
        // fxp * fxp = fix-point multiplication with result is in fxp
        value += (mat_frac*amax_frac + ROUND_CONST) >> FXP_VALUE;
        // convert fxp to int and add to integer parts as final value should be a rounded int
        value = ((value + ROUND_CONST) >> FXP_VALUE) + mat_int*amax_int;

        mat_q[i] = (int8_t)MIN(MAX(value, _INT8_MIN), _INT8_MAX);
    }
}

void dequantize_bias(const int8_t *bias_q, float *bias, const float amax, const unsigned int size)
{
    unsigned int i;
    for (i = 0; i < size; i++)
        bias[i] = (float)bias_q[i] * amax;
}

void fxp_dequantize_bias(const int8_t *bias_q, int *bias, const int amax, const unsigned int size)
{
    unsigned int i;
    for (i = 0; i < size; i++)
        bias[i] = (int)bias_q[i] * amax;
}

void dequantize_per_row(int *mat_q, float *mat, const float *amax, const unsigned int N, const unsigned int M)
{
    /* per row dequantization such that each row is multiplied by the corresponding column value in amax
    * offline calculate reciprocal(amax) so we can replace division by multiplication
    */
    unsigned int i;
    for (i = 0; i < N*M; i++)
        mat[i] = (float)mat_q[i];
    // offline scale amax such that it is equal 1 / amax so we can replace division by multiplication
    _broadcast_mat_vec_mult(mat, amax, N, M);
}

void fxp_dequantize_per_row(int *mat_q, const int *amax, const unsigned int N, const unsigned int M)
{
    /* per row dequantization such that each row is multiplied by the corresponding column value in amax
    * offline calculate reciprocal(amax) so we can replace division by multiplication
    */
    _fxp_broadcast_mat_vec_mult(mat_q, amax, N, M);
}

void argmax_over_cols(const float *mat, unsigned int *indices, const unsigned int N, const unsigned int M)
{
    // calculate max of each row
    unsigned int n, m, max_idx;
    float row_max, value;
    for (n = 0; n < N; n++)
    {
        row_max = MIN_INT;
        max_idx = 0;
        for (m = 0; m < M; m++)
        {
            value = mat[n*M + m];
            if (value > row_max)
            {
                row_max = value;
                max_idx = m; // return column
            }
        }
        indices[n] = max_idx;
    }
}

void fxp_argmax_over_cols(const int *mat, unsigned int *indices, const unsigned int N, const unsigned int M)
{
    // calculate max of each row
    unsigned int n, m, max_idx;
    int row_max, value;
    for (n = 0; n < N; n++)
    {
        row_max = MIN_INT;
        max_idx = 0;
        for (m = 0; m < M; m++)
        {
            value = mat[n*M + m];
            if (value > row_max)
            {
                row_max = value;
                max_idx = m; // return column
            }
        }
        indices[n] = max_idx;
    }
}
