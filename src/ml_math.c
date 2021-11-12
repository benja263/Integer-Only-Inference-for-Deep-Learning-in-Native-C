#define MIN_INT -2147483648

#include "ml_math.h"
#include <math.h>


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

void _mat_vec_add(float *mat, const float *vec, const unsigned int N, const unsigned int M)
{
    /*
    in place addition of an Nx1 column vector and an matrix NxM matrix 
    */ 
    unsigned int k, n;
    for (n = 0; n < N; n++)
    {
        for (k = 0; k < M; k++)
            mat[n*M + k] += vec[n];
    }
}

void _mat_vec_mult(float *mat, const float *vec, const unsigned int N, const unsigned int M)
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

void relu(float *mat, const int size)
{
    unsigned int i;
    for (i = 0; i < size; i++)
        mat[i] = MAX(mat[i], 0);
}

void quantize(float *mat, int8_t *mat_q, const float amax, const unsigned int size)
{
    unsigned int i;
    for (i = 0; i < size; i++)
        mat_q[i] = MIN(MAX(roundf((float)mat[i] * amax), INT8_MIN), INT8_MAX);
}

void dequantize_bias(const int8_t *bias_q, float *bias, const float amax, const unsigned int size)
{
    unsigned int i;
    for (i = 0; i < size; i++)
        bias[i] = (float)bias_q[i] / amax;
}

void dequantize_per_row(int *mat_q, float *mat, const float *amax, const unsigned int N, const unsigned int M)
{
    
    unsigned int i;
    for (i = 0; i < N*M; i++)
        mat[i] = (float)mat_q[i];
    // offline scale amax such that it is equal 1 / amax so we can replace division by multiplication
    _mat_vec_mult(mat, amax, N, M);
}

void argmax_over_cols(float *mat, unsigned int *indices, const unsigned int N, const unsigned int M)
{
    // calculate max of each row
    unsigned int n, m, max_idx;
    int row_max;
    float value;
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
