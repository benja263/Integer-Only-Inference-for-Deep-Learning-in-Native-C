#ifndef ML_MATH.H
#define ML_MATH.H

#define MAX(a,b) (((a)>(b))?(a):(b))
#define MIN(a,b) (((a)<(b))?(a):(b))
#define INT8_MAX 127
#define INT8_MIN -127


#include <stdint.h>
#include <math.h>

void mat_mult(int8_t *mat_l, int8_t *mat_r, int *result, unsigned int N, unsigned int K, unsigned int M)
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

void _mat_vec_add(float *mat, float *vec, unsigned int N, unsigned int M)
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

void _mat_vec_mult(float *mat, float *vec, unsigned int N, unsigned int M)
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

void relu(float *mat, int size)
{
    unsigned int i;
    for (i = 0; i < size; i++)
        mat[i] = MAX(mat[i], 0);
}

void quantize(float *mat, int8_t *mat_q, float amax, unsigned int size)
{
    unsigned int i;
    for (i = 0; i < size; i++)
        mat_q[i] = MIN(MAX(roundf((float)mat[i] * amax), INT8_MIN), INT8_MAX);
}

void dequantize_bias(int8_t *bias_q, float *bias, float amax, unsigned int size)
{
    unsigned int i;
    for (i = 0; i < size; i++)
        bias[i] = (float)bias_q[i] / amax;
}

void dequantize_per_row(int *mat_q, float *mat, float *amax, unsigned int N, unsigned int M)
{
    
    unsigned int i;
    for (i = 0; i < N*M; i++)
        mat[i] = (float)mat_q[i];
    // offline scale amax such that it is equal 1 / amax so we can replace division by multiplication
    _mat_vec_mult(mat, amax, N, M);
}
#endif //