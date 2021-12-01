
#include "nn_math.h"
#include "nn_params.h"


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
                // if (k == K) /* add bias */
                //     sum_ += mat_r[col + m];
                // else
                    sum_ += mat_l[row + k] * mat_r[col + m];
            }
            result[n*M + m] = sum_;
        }
    }
}

void _broadcast_mat_vec_mult(int *mat, const int *vec, const unsigned int N, const unsigned int M)
{
    unsigned int k, n;

    for (n = 0; n < N; n++)
    {
        for (k = 0; k < M; k++)
            mat[n*M + k] *= vec[k];
    }
}

void relu(int *mat, const unsigned int size)
{
    unsigned int i;
    for (i = 0; i < size; i++)
        mat[i] = MAX(mat[i], 0);
}


void quantize(const int *mat_in, int8_t *mat_q, const int amax, const unsigned int size)
{
    unsigned int i;
    for (i = 0; i < size; i++)
    {
        int rounded_value;
        // separation to integer and fraction parts
        int amax_int = amax >> FXP_VALUE;
        int amax_frac = amax - (amax_int << FXP_VALUE);

        int mat_int = mat_in[i] >> FXP_VALUE;
        int mat_frac = mat_in[i] - (mat_int << FXP_VALUE);

        rounded_value = mat_int*amax_frac + amax_int*mat_frac; /* int * fxp = normal multiplicaation with result is in fxp */
        rounded_value += (mat_frac*amax_frac + ROUND_CONST) >> FXP_VALUE; /* fxp * fxp = fix-point multiplication with result is in fxp */

        rounded_value = ((rounded_value + ROUND_CONST) >> FXP_VALUE) + mat_int*amax_int; /* convert fxp to int and add to integer parts as final value should be a rounded integer */

        mat_q[i] = (int8_t)MIN(MAX(rounded_value, _INT8_MIN), _INT8_MAX); /* store quantized value in output matrix */
    }
}


void dequantize_per_row(int *mat_in, const int *amax, const unsigned int N, const unsigned int M)
{
    _broadcast_mat_vec_mult(mat_in, amax, N, M);
}

void argmax_over_cols(const int *mat_in, unsigned int *indices, const unsigned int N, const unsigned int M)
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
