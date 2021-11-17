/*******************************************************************
*  The function/typedef name
*
* Description of the function/typedef purpose
*******************************************************************/
#ifndef NN_MATH_H
#define NN_MATH_H

#define MAX(a,b) (((a)>(b))?(a):(b))
#define MIN(a,b) (((a)<(b))?(a):(b))

#define MIN_INT -2147483648
#define _INT8_MAX 127
#define _INT8_MIN -127

#define ROUND_CONST (1 << (FXP_VALUE - 1)) // = 0.5 to before right shifting to improve rounding

#include "nn_params.h"

void mat_mult(const int8_t *mat_l, const int8_t *mat_r, int *result, const unsigned int N, const unsigned int K, const unsigned int M);

void _broadcast_mat_vec_mult(int *mat, const int *vec, const unsigned int N, const unsigned int M);

void relu(int *mat, const unsigned int size);

void quantize(const int *mat, int8_t *mat_q, const int amax, const unsigned int size);

void dequantize_per_row(int *mat_q, const int *amax, const unsigned int N, const unsigned int M);

void argmax_over_cols(const int *mat, unsigned int *indices, const unsigned int N, const unsigned int M);

#endif //

