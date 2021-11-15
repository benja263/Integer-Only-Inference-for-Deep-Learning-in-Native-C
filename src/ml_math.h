#ifndef ML_MATH_H
#define ML_MATH_H

#define MAX(a,b) (((a)>(b))?(a):(b))
#define MIN(a,b) (((a)<(b))?(a):(b))

#define MIN_INT -2147483648
#define _INT8_MAX 127
#define _INT8_MIN -127

#define ROUND_CONST (1 << (FXP_VALUE - 1))

#include "nn_params.h"

// int fxp_mult(int a, int b); 

void mat_mult(const int8_t *mat_l, const int8_t *mat_r, int *result, const unsigned int N, const unsigned int K, const unsigned int M);

void _broadcast_mat_vec_mult(float *mat, const float *vec, const unsigned int N, const unsigned int M);

void _fxp_broadcast_mat_vec_add(int *mat, const int *vec, const unsigned int N, const unsigned int M);

void _broadcast_mat_vec_mult(float *mat, const float *vec, const unsigned int N, const unsigned int M);

void _fxp_broadcast_mat_vec_mult(int *mat, const int *vec, const unsigned int N, const unsigned int M);

void relu(float *mat, const unsigned int size);

void fxp_relu(int *mat, const unsigned int size);

void quantize(const float *mat, int8_t *mat_q, const float amax, const unsigned int size);

void fxp_quantize(const int *mat, int8_t *mat_q, const int amax, const unsigned int size);

void dequantize_bias(const int8_t *bias_q, float *bias, const float amax, const unsigned int size);

void fxp_dequantize_bias(const int8_t *bias_q, int *bias, const int amax, const unsigned int size);

void dequantize_per_row(int *mat_q, float *mat, const float *amax, const unsigned int N, const unsigned int M);

void fxp_dequantize_per_row(int *mat_q, const int *amax, const unsigned int N, const unsigned int M);

void argmax_over_cols(const float *mat, unsigned int *indices, const unsigned int N, const unsigned int M);

void fxp_argmax_over_cols(const int *mat, unsigned int *indices, const unsigned int N, const unsigned int M);

#endif //
