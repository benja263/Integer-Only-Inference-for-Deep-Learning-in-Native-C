#ifndef ML_MATH.H
#define ML_MATH.H

#define MAX(a,b) (((a)>(b))?(a):(b))
#define MIN(a,b) (((a)<(b))?(a):(b))
#define INT8_MAX 127
#define INT8_MIN -127


#include <stdint.h>

void mat_mult(const int8_t *mat_l, const int8_t *mat_r, int *result, const unsigned int N, const unsigned int K, const unsigned int M);

void _mat_vec_add(float *mat, const float *vec, const unsigned int N, const unsigned int M);

void _mat_vec_mult(float *mat, const float *vec, const unsigned int N, const unsigned int M);

void relu(float *mat, const int size);

void quantize(float *mat, int8_t *mat_q, const float amax, const unsigned int size);

void dequantize_bias(const int8_t *bias_q, float *bias, const float amax, const unsigned int size);

void dequantize_per_row(int *mat_q, float *mat, const float *amax, const unsigned int N, const unsigned int M);

void max_over_cols(float *mat, float *values, const unsigned int N, const unsigned int M);

#endif //