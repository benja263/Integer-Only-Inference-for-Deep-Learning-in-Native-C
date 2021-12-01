/*******************************************************************
@file nn_math.h
 *  @brief Function prototypes for mathematical functions
 *
 *
 *  @author Benjamin Fuhrer
 *
*******************************************************************/
#ifndef NN_MATH_H
#define NN_MATH_H

#define MAX(a,b) (((a)>(b))?(a):(b))
#define MIN(a,b) (((a)<(b))?(a):(b))

#define MIN_INT -2147483648
#define _INT8_MAX 127
#define _INT8_MIN -127
#define MIN_FLOAT -99999999

#define ROUND_CONST (1 << (FXP_VALUE - 1)) // = 0.5 to before right shifting to improve rounding

#include "nn_params.h"

void mat_mult(const int8_t *mat_l, const int8_t *mat_r, int *result, const unsigned int N, const unsigned int K, const unsigned int M);
/**
 * @brief Calculates matrix multiplication as: Y = XW
 *  
 * 
 * @param mat_l - left matrix (X), size NxK
 * @param mat_r - right matrix (W), size (K+1)xM, the last row of W contains the bias vector
 * @param result - output matrix (Y), size NxM
 * @param N - number of rows in X
 * @param K - number of columns/rows in X/W
 * @param M - number of columns in W
 * @return Void
 */

int get_output_dim(int input_dim, int padding, int kernel_size, int stride, int dilation);

void conv2d(int8_t *x, int8_t *w, int *y, int N, int C_in, int C_out, int H, int W, int H_new, int W_new,
            int k_size_h, int k_size_w,  int stride_h, int stride_w, int padding_h,
            int padding_w, int dilation_h, int dilation_w);

void pooling2d(int *x, int *y, int N, int C_out, int H, int W, int H_new, int W_new,
            int k_size_h, int k_size_w,  int stride_h, int stride_w, int padding_h,
            int padding_w, int dilation_h, int dilation_w); 

void _broadcast_mat_vec_mult(int *mat, const int *vec, const unsigned int N, const unsigned int M);
/**
* @brief in place element-wise multplication of an 1xM row vector and an matrix NxM matrix , such that 
         each element of column m in mat is multiplied by the m-th element in vec
* 
* @param mat - NxM matrix
* @param vec - 1xM row vector
* @param N
* @param M
* @return Void
*/

void relu(int *tensor_in, const unsigned int size);
/**
 * @brief ReLU activation function
 * 
 * @param tensor_in - input tensor
 * @param size - product of all dimensions of tensor
 * @return Void
 */

void quantize(const int *mat_in, int8_t *mat_q, const int amax, const unsigned int size);
/**
 * @brief Scale quantization of a matrix by a single amax value
 * 
 * @param mat_in - NxM input matrix
 * @param mat_q - NxM output matrix
 * @param amax - amax value
 * @param size - size of vectorized matrix (MxN)
 * @return Void
 */

void dequantize_per_row(int *mat_in, const int *amax, const unsigned int N, const unsigned int M);
/**
 * @brief Scale dequantization with per-row granulity
 * Each row is multiplied by the corresponding column amax value
 * offline calculate reciprocal(amax) so we can replace division by multiplication
 * 
 * @param mat_in - NxM input matrix to dequantize
 * @param amax -1XM row vector of amax values
 * @param N
 * @param M
 * @return Void
*/

void dequantize_per_channel(int *tensor_in, const int *amax, const unsigned int N, const unsigned int C, const unsigned int K);

void argmax_over_cols(const int *mat_in, unsigned int *indices, const unsigned int N, const unsigned int M);
/**
 * @brief Calculate argmax per columns of an NxM matrix
 * 
 * @param mat_in - NxM input matrix
 * @param indices - 1xM indices to store argmax of each column
 * @param N
 * @param M
 * @return Void
 */

#endif //

