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

#define NUM_BITS 8
#define INT8_MAX_VALUE 127
#define FXP_VALUE 16
#define ROUND_CONST (1 << (FXP_VALUE - 1)) // = 0.5 to before right shifting to improve rounding

#include <stdint.h>

void mat_mult(const int8_t *mat_l, const int8_t *mat_r, int *result, const unsigned int N, const unsigned int K, const unsigned int  M);
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

int get_output_dim(int input_dim, int kernel_size, int stride);

void conv2d(const int8_t *x, const int8_t *w, int *y, int N, int C_in, int C_out, int H, int W, int H_new, int W_new,
            int k_size_h, int k_size_w,  int stride_h, int stride_w);

void pooling2d(int *x, int *y, int N, int C_out, int H, int W, int H_new, int W_new,
            int k_size_h, int k_size_w,  int stride_h, int stride_w); 

void relu(int *tensor_in, const unsigned int size);
/**
 * @brief ReLU activation function
 * 
 * @param tensor_in - input tensor
 * @param size - size of flattened tensor
 * @return Void
 */


void quantize(const int *tensor_in, int8_t *tensor_q, const int scale_factor,
              const int scale_factor_inv, const unsigned int size);
/**
 * @brief Scale quantization of a tensor by a single amax value
 * 
 * @param tensor_in - input tensor
 * @param tensor_q - output quantized tensor
 * @param scale_factor - 127 / amax
 * @param scale_factor_inv - 1 / scale_factor
 * @param size - size of flattened tensor
 * @return Void
 */

void dequantize_per_row(int *mat_in, const int *scale_factor_w_inv, const int scale_factor_x_inv, const unsigned int  N, const unsigned int  M);
/**
 * @brief Scale dequantization with per-row granulity
 * Each row is multiplied by the corresponding column amax value
 * offline calculate reciprocal(amax) so we can replace division by multiplication
 * 
 * @param mat_in - NxM input matrix to dequantize
 * @param scale_factor_w_inv -1XM row vector of layer's weight matrix scale factor values
 * @param scale_factor_x_inv - input inverse scale factor
 * @param N
 * @param M
 * @return Void
*/

void dequantize_per_channel(int *tensor_in, const int *amax_w, const int amax_x, const unsigned int N, const unsigned int C, const unsigned int K);
/**
 * @brief Scale dequantization with per-channel granulity
 * Each channel is multiplied by the corresponding channel amax value
 * offline calculate reciprocal(amax) so we can replace division by multiplication
 * 
 * @param tensor_in - input tensor to dequantize (N, C, ...)
 * @param amax -1XC row vector of amax values
 * @param N - number of samples
 * @param C - number of channels
 * @param K - number of remaining flattened dimensions
 * @return Void
*/

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

