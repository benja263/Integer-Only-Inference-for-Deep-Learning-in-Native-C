/*******************************************************************
@file nn.h
 *  @brief Function prototypes for neural network layers
 *
 *
 *  @author Benjamin Fuhrer
 *
*******************************************************************/
#ifndef NN_H
#define NN_H

#include <stdint.h>

void linear_layer(const int *x, const int8_t *w, int *output, const int x_amax_quant,
                  const int *w_amax_dequant, const int x_amax_dequant, const unsigned int N,  const unsigned int K,
                  const unsigned int M, const unsigned int not_output);
/**
 * @brief A neural network linear layer withthout bias  Y = ReLU(XW)
 *  x is quantized before multiplication with w and then dequantized per-row granulity prior to the activation function
 * 
 * @param x - NxK input matrix
 * @param w - KxM layer weight matrix
 * @param output - NxM output matrix
 * @param x_amax_quant - amax value for quantization of input matrix
 * @param x_w_amax_dequant - 1XM amax values for dequantization of Z=XW
 * @param N
 * @param K
 * @param M
 * @param not_output - boolean value if layer is output layer (no activation)
 * 
 * @return Void
 */

void conv2d_layer(const int *x, const int8_t *w,int *output, const int x_amax_quant, const int *w_amax_dequant, const int x_amax_dequant,
                  const unsigned int N, const unsigned int C_in, const unsigned int C_out, const int H, const int W,
                  const int H_conv, const int W_conv, const int k_size_h, const int k_size_w,  const int stride_h, const int stride_w);
/**
 * @brief A neural network 2D convolutional layer with ReLU activation function
 *  x is quantized before the convolution operation and then dequantized with per-column granulity prior to the activation function
 * 
 * @param x - (N, C_in, H, W) input tensor
 * @param w - (C_out, C_in, H, W) weight tensor
 * @param output - (N, C_out, H_conv, W_conv) output tensor_in
 * @param x_amax_quant - amax value for input tensor quantization
 * @param w_amax_dequant - amax per channel values for weight tensor dequantization
 * @param x_amax_dequant - amax value for input tensor dequantization (= 1 / x_amax_quant)
 *
 * @return Void
 */ 


#endif 

