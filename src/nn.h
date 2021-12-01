/*******************************************************************
@file nn.h
 *  @brief Function prototypes to create and run an MLP for inference
 *  with only integers (8-bit integers and 32-bit integers
 *  in fixed-point)
 *
 *  @author Benjamin Fuhrer
 *
*******************************************************************/
#ifndef NN_H
#define NN_H

#include "nn_params.h"

void linear_layer(const int *x, const int8_t *w, int *output, const int x_amax_quant,
                  const int *x_w_amax_dequant, const unsigned int N,  const unsigned int K,
                  const unsigned int M, const unsigned int not_output);
/**
 * @brief A neural network linear layer with bias included within the layer parameters Y = ReLU(XW)
 *  x is quantized before multiplication with w and then dequantized prior to the activation function
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
 */


void run_mlp(int *x, const unsigned int N, unsigned int *class_indices);
/**
 * @brief Function to run an mlp for classification
 * 
 * @param x - NxK input matrix
 * @param N
 * @param class_indices - Nx1 vector for storing class index prediction
 */

void run_convnet(int *x, const unsigned int N, unsigned int *class_indices);
#endif 
