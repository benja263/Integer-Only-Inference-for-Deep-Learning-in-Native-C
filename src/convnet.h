/*******************************************************************
@file convnet.h
 *  @brief Function prototypes to create and run a convolutional neural network for inference
 *  with only integers (8-bit integers and 32-bit integers
 *  in fixed-point)
 *
 *  @author Benjamin Fuhrer
 *
*******************************************************************/
#ifndef CONVNET_H
#define CONVNET_H

#define BATCH_SIZE 1 // don't use larger batches to avoid stack overflow

#include "convnet_params.h"

void run_convnet(const int *x, unsigned int *class_indices);
/**
 * @brief A function to run a pre-specified convolutional neural network with relu activation function and max pooling
 * 
 * @param x - input tensor
 * @param class_indices  - Nx1 vector for storing class index prediction, where N = batch size
 */

#endif 
