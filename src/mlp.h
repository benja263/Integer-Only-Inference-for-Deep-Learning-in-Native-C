/*******************************************************************
@file mlp.h
 *  @brief Function prototypes to create and run an MLP for inference
 *  with only integers (8-bit integers and 32-bit integers
 *  in fixed-point)
 *
 *  @author Benjamin Fuhrer
 *
*******************************************************************/
#ifndef MLP_H
#define MLP_H

void run_mlp(const int *x, const unsigned int N, unsigned int *class_indices);
/**
 * @brief Function to run an mlp for classification
 * 
 * @param x - NxK input matrix
 * @param N
 * @param class_indices - Nx1 vector for storing class index prediction
 */


#endif 
