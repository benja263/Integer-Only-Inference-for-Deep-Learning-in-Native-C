/*******************************************************************
@file nn_params.h
*  @brief variable prototypes for model parameters and amax values
*
*
*  @author Benjamin Fuhrer
*
*******************************************************************/
#ifndef NN_PARAMS
#define NN_PARAMS

#define INPUT_DIM 784
#define H_MLP1 128
#define H_MLP2 64
#define H1 28
#define W1 28
#define C1 1
#define C2 32
#define C3 64
#define OUTPUT_DIM 10
#define FXP_VALUE 16

#include <stdint.h>


// quantization/dequantization constants
extern const int layer_1_s_x;
extern const int layer_1_s_x_inv;
extern const int layer_1_s_w_inv[32];
extern const int layer_2_s_x;
extern const int layer_2_s_x_inv;
extern const int layer_2_s_w_inv[64];
extern const int layer_3_s_x;
extern const int layer_3_s_x_inv;
extern const int layer_3_s_w_inv[10];
// Layer quantized parameters
extern const int8_t layer_1_weight[288];
extern const int8_t layer_2_weight[18432];
extern const int8_t layer_3_weight[16000];

#endif // end of NN_PARAMS
