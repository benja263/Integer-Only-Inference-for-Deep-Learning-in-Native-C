/*******************************************************************
@file convnet_params.h
*  @brief variable prototypes for model parameters and amax values
*
*
*  @author Benjamin Fuhrer
*
*******************************************************************/
#ifndef CONVNET_PARAMS
#define CONVNET_PARAMS

#define INPUT_DIM 784
#define H1 28
#define W1 28
#define H1_conv 26
#define W1_conv 26
#define H1_pool 13
#define W1_pool 13
#define H2_conv 11
#define W2_conv 11
#define H2_pool 5
#define W2_pool 5
#define C0 1
#define C1 16
#define C2 16
#define OUTPUT_DIM 10
#define FXP_VALUE 16
#define BATCH_SIZE 1

#include <stdint.h>


// quantization/dequantization constants
extern const int layer_1_s_x;
extern const int layer_1_s_x_inv;
extern const int layer_1_s_w_inv[16];
extern const float layer_1_s_x_f;
extern const float layer_1_s_x_inv_f;
extern const float layer_1_s_w_inv_f[16];
extern const int layer_2_s_x;
extern const int layer_2_s_x_inv;
extern const int layer_2_s_w_inv[16];
extern const float layer_2_s_x_f;
extern const float layer_2_s_x_inv_f;
extern const float layer_2_s_w_inv_f[16];
extern const int layer_3_s_x;
extern const int layer_3_s_x_inv;
extern const int layer_3_s_w_inv[10];
extern const float layer_3_s_x_f;
extern const float layer_3_s_x_inv_f;
extern const float layer_3_s_w_inv_f[10];
// Layer quantized parameters
extern const int8_t layer_1_weight[144];
extern const int8_t layer_2_weight[2304];
extern const int8_t layer_3_weight[4000];

#endif // end of CONVNET_PARAMS
