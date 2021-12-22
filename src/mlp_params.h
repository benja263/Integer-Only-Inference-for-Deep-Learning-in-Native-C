/*******************************************************************
@file mlp_params.h
*  @brief variable prototypes for model parameters and amax values
*
*
*  @author Benjamin Fuhrer
*
*******************************************************************/
#ifndef MLP_PARAMS
#define MLP_PARAMS

#define INPUT_DIM 784
#define H1 128
#define H2 64
#define OUTPUT_DIM 10
#define FXP_VALUE 16
#define BATCH_SIZE 10

#include <stdint.h>


// quantization/dequantization constants
extern const int layer_1_s_x;
extern const int layer_1_s_x_inv;
extern const int layer_1_s_w_inv[128];
extern const float layer_1_s_x_f;
extern const float layer_1_s_x_inv_f;
extern const float layer_1_s_w_inv_f[128];
extern const int layer_2_s_x;
extern const int layer_2_s_x_inv;
extern const int layer_2_s_w_inv[64];
extern const float layer_2_s_x_f;
extern const float layer_2_s_x_inv_f;
extern const float layer_2_s_w_inv_f[64];
extern const int layer_3_s_x;
extern const int layer_3_s_x_inv;
extern const int layer_3_s_w_inv[10];
extern const float layer_3_s_x_f;
extern const float layer_3_s_x_inv_f;
extern const float layer_3_s_w_inv_f[10];
// Layer quantized parameters
extern const int8_t layer_1_weight[100352];
extern const int8_t layer_2_weight[8192];
extern const int8_t layer_3_weight[640];

#endif // end of MLP_PARAMS
