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
#define HIDDEN_1 128
#define HIDDEN_2 64
#define OUTPUT_DIM 10
#define FXP_VALUE 16

#include <stdint.h>


// quantization/dequantization constants
extern const int net_0_s_wx_inv[128];
extern const int net_0_s_x;
extern const int net_2_s_wx_inv[64];
extern const int net_2_s_x;
extern const int net_4_s_wx_inv[10];
extern const int net_4_s_x;
// Layer quantized parameters
extern const int8_t net_0_weight[100480];
extern const int8_t net_2_weight[8256];
extern const int8_t net_4_weight[650];

#endif // end of NN_PARAMS
