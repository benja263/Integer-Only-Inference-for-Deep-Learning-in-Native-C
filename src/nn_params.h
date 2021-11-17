#ifndef NN_PARAMS
#define NN_PARAMS

#define INPUT_DIM 784
#define HIDDEN_1 64
#define HIDDEN_2 32
#define OUTPUT_DIM 10
#define FXP_VALUE 16

#include <stdint.h>


// quantization/dequantization constants
extern const float net_0_wx_scale[64];
extern const float net_0_input;
extern const float net_2_wx_scale[32];
extern const float net_2_input;
extern const float net_4_wx_scale[10];
extern const float net_4_input;
// Layer quantized parameters
extern const int8_t net_0_weight[50176];
extern const int8_t net_0_bias[64];
extern const int8_t net_2_weight[2048];
extern const int8_t net_2_bias[32];
extern const int8_t net_4_weight[320];
extern const int8_t net_4_bias[10];

#endif // end of NN_PARAMS
