#ifndef NN_PARAMS
#define NN_PARAMS

#define INPUT_DIM 784
#define HIDDEN_1 128
#define HIDDEN_2 64
#define OUTPUT_DIM 10
#define FXP_VALUE 16

#include <stdint.h>


// quantization/dequantization constants
extern const float net_0_wx_scale[128];
extern const int fxp_net_0_wx_scale[128];
extern const float net_0_input;
extern const int fxp_net_0_input;
extern const float net_2_wx_scale[64];
extern const int fxp_net_2_wx_scale[64];
extern const float net_2_input;
extern const int fxp_net_2_input;
extern const float net_4_wx_scale[10];
extern const int fxp_net_4_wx_scale[10];
extern const float net_4_input;
extern const int fxp_net_4_input;
// Layer quantized parameters
extern const int8_t net_0_weight[100480];
extern const int8_t net_2_weight[8256];
extern const int8_t net_4_weight[650];

#endif // end of NN_PARAMS
