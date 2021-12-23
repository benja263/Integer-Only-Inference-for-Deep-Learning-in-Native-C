
#include "convnet.h"
#include "nn.h"
#include "nn_math.h"

void run_convnet(const int *x, unsigned int *class_indices)
{
    
    int out_conv1[BATCH_SIZE*C1*H1_conv*W1_conv];

    conv2d_layer(x, layer_1_weight, out_conv1, layer_1_s_x, layer_1_s_w_inv, layer_1_s_x_inv,
                 BATCH_SIZE, C0, C1, H1, W1, H1_conv, W1_conv,
                 3, 3,  1, 1);

    int out_pool1[BATCH_SIZE*C1*H1_pool*W1_pool];
    pooling2d(out_conv1, out_pool1, BATCH_SIZE, C1, H1_conv, W1_conv, H1_pool, W1_pool, 2, 2,  2, 2);
    
    int out_conv2[BATCH_SIZE*C2*H2_conv*W2_conv];
    conv2d_layer(out_pool1, layer_2_weight, out_conv2, layer_2_s_x, layer_2_s_w_inv, layer_2_s_x_inv,
                 BATCH_SIZE, C1, C2, H1_pool, W1_pool, H2_conv, W2_conv,
                 3, 3,  1, 1);

    int out_pool2[BATCH_SIZE*C1*H1_pool*W1_pool];
    pooling2d(out_conv2, out_pool2, BATCH_SIZE, C2, H2_conv, W2_conv, H2_pool, W2_pool, 2, 2,  2, 2);

    int output[BATCH_SIZE*OUTPUT_DIM];
    linear_layer(out_pool2, layer_3_weight, output, layer_3_s_x,
                  layer_3_s_w_inv, layer_3_s_x_inv,
                  BATCH_SIZE, C2*H2_pool*W2_pool, OUTPUT_DIM, 0);

    argmax_over_cols(output, class_indices, BATCH_SIZE, OUTPUT_DIM);
}
