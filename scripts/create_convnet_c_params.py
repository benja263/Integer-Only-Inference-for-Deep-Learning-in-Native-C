"""
Script for writing param header and source files in C with weights and amax values calculate in python
"""
import argparse
import subprocess

import torch

def get_output_dim(input_dim, kernel_size, stride):
            output_dim = (input_dim -(kernel_size-1) - 1) / stride
            return int(output_dim + 1)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Script for post-training quantization of a pre-trained model",
                                     formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--filename', help='filename', type=str, default='convnet_mnist_quant.th')
    parser.add_argument('--num_bits', help='number of bits', type=int, default=8)

    args = parser.parse_args()

    saved_stats = torch.load('../saved_models/' + args.filename)
    state_dict = saved_stats['state_dict']
    channel_sizes = saved_stats['channel_sizes']

    H1_conv = get_output_dim(28, kernel_size=3, stride=1)
    W1_conv = H1_conv
    H1_pool = get_output_dim(H1_conv, kernel_size=2, stride=2)
    W1_pool = H1_pool

    H2_conv = get_output_dim(H1_pool, kernel_size=3, stride=1)
    W2_conv = H2_conv
    H2_pool = get_output_dim(H2_conv, kernel_size=2, stride=2)
    W2_pool = H2_pool


    
    # create header file
    with open('../src/convnet_params.h', 'w') as f:
        f.write('/*******************************************************************\n')
        f.write('@file convnet_params.h\n*  @brief variable prototypes for model parameters and amax values\n*\n*\n')
        f.write('*  @author Benjamin Fuhrer\n*\n')
        f.write('*******************************************************************/\n')
        f.write('#ifndef CONVNET_PARAMS\n#define CONVNET_PARAMS\n\n')

        f.write(f'#define INPUT_DIM {28*28}\n')
        f.write(f'#define H1 28\n#define W1 28\n#define H1_conv {H1_conv}\n#define W1_conv {W1_conv}\n#define H1_pool {H1_pool}\n#define W1_pool {W1_pool}\n')
        f.write(f'#define H2_conv {H2_conv}\n#define W2_conv {W2_conv}\n#define H2_pool {H2_pool}\n#define W2_pool {W2_pool}\n')
        f.write(f'#define C0 1\n#define C1 {channel_sizes[0]}\n#define C2 {channel_sizes[1]}\n')
        f.write(f'#define OUTPUT_DIM {10}\n\n')
        f.write('#include <stdint.h>\n\n\n')


        f.write('// quantization/dequantization constants\n')

        for layer_idx in range(1, 4):

            
            name = f'layer_{layer_idx}_s_x'
            f.write(f"extern const int {name};\n")

            name = f'layer_{layer_idx}_s_x_inv'
            f.write(f"extern const int {name};\n")

            name = f'layer_{layer_idx}_s_w_inv'
            value = state_dict[name]
            f.write(f"extern const int {name}[{len(value)}];\n")

        f.write('// Layer quantized parameters\n')
        for layer_idx in range(1, 4):
            name = f'layer_{layer_idx}_weight'
            param = state_dict[f'layer_{layer_idx}_weight']
            f.write(f"extern const int8_t {name}[{len(param.flatten())}];\n")

        f.write('\n#endif // end of CONVNET_PARAMS\n')

    # create source file
    with open('../src/convnet_params.c', 'w') as f:
        f.write('#include "convnet_params.h"\n\n\n')

        for layer_idx in range(1, 4):
            name = f'layer_{layer_idx}_s_x'
            fxp_value = (state_dict[name] * (2**16)).round()
            f.write(f"const int {name} = {int(fxp_value)};\n\n")

            name = f'layer_{layer_idx}_s_x_inv'
            fxp_value = (state_dict[name] * (2**16)).round()
            f.write(f"const int {name} = {int(fxp_value)};\n\n")

            name = f'layer_{layer_idx}_s_w_inv'
            fxp_value = (state_dict[name] * (2**16)).round()
            f.write(f"const int {name}[{len(fxp_value)}] = {{")

            for idx in range(len(fxp_value)):
                f.write(f"{int(fxp_value[idx])}")
                if idx < len(fxp_value) - 1:
                     f.write(", ")
            f.write("};\n\n")

        for layer_idx in range(1, 4):
                name = f'layer_{layer_idx}_weight'
                param = state_dict[f'layer_{layer_idx}_weight']
                param = param.flatten()
                f.write(f"const int8_t {name}[{len(param)}] = {{")
                for idx in range(len(param)):
                    f.write(f"{param[idx]}")
                    if idx < len(param) - 1:
                        f.write(", ")
                f.write("};\n")
