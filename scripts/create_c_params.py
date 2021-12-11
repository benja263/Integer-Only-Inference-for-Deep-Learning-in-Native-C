"""
Script for writing param header and source files in C with weights and amax values calculate in python
"""
import argparse
import subprocess

import torch

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Script for post-training quantization of a pre-trained model",
                                     formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--filename', help='filename', type=str, default='convnet_mnist_quant.th')
    parser.add_argument('--fxp_value', help='fxp value for quantization', type=int, default=16)
    parser.add_argument('--num_bits', help='number of bits', type=int, default=8)

    args = parser.parse_args()

    IS_CONVNET = True if 'convnet' in args.filename else False

    saved_stats = torch.load('../saved_models/' + args.filename)
    state_dict = saved_stats['state_dict']
    hidden_sizes = saved_stats['hidden_sizes']
    
    # create header file
    with open('../src/nn_params.h', 'w') as f:
        f.write('/*******************************************************************\n')
        f.write('@file nn_params.h\n*  @brief variable prototypes for model parameters and amax values\n*\n*\n')
        f.write('*  @author Benjamin Fuhrer\n*\n')
        f.write('*******************************************************************/\n')
        f.write('#ifndef NN_PARAMS\n#define NN_PARAMS\n\n')

        f.write(f'#define INPUT_DIM {28*28}\n')
        for idx, hidden_size in enumerate(hidden_sizes, start=1):
            f.write(f'#define H_MLP{idx} {hidden_size}\n')
        f.write('#define H1 28\n#define W1 28\n')
        f.write('#define C1 1\n#define C2 32\n#define C3 64\n')
        f.write(f'#define OUTPUT_DIM {10}\n')
        f.write(f'#define FXP_VALUE {args.fxp_value}\n\n')
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

        f.write('\n#endif // end of NN_PARAMS\n')

    # create source file
    with open('../src/nn_params.c', 'w') as f:
        f.write('#include "nn_params.h"\n\n\n')

        for layer_idx in range(1, 4):
            name = f'layer_{layer_idx}_s_x'
            fxp_value = (state_dict[name] * (2**args.fxp_value)).round()
            f.write(f"const int {name} = {int(fxp_value)};\n\n")

            name = f'layer_{layer_idx}_s_x_inv'
            fxp_value = (state_dict[name] * (2**args.fxp_value)).round()
            f.write(f"const int {name} = {int(fxp_value)};\n\n")

            name = f'layer_{layer_idx}_s_w_inv'
            fxp_value = (state_dict[name] * (2**args.fxp_value)).round()
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
