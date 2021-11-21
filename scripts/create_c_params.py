"""
Script for writing param header and source files in C with weights and amax values calculate in python
"""
import argparse
import subprocess

import torch

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Script for post-training quantization of a pre-trained model",
                                     formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--filename', help='filename', type=str, default='mlp_mnist_quant.th')
    parser.add_argument('--fxp_value', help='fxp value', type=int, default=16)
    parser.add_argument('--num_bits', help='number of bits', type=int, default=8)

    args = parser.parse_args()

    saved_stats = torch.load('../saved_models/' + args.filename)
    state_dict = saved_stats['state_dict']
    hidden_sizes = saved_stats['hidden_sizes']
    amax = saved_stats['amax']

    for idx in range(0, 2 + len(hidden_sizes) + 1, 2):
        # invert s_wx such that we can replace fixed-point division with multiplication
        amax[f'net.{idx}.s_wx_inv'] = (amax[f'net.{idx}.s_x'] * amax[f'net.{idx}.s_w']) / ((2**args.num_bits - 1)**2)
        amax[f'net.{idx}.s_x'] = (2**args.num_bits - 1) / amax[f'net.{idx}.s_x']

    
    # create header file
    with open('../src/nn_params.h', 'w') as f:
        f.write('/*******************************************************************\n')
        f.write('@file nn_params.h\n*  @brief variable prototypes for model parameters and amax values\n*\n*\n')
        f.write('*  @author Benjamin Fuhrer\n*\n')
        f.write('*******************************************************************/\n')
        f.write('#ifndef NN_PARAMS\n#define NN_PARAMS\n\n')

        f.write(f'#define INPUT_DIM {28*28}\n')
        for idx, hidden_size in enumerate(hidden_sizes, start=1):
            f.write(f'#define HIDDEN_{idx} {hidden_size}\n')
        f.write(f'#define OUTPUT_DIM {10}\n')
        f.write(f'#define FXP_VALUE {args.fxp_value}\n\n')
        f.write('#include <stdint.h>\n\n\n')


        f.write('// quantization/dequantization constants\n')

        for layer_idx in range(0, 2 + len(hidden_sizes) + 1, 2):

            name = f'net.{layer_idx}.s_wx_inv'.replace('.', '_')
            value = amax[f'net.{layer_idx}.s_wx_inv']
            f.write(f"extern const int {name}[{len(value)}];\n")

            name = f'net.{layer_idx}.s_x'.replace('.', '_')
            value = amax[f'net.{layer_idx}.s_x']
            f.write(f"extern const int {name};\n")


        f.write('// Layer quantized parameters\n')
        for name, param in state_dict.items():
            f.write(f"extern const int8_t {name.replace('.', '_')}[{len(param.flatten())}];\n")

        f.write('\n#endif // end of NN_PARAMS\n')

    # create source file
    with open('../src/nn_params.c', 'w') as f:
        f.write('#include "nn_params.h"\n\n\n')

        for layer_idx in range(0, 2 + len(hidden_sizes) + 1, 2):
            fxp_value = (amax[f'net.{layer_idx}.s_x'] * (2**args.fxp_value)).round()
            name = f'net.{layer_idx}.s_x'.replace('.', '_')
            f.write(f"const int {name} = {int(fxp_value)};\n\n")

            name = f'net.{layer_idx}.s_wx_inv'.replace('.', '_')
            fxp_value = (amax[f'net.{layer_idx}.s_wx_inv'] * (2**args.fxp_value)).round()
            f.write(f"const int {name}[{len(fxp_value)}] = {{")

            for idx in range(len(fxp_value)):
                f.write(f"{fxp_value[idx]}")
                if idx < len(fxp_value) - 1:
                     f.write(", ")
            f.write("};\n\n")


        for name, param in state_dict.items():
                param = param.T.flatten()
                f.write(f"const int8_t {name.replace('.', '_')}[{len(param)}] = {{")
                for idx in range(len(param)):
                    f.write(f"{param[idx]}")
                    if idx < len(param) - 1:
                        f.write(", ")
                f.write("};\n")
