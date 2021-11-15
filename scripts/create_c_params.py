import argparse
import torch
import subprocess

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Script for post-training quantization of a pre-trained model",
                                     formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--filename', help='filename', type=str, default='mlp_mnist_quant.th')
    parser.add_argument('--fxp_value', help='fxp value', type=int, default=16)
    parser.add_argument('--fxp_prec_value', help='fxp higher precision value', type=int, default=16)

    args = parser.parse_args()

    saved_stats = torch.load('../saved_models/' + args.filename)
    state_dict = saved_stats['state_dict']
    h_size = saved_stats['h_size']
    amax = saved_stats['amax']

    for idx in range(0, 2 + len(h_size) + 1, 2):
        amax[f'net.{idx}.wx_scale'] = (amax[f'net.{idx}.input'] * amax[f'net.{idx}.weight']) / (127**2)
        amax[f'net.{idx}.bias_scale'] = amax[f'net.{idx}.bias'] / (127)
        amax[f'net.{idx}.input'] = 127 / amax[f'net.{idx}.input']

    
    # create header file
    with open('../src/nn_params.h', 'w') as f:
        f.write('#ifndef NN_PARAMS\n#define NN_PARAMS\n\n')
        # f.write(f'#define INPUT_DIM {28*28}\n#define HIDDEN_DIM {len(h_size)}\n#define OUTPUT_DIM {10}\n')
        f.write(f'#define INPUT_DIM {28*28}\n#define OUTPUT_DIM {10}\n')
        f.write(f'#define FXP_VALUE {args.fxp_value}\n\n')
        f.write('#include <stdint.h>\n\n\n')


        f.write('// quantization/dequantization constants\n')

        for layer_idx in range(0, 2 + len(h_size) + 1, 2):
            name = f'net.{layer_idx}.bias_scale'.replace('.', '_')
            value = amax[f'net.{layer_idx}.bias_scale']
            f.write(f"extern const float {name};\n")

            fxp_name = 'fxp_' + name
            f.write(f"extern const int {fxp_name};\n")

            name = f'net.{layer_idx}.wx_scale'.replace('.', '_')
            value = amax[f'net.{layer_idx}.wx_scale']
            f.write(f"extern const float {name}[{len(value)}];\n")

            fxp_name = 'fxp_' + name
            f.write(f"extern const int {fxp_name}[{len(value)}];\n")

            name = f'net.{layer_idx}.input'.replace('.', '_')
            value = amax[f'net.{layer_idx}.input']
            f.write(f"extern const float {name};\n")

            fxp_name = 'fxp_' + name
            f.write(f"extern const int {fxp_name};\n")

        f.write('// Layer quantized parameters\n')
        for name, param in state_dict.items():
            param = param.flatten()
            f.write(f"extern const int8_t {name.replace('.', '_')}[{len(param)}];\n")

        f.write('\n#endif // end of NN_PARAMS\n')

    # create source file
    with open('../src/nn_params.c', 'w') as f:
        f.write('#include "nn_params.h"\n\n\n')

        for layer_idx in range(0, 2 + len(h_size) + 1, 2):
            name = f'net.{layer_idx}.input'.replace('.', '_')
            value = amax[f'net.{layer_idx}.input']
            f.write(f"const float {name} = {value};\n")

            fxp_value = (value * (2**args.fxp_value)).round()
            fxp_name = 'fxp_' + name
            f.write(f"const int {fxp_name} = {int(fxp_value)};\n\n")

            name = f'net.{layer_idx}.bias_scale'.replace('.', '_')
            value = amax[f'net.{layer_idx}.bias_scale']
            f.write(f"const float {name} = {value};\n")

            fxp_value = (value * (2**args.fxp_prec_value)).round()
            fxp_name = 'fxp_' + name
            f.write(f"const int {fxp_name} = {int(fxp_value)};\n\n")

            name = f'net.{layer_idx}.wx_scale'.replace('.', '_')
            value = amax[f'net.{layer_idx}.wx_scale']
            f.write(f"const float {name}[{len(value)}] = {{")

            for idx in range(len(value)):
                f.write(f"{value[idx]}")
                if idx < len(value) - 1:
                     f.write(", ")
            f.write("};\n\n")

            fxp_value = (value * (2**args.fxp_prec_value)).round()
            fxp_name = 'fxp_' + name
            f.write(f"const int {fxp_name}[{len(fxp_value)}] = {{")

            for idx in range(len(fxp_value)):
                f.write(f"{int(fxp_value[idx])}")
                if idx < len(fxp_value) - 1:
                     f.write(", ")
            f.write("};\n\n")

        for name, param in state_dict.items():
            if 'weight' in name:
                param = param.T
            param = param.flatten()
            f.write(f"const int8_t {name.replace('.', '_')}[{len(param)}] = {{")
            for idx in range(len(param)):
                 f.write(f"{param[idx]}")
                 if idx < len(param) - 1:
                     f.write(", ")
            f.write("};\n")
