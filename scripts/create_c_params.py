import argparse
import torch
import subprocess

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Script for post-training quantization of a pre-trained model",
                                     formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--filename', help='filename', type=str, default='mlp_mnist_quant.th')

    args = parser.parse_args()

    saved_stats = torch.load('../saved_models/' + args.filename)
    state_dict = saved_stats['state_dict']
    h_size = saved_stats['h_size']
    amax = saved_stats['amax']

    # quantize all parameters
    for name, param in state_dict.items():
        param_amax = amax[name]
        # quantize 
        if 'weight' in name:
            param = param.T
        param = param * (127 / param_amax)
        state_dict[name] = torch.clamp(param.round(), min=-127, max=127).to(int)

    for idx in range(0, 2 + len(h_size) + 1, 2):
        amax[f'net.{idx}.wx_scale'] = (amax[f'net.{idx}.input'] * amax[f'net.{idx}.weight']) / (127**2)
        amax[f'net.{idx}.bias_scale'] = amax[f'net.{idx}.bias'] / (127)

    
    
    with open('../src/nn_params.h', 'w') as f:
        f.write('#ifndef NN_PARAMS\n#define NN_PARAMS\n\n')
        f.write(f'#define INPUT_DIM {28*28}\n#define HIDDEN_DIM {len(h_size)}\n#define OUTPUT_DIM {10}\n\n')
        f.write('#include "ml_math.h"\n\n\n')

        

        f.write(f'const unsigned int hidden_dims[{len(h_size)}] = {{')
        for idx in range(len(h_size)):
            f.write(f'{h_size[idx]}')
            if idx < len(h_size) - 1:
                f.write(', ')
        f.write('};\n\n')
        for name, param in state_dict.items():
            param = param.flatten()
            f.write(f"const int8_t {name.replace('.', '_')}[{len(param)}] = {{")
            for idx in range(len(param)):
                 f.write(f"{param[idx]}")
                 if idx < len(param) - 1:
                     f.write(", ")
            f.write("};\n\n")

        for layer_idx in range(0, 2 + len(h_size) + 1, 2):
            name = f'net.{layer_idx}.wx_scale'.replace('.', '_')
            value = amax[f'net.{layer_idx}.wx_scale']
            f.write(f"const float {name}[{len(value)}] = {{")

            for idx in range(len(value)):
                f.write(f"{value[idx]}")
                if idx < len(value) - 1:
                     f.write(", ")
            f.write("};\n\n")

            name = f'net.{layer_idx}.bias_scale'.replace('.', '_')
            value = amax[f'net.{layer_idx}.bias_scale']
            f.write(f"const float {name} = {value};\n")

            name = f'net.{layer_idx}.input'.replace('.', '_')
            value = amax[f'net.{layer_idx}.input']
            f.write(f"const float {name} = {value};\n\n")

        f.write('#endif // end of NN_PARAMS')
    # p = subprocess.Popen(args)