import os
import subprocess
import argparse

import torch
import torchvision.models as models
from torchvision.models import list_models, get_model

# sm_partitions = [5, 10, 20, 30, 60, 80, 100]
sm_partitions = [5, 10, 30, 60, 80, 100]
# sm_partitions = [5]
# batch_sizes_list = [1, 2, 4]
batch_sizes_list = [2]

all_models = ['alexnet', 'convnext_base', 'convnext_large', 'convnext_small', 'convnext_tiny', 'densenet121', 'densenet161', 'densenet169', 'densenet201', 'efficientnet_b0', 'efficientnet_b1', 'efficientnet_b2', 'efficientnet_b3', 'efficientnet_b4', 'efficientnet_b5', 'efficientnet_b6', 'efficientnet_b7', 'efficientnet_v2_l', 'efficientnet_v2_m', 'efficientnet_v2_s', 'googlenet', 'inception_v3', 'maxvit_t', 'mnasnet0_5', 'mnasnet0_75', 'mnasnet1_0', 'mnasnet1_3', 'mobilenet_v2', 'mobilenet_v3_large', 'mobilenet_v3_small', 'regnet_x_16gf', 'regnet_x_1_6gf', 'regnet_x_32gf', 'regnet_x_3_2gf', 'regnet_x_400mf', 'regnet_x_800mf', 'regnet_x_8gf', 'regnet_y_128gf', 'regnet_y_16gf', 'regnet_y_1_6gf', 'regnet_y_32gf', 'regnet_y_3_2gf', 'regnet_y_400mf', 'regnet_y_800mf', 'regnet_y_8gf', 'resnet101', 'resnet152', 'resnet18', 'resnet34', 'resnet50', 'resnext101_32x8d', 'resnext101_64x4d', 'resnext50_32x4d', 'shufflenet_v2_x0_5', 'shufflenet_v2_x1_0', 'shufflenet_v2_x1_5', 'shufflenet_v2_x2_0', 'squeezenet1_0', 'squeezenet1_1', 'swin_b', 'swin_s', 'swin_t', 'swin_v2_b', 'swin_v2_s', 'swin_v2_t', 'vgg11', 'vgg11_bn', 'vgg13', 'vgg13_bn', 'vgg16', 'vgg16_bn', 'vgg19', 'vgg19_bn', 'vit_b_16', 'vit_b_32', 'vit_h_14', 'vit_l_16', 'vit_l_32', 'wide_resnet101_2', 'wide_resnet50_2']



log_file_dir = './runtime_op.log'

def exec_runtime_feature_extract(model_name, prefix_dir):
    for batch_size in batch_sizes_list:
        for sm in sm_partitions:
            command = f'python3 op_runtime_feature_extractor.py --model_name {model_name} --input_shape "3,224,224" --sm_partition {sm} --batch_size {batch_size} --prefix_dir {prefix_dir} > {log_file_dir} 2>&1'
            try:
                result = subprocess.run(
                    command,
                    shell=True, 
                    capture_output=True, 
                    text=True
                )
                print(f"[SUCCESS] Model: {model_name}_{sm}\nOutput: {result.stdout}")
            except subprocess.CalledProcessError as e:
                print(f"[ERROR] Model: {model_name}_{sm}\nExit Code: {e.returncode}\nError: {e.stderr}")
            except Exception as e:
                print(f"[ERROR] Model: {model_name}_{sm}\nType: {type(e).__name__}\nDetails: {str(e)}")
            
        
    
def main():
    parser = argparse.ArgumentParser(description="get operator runtime features for all models.")
    # parser.add_argument("--batch_size", type=int, default=1, help="the batch size for profiling")
    # parser.add_argument("--prefix_dir", type=str, default="./data/runtime_features")
    parser.add_argument("--prefix_dir", type=str, default="./data/op_runtime_new")
    args = parser.parse_args()
    
    
    for model in all_models:
        exec_runtime_feature_extract(model, args.prefix_dir)

    
if __name__ == "__main__":
    main()