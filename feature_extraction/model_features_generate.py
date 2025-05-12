import argparse
import subprocess


all_models = ['alexnet', 'convnext_base', 'convnext_large', 'convnext_small', 'convnext_tiny', 'densenet121', 'densenet161', 'densenet169', 'densenet201', 'efficientnet_b0', 'efficientnet_b1', 'efficientnet_b2', 'efficientnet_b3', 'efficientnet_b4', 'efficientnet_b5', 'efficientnet_v2_l', 'efficientnet_v2_m', 'efficientnet_v2_s', 'googlenet', 'inception_v3', 'maxvit_t', 'mnasnet0_5', 'mnasnet0_75', 'mnasnet1_0', 'mnasnet1_3', 'mobilenet_v2', 'mobilenet_v3_large', 'mobilenet_v3_small', 'regnet_x_16gf', 'regnet_x_32gf', 'regnet_x_1_6gf', 'regnet_x_3_2gf', 'regnet_x_400mf', 'regnet_x_800mf', 'regnet_x_8gf', 'regnet_y_128gf', 'regnet_y_16gf', 'regnet_y_1_6gf', 'regnet_y_32gf', 'regnet_y_3_2gf', 'regnet_y_400mf', 'regnet_y_800mf', 'regnet_y_8gf', 'resnet101', 'resnet152', 'resnet18', 'resnet34', 'resnet50', 'resnext101_32x8d', 'resnext101_64x4d', 'resnext50_32x4d', 'shufflenet_v2_x0_5', 'shufflenet_v2_x1_0', 'shufflenet_v2_x1_5', 'shufflenet_v2_x2_0', 'squeezenet1_0', 'squeezenet1_1', 'vgg11', 'vgg11_bn', 'vgg13', 'vgg13_bn', 'vgg16', 'vgg16_bn', 'vgg19', 'vgg19_bn', 'wide_resnet101_2', 'wide_resnet50_2']

# 

def model_features_extract(model_name, output_prefix):
    command = f'python3 model_features.py --model_name {model_name} --output_prefix {output_prefix}'
    try:
        result = subprocess.run(
            command,
            shell=True, 
            capture_output=True, 
            text=True
        )
        print(f"[SUCCESS] Model: {model_name}\nOutput: {result.stdout}")
    except subprocess.CalledProcessError as e:
        print(f"[ERROR] Model: {model_name}\nExit Code: {e.returncode}\nError: {e.stderr}")
    except Exception as e:
        print(f"[ERROR] Model: {model_name}\nType: {type(e).__name__}\nDetails: {str(e)}")


def main():
    parser = argparse.ArgumentParser(description="get operator static features for all models.")
    parser.add_argument("--output_prefix", type=str, default="./data/model_features")
    args = parser.parse_args()
    
    
    for model in all_models:
        model_features_extract(model, args.output_prefix)

    
if __name__ == "__main__":
    main()