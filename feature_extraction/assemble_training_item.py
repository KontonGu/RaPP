import argparse
import os
import pickle
import json
import numpy as np
from torch_geometric.utils.convert import from_networkx
import copy
import torch

all_models = ['alexnet', 'convnext_base', 'convnext_large', 'convnext_small', 'convnext_tiny', 'densenet121', 'densenet161', 'densenet169', 'densenet201', 'efficientnet_b0', 'efficientnet_b1', 'efficientnet_b2', 'efficientnet_b3', 'efficientnet_b4', 'efficientnet_b5', 'efficientnet_v2_l', 'efficientnet_v2_m', 'efficientnet_v2_s', 'googlenet', 'inception_v3', 'maxvit_t', 'mnasnet0_5', 'mnasnet0_75', 'mnasnet1_0', 'mnasnet1_3', 'mobilenet_v2', 'mobilenet_v3_large', 'mobilenet_v3_small', 'regnet_x_16gf', 'regnet_x_32gf', 'regnet_x_1_6gf', 'regnet_x_3_2gf', 'regnet_x_400mf', 'regnet_x_800mf', 'regnet_x_8gf', 'regnet_y_128gf', 'regnet_y_16gf', 'regnet_y_1_6gf', 'regnet_y_32gf', 'regnet_y_3_2gf', 'regnet_y_400mf', 'regnet_y_800mf', 'regnet_y_8gf', 'resnet101', 'resnet152', 'resnet18', 'resnet34', 'resnet50', 'resnext101_32x8d', 'resnext101_64x4d', 'resnext50_32x4d', 'shufflenet_v2_x0_5', 'shufflenet_v2_x1_0', 'shufflenet_v2_x1_5', 'shufflenet_v2_x2_0', 'squeezenet1_0', 'squeezenet1_1', 'vgg11', 'vgg11_bn', 'vgg13', 'vgg13_bn', 'vgg16', 'vgg16_bn', 'vgg19', 'vgg19_bn', 'wide_resnet101_2', 'wide_resnet50_2']


model_start_idx = {}
batch_sizes = [1, 2, 4, 8]
sm_config_num = 20
sm_configs = []
quota_config_num = 10
quota_configs = []




def generate_training_item(model_name, model_prefix, graph_prefix, output_prefix,):
    ## load the feature file (model features)
    feature_file = model_name + ".pkl"
    feature_file_path = os.path.join(model_prefix, feature_file)
    with open(feature_file_path, 'rb') as f:
        graph = pickle.load(f)
         
    model_cnt = model_start_idx[model_name]
    for batch in batch_sizes:
        for sm in sm_configs:
            ## load profiling data
            profile_file = model_name + "_" + str(sm) + ".json"
            profile_file_path = os.path.join(graph_prefix, profile_file)
            with open(profile_file_path, "r", encoding="utf-8") as f:
                profile_data = json.load(f)
            for quota in quota_configs:
                # print(f"now: batch={batch}, sm={sm}, quota={quota}")
                new_graph = copy.deepcopy(graph)
                ### add input to the node operator feature
                for node, data in new_graph.nodes(data=True):
                    input_ft = [float(batch), float(sm/100.0)]
                    ori_feature = new_graph.nodes[node]["attributes"]
                    new_graph.nodes[node]["attributes"] = np.concatenate((ori_feature, input_ft)).astype(np.float32)
                
                input_graph_ft = [float(batch), float(quota/100.0)]
                ori_graph_static_ft = new_graph.graph["graph_static"]
                ###  5 (graph_static_feature) + 5 (graph_runtime_feature) + 2 (batch, quota) = 12
                new_graph.graph["graph_static"] = np.concatenate((ori_graph_static_ft, input_graph_ft)).astype(np.float32)   
                
                new_graph.graph["y"] = np.float32(0.0)
                ## get the label
                profile_result = profile_data["profiling_result"]
                found = False
                for item in profile_result:
                    if item["batch"] == batch and item["sm"] == sm and item["quota"] == quota:
                        new_graph.graph["y"] = np.float32(item["latency"])
                        found = True
                if not found:
                    continue
                final_data = from_networkx(new_graph, all)
                final_file_name = str(model_cnt).zfill(6)+".pt"
                final_file_path = os.path.join(output_prefix, final_file_name)
                model_cnt += 1
                
                torch.save(final_data, final_file_path)
                
                

def main():
    parser = argparse.ArgumentParser(description="assemble both runtime and static && oprator and graph features.")
    parser.add_argument("--output_prefix", type=str, default="./data/model_dataset")
    parser.add_argument("--model_prefix", type=str, default="./data/model_features_all_models")
    parser.add_argument("--graph_prefix", type=str, default="./data/graph_features_all_models")
    args = parser.parse_args()
    
    
    quota_interval = 10
    quota_val = 10
    while quota_val <= 100:
        quota_configs.append(quota_val)
        quota_val += quota_interval
        
    sm_interval = 5
    sm_val = 5
    while sm_val <= 100:
        sm_configs.append(sm_val)
        sm_val += sm_interval
    
    for idx in range(len(all_models)):
        model_start_idx[all_models[idx]] = idx * len(batch_sizes) * len(sm_configs) * len(quota_configs)
    print(model_start_idx)
    
    for model_name in all_models:
        generate_training_item(model_name, args.model_prefix,  args.graph_prefix, args.output_prefix)
    
    
    
if __name__ == "__main__":
    main()