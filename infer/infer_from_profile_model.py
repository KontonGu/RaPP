import argparse
import pickle
import numpy as np
import copy
import os

import torch
from torch_geometric.utils.convert import from_networkx
from torch_geometric.data import Data, Batch

from rapp_model import RaPPModel


def predict(model_file, infer_data):
    model = RaPPModel()
    cpt = torch.load(model_file)
    model.load_state_dict(cpt["state_dict"])
    model.eval()
    
    batch = Batch()
    batch = batch.from_data_list([infer_data])
    output = model(batch)
    
    result = output.data.cpu().numpy()[0]
    return result
    

def infer(model_name, batch, quota, sm, model_prefix):
    feature_file = model_name + ".pkl"
    feature_file_path = os.path.join(model_prefix, feature_file)
    with open(feature_file_path, 'rb') as f:
        graph = pickle.load(f)

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
    infer_data = from_networkx(new_graph, all)
    
    # latency = predict("./checkpoint/epoch=5-step=224280.ckpt", infer_data)
    latency = predict("./lightning_logs/version_1/checkpoints/epoch=183-step=8351760.ckpt", infer_data)
    
    print(f"predict latency for model={model_name}, batch={batch}, quota={quota}, sm={sm},  latency={latency}")
    return f"batch:{batch}, sm:{sm}, quota:{quota}, latency:{latency}"
    
    
    

def main():
    parser = argparse.ArgumentParser(description="The script for model inference.")
    parser.add_argument("--model_name", type=str, default="convnext_base")
    parser.add_argument("--model_prefix", type=str, default="../data/model_features_all_models")
    
    args = parser.parse_args()
    
    all_results = []
    input_shape = [3, 224, 224]
    sm_configs = np.arange(5, 101, 5)
    quota_configs = np.arange(10, 101, 5)
    batch_sizes = [1, 2, 4 ,8]
    
    batch = 4
    output_file = f"./inference/{args.model_name}_{batch}_predict.txt"
    for sm in sm_configs:
        for quota in quota_configs:
            result = infer(args.model_name, batch, quota, sm, args.model_prefix)
            all_results.append(result)
    with open(output_file, "w") as file:
        for item in all_results:
            file.write(item+"\n")
    
    
    
    


if __name__ == '__main__':
    main()