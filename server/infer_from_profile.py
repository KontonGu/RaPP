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
    new_graph.graph["graph_static"] = np.concatenate((ori_graph_static_ft, input_graph_ft)).astype(np.float32) 
    infer_data = from_networkx(new_graph, all)
    
    latency = predict("./checkpoints/epoch=183-step=8351760.ckpt", infer_data)
    
    print(f"predict latency for model={model_name}, batch={batch}, quota={quota}, sm={sm},  latency={latency}")
    
    
    

def main():
    parser = argparse.ArgumentParser(description="The script for model inference.")
    parser.add_argument("--model_name", type=str, default="resnet50", help="the directory to the infer")
    parser.add_argument("--batch_size", type=int)
    parser.add_argument("--quota", type=int)
    parser.add_argument("--sm", type=int)
    parser.add_argument("--model_prefix", type=str, default="./model_features_all_models")
    
    args = parser.parse_args()
    
    input_shape = [3, 224, 224]
    infer(args.model_name, args.batch_size, args.quota, args.sm, args.model_prefix)
    


if __name__ == '__main__':
    main()