# server.py
from flask import Flask, request, jsonify
import argparse
import pickle
import numpy as np
import copy
import os

import torch
from torch_geometric.utils.convert import from_networkx
from torch_geometric.data import Data, Batch
import time

from rapp_model import RaPPModel

app = Flask(__name__)
model = None
model_file = "./checkpoints/epoch=183-step=8351760.ckpt"
model_ft_files = "./model_features_all_models"


def initialize():
    global model
    model = RaPPModel()
    cpt = torch.load(model_file)
    model.load_state_dict(cpt["state_dict"])
    model.eval()
    
    
    
def predict(infer_data):
    global model
    batch = Batch()
    batch = batch.from_data_list([infer_data])
    output = model(batch)
    result = output.data.cpu().numpy()[0]
    return result
    

def infer(model_name, batch, quota, sm):
    feature_file = model_name + ".pkl"
    feature_file_path = os.path.join(model_ft_files, feature_file)
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
    
    latency = predict(infer_data)
    print(f"predict latency for model={model_name}, batch={batch}, quota={quota}, sm={sm},  latency={latency}")
    return float(latency)
    

@app.route('/rapp_query', methods=['POST'])
def run_model():
    data = request.get_json()

    if not data:
        return jsonify({'error': 'No JSON payload received'}), 400

    model_name = data.get('model_name')
    batch_size = data.get('batch_size')
    sm = data.get('sm')
    quota = data.get('quota')
    
    latency = infer(model_name, batch_size, quota, sm)

    return jsonify({
        'status': 'success',
        'predict': {
            'model_name': model_name,
            'sm': sm,
            'batch_size': batch_size,
            'quota': quota,
            'latency': latency,
        }
    })

if __name__ == '__main__':
    initialize()
    app.run(debug=True, port=8080)