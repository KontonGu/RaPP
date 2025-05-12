import os
import argparse
import numpy as np
import json
import pickle




key_ops = {"nn.conv2d": 0, "nn.conv2d_transpose": 1, "nn.dense": 2, "nn.layer_norm": 3, "nn.relu": 4, "nn.bias_add": 5, "nn.max_pool2d": 6, "nn.adaptive_avg_pool2d": 7, "nn.batch_matmul": 8, "nn.softmax": 9, "nn.batch_norm": 10, "multiply": 11, 'add': 12, 'squeeze': 13, "reshape": 14, "transpose": 15}

key_ops_map = {v:k for k, v in key_ops.items()}

op_runtime_batch_size = 1  ## batch size for the operator runtime features (prior knowledge)
graph_ft_batch_size = 4
static_batch_size = 1
runtime_sm_ft_len = 6

sm_partitions = [5, 10, 30, 60, 80, 100]  ## 6 sm configuration for prior knowledge
graph_quotas = [20, 40, 60, 80, 100]  ## 5 quota configuration for prior knowledge

sm_data = []
ops_sm_features = {}

## collect the runtime features of key operators. For each opertor, it has a `nxm`` matrix
def get_ops_sm_features_orderd(model_name, runtime_prefix):
    ## load all runtime features of a model with different sms
    for sm_p in sm_partitions:
        file_name = model_name + "_" + str(op_runtime_batch_size) + "_" + str(sm_p) + ".json"
        file_path = os.path.join(runtime_prefix, file_name)
        with open(file_path, "r", encoding="utf-8") as f:
            data = json.load(f)
            sm_data.append(data)
    
    ## {op: `nxm` matrx} n, the number of a specific operator; m: the number of different sm configuration
    for op in key_ops:
        for i in range(len(sm_data)):
            col_vec = np.array(sm_data[i].get(op)).reshape(-1, 1)
            if op not in ops_sm_features:
                ops_sm_features[op] = col_vec
            else:
                ops_sm_features[op] = np.hstack((ops_sm_features[op], col_vec))


## load the operator static features and graph, extend the runtime features `n x m` to the operator feature in the graph 
def attach_features_to_ops(model_name, static_prefix):
    file_name = model_name + "_" + str(static_batch_size) + ".pkl"
    file_path = os.path.join(static_prefix, file_name)
    key_ops_cnt = {}
    for key in key_ops:
        key_ops_cnt[key] = 0
    ## load the operator graph and static features inside it
    with open(file_path, "rb") as f:
        G_loaded = pickle.load(f)
        
    ## attach sm runtime features to node of the operator graph (+6, originally 32)
    for node, data in G_loaded.nodes(data=True):
        attributes = data.get("attributes", [])
        one_hot = attributes[:16]
        index = np.argmax(one_hot)
        op_name = key_ops_map[index]
        op_len = ops_sm_features[op_name].shape[0]
        if op_name in key_ops and key_ops_cnt[op_name] < op_len:
            op_runtime_sm_ft = ops_sm_features[op_name][key_ops_cnt[op_name], :]
            key_ops_cnt[op_name] += 1
        else:
            op_runtime_sm_ft = np.zeros(runtime_sm_ft_len, dtype="float32")
        updated_features = np.concatenate((attributes, op_runtime_sm_ft))
        G_loaded.nodes[node]["attributes"] = updated_features
    return G_loaded
    # print(len(G_loaded.nodes))
    # for node, data in G_loaded.nodes(data=True):
        # print(data["attributes"])
        # print(data["attributes"].shape)
    

## attached the graph runtime feature (quota features) to the graph static features
def attach_features_to_graph(model_name, graph_prefix, G):
    ## load the latency profiling file of the model with sm = 100, differnt quota (20, 40, 60, 80, 100)
    sm_config = 100
    file_name = model_name + "_" + str(sm_config) + ".json"
    file_path = os.path.join(graph_prefix, file_name)
    with open(file_path, "r", encoding="utf-8") as f:
        data = json.load(f)
    profile_result = data.get("profiling_result")
    graph_quota_features = []
    for quota in graph_quotas:
        for item in profile_result:
            if item["batch"] == graph_ft_batch_size and item["quota"] == quota:
                graph_quota_features.append(item["latency"])
    # print(G.graph['graph_static'])
    updated_graph_static_ft = np.concatenate((G.graph['graph_static'], graph_quota_features))
    G.graph['graph_static'] = updated_graph_static_ft    ## 5 (graph_static_feature) + 5 (graph_runtime_feature) + 2 (batch, quota) = 12
    # print(G.graph['graph_static'].shape)
    # print(graph_quota_features)
    return G


def store_model_features(model_name, output_prefix, G):
    store_name = model_name + ".pkl";
    file_path = os.path.join(output_prefix, store_name)
    with open(file_path, "wb") as f:
        pickle.dump(G, f)
    print(f"Finished: {file_path}")


def main():
    parser = argparse.ArgumentParser(description="assemble both runtime and static && oprator and graph features.")
    parser.add_argument("--model_name", type=str, help="the name of the model for operators's feature extraction")
    # parser.add_argument("--prefix_dir", type=str, default="./", help="the prefix path to file storing assembled feature.")
    parser.add_argument("--runtime_prefix", type=str, default="./data/op_runtime_features_all_models")
    parser.add_argument("--static_prefix", type=str, default="./data/static_features_all_models")
    parser.add_argument("--graph_prefix", type=str, default="./data/graph_features_all_models")
    parser.add_argument("--output_prefix", type=str, default="./data/model_features")
    args = parser.parse_args()
    
    
    get_ops_sm_features_orderd(args.model_name, args.runtime_prefix)
    model_graph = attach_features_to_ops(args.model_name, args.static_prefix)
    model_graph = attach_features_to_graph(args.model_name, args.graph_prefix, model_graph)
    store_model_features(args.model_name, args.output_prefix, model_graph)
    
    

if __name__ == "__main__":
    main()