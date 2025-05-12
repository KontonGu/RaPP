import torch
import torchvision.models as models
from torchvision.models import get_model
import tvm
from tvm import relay

import argparse
import networkx as nx
import numpy as np
import os
import pickle

## Operators dominating the majority of computational time
key_ops = {"nn.conv2d": 0, "nn.conv2d_transpose": 1, "nn.dense": 2, "nn.layer_norm": 3, "nn.relu": 4, "nn.bias_add": 5, "nn.max_pool2d": 6, "nn.adaptive_avg_pool2d": 7, "nn.batch_matmul": 8, "nn.softmax": 9, "nn.batch_norm": 10, "multiply": 11, 'add': 12, 'squeeze': 13, "reshape": 14, "transpose": 15}


    
class StaticFeatureExtractor():
    def __init__(self):
        self.node_dict = {}
        self.nodes_list = []
        self.ops = []
        self.graph = nx.DiGraph()
        self.shape_len = 4
        self.attr_len = 12
        self.op_type_len = len(key_ops)
        
    def _traverse_node(self, node):
        if node in self.node_dict:
            return
        self.node_dict[node] = len(self.node_dict)
        
    def extract_features(self, expr):
        cnt = 0
        # out_file = open("model.txt", "w", encoding="utf-8")
        tvm.relay.analysis.post_order_visit(expr, lambda x: self._traverse_node(x))
        for node, node_id in sorted(self.node_dict.items(), key=lambda x: x[1]):
            if isinstance(node, tvm.relay.Call) and isinstance(node.op, tvm.ir.Op):
                self.ops.append(node.op.name)
                op_shape_list = []
                # print(f"------------node_op: {node.op.name}----------------")
                for arg in node.args:
                    # print(f">node_op has arg: {node.op.name}<")
                    # if(node.op.name == "multiply"):
                        # print(f"args type:<<< {type(arg)} >>>")
                    if isinstance(arg, tvm.relay.Var):
                        # print(f"--realy.Var: {node.op.name}\n args: {arg.type_annotation.shape}")
                        try:
                            op_shape = np.array(list([int(x) for x in arg.type_annotation.shape]), dtype='float32')
                            op_shape_list.append(op_shape)
                        except:
                            continue
                if op_shape_list:
                    print("KONTON_Name:", node.op.name)
                    ###--- one-hot for operator type ---
                    op_type = np.zeros(self.op_type_len, dtype="float32")
                    if node.op.name in key_ops:
                        op_type[key_ops[node.op.name]] = 1
                        
                    ###---- operator internal features  ----
                    attrs = {}
                    if hasattr(node.attrs, "keys"):
                        # print(f"op_name={node.op.name}, attrs: \n{node.attrs.keys()}")
                        for k in node.attrs.keys():
                            attrs[k] = getattr(node.attrs, k)
                    op_attr = np.zeros(self.attr_len, dtype="float32")
                    if "conv2d" in node.op.name:
                        op_attr = np.array(list([int(x) for x in attrs['kernel_size']]) + [int(attrs['channels'])] + 
                                              list([int(x) for x in attrs['strides']]) + list([int(x) for x in attrs['padding']]) + list([int(x) for x in attrs['dilation']]) + [int(attrs['groups'])], dtype="float32")
                        if op_attr.size < self.attr_len:
                            op_attr = np.concatenate(op_attr, np.zeros(self.attr_len - op_attr.size), dtype="float32")
                        elif op_attr.size > self.attr_len:
                            op_attr = op_attr[:self.attr_len]
                            
                    ###--- oeprator shape ---
                    op_shape = op_shape_list[-1]
                    if op_shape.size < self.shape_len:
                        op_shape = np.concatenate((op_shape, np.zeros(self.shape_len - op_shape.size, dtype="float32")))
                    elif op_shape.size > self.shape_len:
                        op_shape = op_shape[:self.shape_len]
                        
                    op_features = np.concatenate((op_type, op_attr, op_shape), dtype="float32")  ## 16 + 12 + 4 = 32
                    self.graph.add_node(str(node_id), attributes=op_features)
                    self.nodes_list.append(str(node_id))
                    print(f"{node.op.name}: {op_features}")
                    # out_file.write(f"{node_id}: {node.op.name}\n")
        # out_file.close()      
        for i in range(len(self.nodes_list) - 1):
            self.graph.add_edge(self.nodes_list[i], self.nodes_list[i+1])     
        return self.graph 
        # print(self.graph)
   
        
def extract_graph_static_features(mod_main, G, batch_size):
    opt_pass = relay.transform.InferType()
    assert isinstance(opt_pass, tvm.transform.Pass)
    mod = tvm.IRModule.from_expr(mod_main)
    mod = tvm.relay.transform.InferType()(mod)
    mod = opt_pass(mod)
    entry = mod["main"]
    func = entry if isinstance(mod_main, relay.Function) else entry.body
    
    mac = int(relay.analysis.get_total_mac_number(func))/1e9
    dtype = relay.analysis.all_dtypes(func)
    ops_freq = relay.analysis.list_op_freqs(mod)
    if "nn.relu" in ops_freq:
        relu_num = int(ops_freq["nn.relu"])
    else:
        relu_num = 0
    conv2d_num = int(ops_freq["nn.conv2d"])
    dense_num = int(ops_freq.get("nn.dense", 0))
    G.graph['graph_static'] = np.array([batch_size, mac, conv2d_num, dense_num, relu_num], dtype="float32")
    return G
    
    
# if node.op.name == "reshape":
#     print("reshape.")
#     if hasattr(node.attrs, "keys"):
#         for k in node.attrs.keys():
#             print(k, getattr(node.attrs, k))               
        
# cnt+=1
#                 if node.op.name not in key_ops.keys():
#                     print(node.op.name)
#                     cnt-=1
#         print(f"original_node length: {cnt}/{len(self.node_dict)}")    
   
def extract_ops_static_features(model_name, input_shape, batch_size, prefix_dir):
    model = get_model(name=model_name, weights="DEFAULT")
    in_shape = (batch_size, *input_shape)
    in_name = "input0"
    in_data = torch.randn(in_shape)
    model.eval()
    with torch.no_grad():
        script_mod = torch.jit.trace(model, in_data).eval()
    mod, params = tvm.relay.frontend.from_pytorch(script_mod, [(in_name, in_data.shape)])
    mod_main = mod["main"]
    
    extractor = StaticFeatureExtractor()
    G = extractor.extract_features(mod_main)
    G = extract_graph_static_features(mod_main, G, batch_size)
    store_name = model_name + "_" + str(batch_size)+".pkl";
    file_path = os.path.join(prefix_dir, store_name)
    with open(file_path, "wb") as f:
        pickle.dump(G, f)
    return G
    # print(type(mod_main))
    
    
def main():
    parser = argparse.ArgumentParser(description="extract operator static features.")
    parser.add_argument("--model_name", type=str, help="the name of the model for operators's feature extraction")
    parser.add_argument("--input_shape", type=str, help="the shape of the model input")
    parser.add_argument("--prefix_dir", type=str, default="./", help="the prefix path to store feature file.")
    args = parser.parse_args()
    
    in_shape = [int(item) for item in args.input_shape.split(',')]
    
    extract_ops_static_features(model_name=args.model_name, input_shape=in_shape, batch_size=1, prefix_dir=args.prefix_dir)

if __name__ == "__main__":
    main()
