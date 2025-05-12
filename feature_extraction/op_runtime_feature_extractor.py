import torch
import torchvision.models as models
from torchvision.models import get_model
import tvm
from tvm import relay
from tvm.contrib.debugger import debug_executor
from tvm import tir

import argparse
import os
import time
import json

## Operators dominating the majority of computational time
key_ops = {"nn.conv2d": 0, "nn.conv2d_transpose": 1, "nn.dense": 2, "nn.layer_norm": 3, "nn.relu": 4, "nn.bias_add": 5, "nn.max_pool2d": 6, "nn.adaptive_avg_pool2d": 7, "nn.batch_matmul": 8, "nn.softmax": 9, "nn.batch_norm": 10, "multiply": 11, 'add': 12, 'squeeze': 13, "reshape": 14, "transpose": 15}

## the pattern of the nn.batch_norm
pattern = ["add", "rsqrt", "multiply", "nop", "multiply", "negative", "multiply", "add", "nop", "add"]


class RuntimeFeatureExtractor():
    def __init__(self, model_name, model, sm_percent, batch_size, in_name, in_data, prefix_dir):
        self.model_name = model_name
        self.model = model
        self.sm_partition = sm_percent
        self.batch_size = batch_size
        self.in_name = in_name
        self.in_data = in_data
        self.prefix_dir = prefix_dir
        
        
    def extract_features(self, mod, params):
        target = tvm.target.Target("cuda -arch=sm_70 -max_shared_memory_per_block=49152")
        with tvm.transform.PassContext(opt_level=0):
            lib = relay.build(mod, target=target, params=params)
        debug_mod = debug_executor.create(
            lib.get_graph_json(),
            lib.lib,
            tvm.cuda(0)
        )
        debug_mod.set_input(self.in_name, tvm.nd.array(self.in_data.numpy(), device=tvm.gpu(0)))
        debug_mod.run()
        profile_result = debug_mod.profile()
        csv_out = profile_result.csv()
        # with open("profile.csv", "w", encoding="utf-8") as f:
        #     f.write(csv_out)
        
        ### Get the execution time percentage of each operator in a model
        run_ops_list, run_ops_percent = self.extract_ops_percent(profile_result)
        ops_percent_dict = self.group_ops(run_ops_list, run_ops_percent)  ## {"key_op":[percent, ..]}  
        
        ### Get total model execution time based on SM partition
        latency = self.get_end2end_latency()
        # print(f"------------latency: {latency}.")
        ops_exec_time_dict = self.get_ops_exec_time(ops_percent_dict, latency)
        ops_exec_time_dict["latency"] = latency
        ops_exec_time_dict["batch_size"] = self.batch_size
        ops_exec_time_dict["model_name"] = self.model_name
        ops_exec_time_dict["sm_partition"] = self.sm_partition
        # print(ops_exec_time_dict)
        file_name = self.model_name+ "_" + str(self.batch_size) + "_" + str(self.sm_partition) + ".json"
        file_path = os.path.join(self.prefix_dir, file_name)
        with open(file_path, "w", encoding='utf-8') as f:
            json.dump(ops_exec_time_dict, f, ensure_ascii=False, indent=4)
        return ops_exec_time_dict
            
        
    ## convert serial of batch_norm sub-operations to batch_norm
    def extract_ops_percent(self, profile_result):
        ops_list = []
        ops_list_percent = []
        nop_cnt = 0
        if hasattr(profile_result, "calls"):
            # print("has_calls")
            for call_info in profile_result.calls:
                call_name = call_info["Name"]
                ops_list.append(call_name)
                ops_list_percent.append(call_info["Percent"].percent)
            new_ops, new_ops_percent = self.merge_batch_norm(ops_list, ops_list_percent)
            total_percent = 0.0
            for call_info in profile_result.calls:  
                total_percent += call_info["Percent"].percent
            for idx in range(len(new_ops_percent)):
                new_ops_percent[idx] /= total_percent 
                
            for i in range(len(new_ops)):
                print(f"{new_ops[i]}: {new_ops_percent[i]}")
        return new_ops, new_ops_percent
    
    ## merge sub-operators to a batch_norm operator
    def merge_batch_norm(self, ops_list, ops_list_percent):
        match_idx = []
        match_idx_time = {}
        i = 0
        while i <= len(ops_list) - len(pattern):
            j = 0
            i_cnt = i
            exe_time = 0.0
            for op in pattern:
                if op in ops_list[i_cnt]:
                    j+=1
                    i_cnt+=1
                    exe_time += ops_list_percent[i_cnt]
                    if j == len(pattern):
                        match_idx.append(i)
                        match_idx_time[i] = exe_time
                        break
                else:
                    break
            i+=1
        new_ops = []
        new_ops_percent = []
        i = 0
        bm_idx = 0
        while i < len(ops_list):
            if i not in match_idx:
                new_ops.append(ops_list[i])
                new_ops_percent.append(ops_list_percent[i])
                i+=1
            else:
                new_ops.append("tvmgen_default_fused_nn_batch_norm"+"_"+str(bm_idx))
                bm_idx += 1
                new_ops_percent.append(match_idx_time[i])
                i+=len(pattern)
        return new_ops, new_ops_percent


    def group_ops(self, ops_list, ops_list_percent):
        ops_dict = {}
        for op in key_ops:
            ops_dict[op] = []
        new_key_ops = sorted(key_ops, key=len, reverse=True)
        for i in range(len(ops_list)):
            run_ops = ops_list[i].replace('_', '.')
            for op in new_key_ops:
                m_op = op.replace('_', '.')
                if m_op in run_ops:
                    ops_dict[op].append(ops_list_percent[i])
                    break
        return ops_dict
    

    def get_end2end_latency(self):
        device = torch.device("cuda")
        self.model.to(device)
        in_data = self.in_data.to(device)
        ## warm-up
        for _ in range(20):
            _ = self.model(in_data)
        torch.cuda.synchronize()
        
        ## profiling
        new_in_data = in_data
        profile_times = 150
        times = []
        for i in range(profile_times):
            torch.cuda.synchronize()
            start_time = time.time()
            res = self.model(new_in_data)
            torch.cuda.synchronize() 
            end_time = time.time()
            times.append((end_time - start_time)*1000.0)
            time.sleep(0.2) ## interval to simulate independent profiling
        avg_time = sum(times)/len(times)
        return avg_time      
    
    def get_ops_exec_time(self, ops_percent_dict, latency):
        for op in ops_percent_dict:
            ops_percent_dict[op] = [pc * latency for pc in ops_percent_dict[op]]
        return ops_percent_dict
        
         
 
def extract_ops_runtime_features(model_name, input_shape, sm_partition, batch_size, prefix_dir):
    model = get_model(name=model_name, weights="DEFAULT")
    model = model.eval()
    in_shape = (batch_size, *input_shape)
    print(f"in_shape={in_shape}")
    in_name = "input0"
    in_data = torch.randn(in_shape)
    with torch.no_grad():
        script_mod = torch.jit.trace(model, in_data).eval()
    mod, params = tvm.relay.frontend.from_pytorch(script_mod, [(in_name, in_data.shape)])
    mod_main = mod["main"]
    
    extractor = RuntimeFeatureExtractor(model_name, model, sm_partition, batch_size, in_name, in_data, prefix_dir)
    extractor.extract_features(mod, params)


def main():
    parser = argparse.ArgumentParser(description="extract operator runtime features.")
    parser.add_argument("--model_name", type=str, help="the name of the model for operators's feature extraction")
    parser.add_argument("--input_shape", type=str, help="the shape of the model input")
    parser.add_argument("--batch_size", type=int, default=1, help="the batch size for profiling")
    parser.add_argument("--sm_partition", type=int, help="the sm percentage to use.")
    parser.add_argument("--prefix_dir", type=str, default="./", help="the prefix path to store feature file.")
    args = parser.parse_args()
    
    os.environ["CUDA_MPS_ACTIVE_THREAD_PERCENTAGE"] = str(args.sm_partition)
    
    in_shape = [int(item) for item in args.input_shape.split(',')]
    # print(in_shape)
    
    extract_ops_runtime_features(model_name=args.model_name, input_shape=in_shape, sm_partition=args.sm_partition, batch_size=args.batch_size, prefix_dir=args.prefix_dir)



if __name__ == "__main__":
    main()