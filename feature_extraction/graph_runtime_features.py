import os
import argparse
import time
import sys
import json

import torch
import torchvision.models as models
from torchvision.models import get_model



prior_batch_size = 4
batch_sizes = [1, 2, 4, 8]
# batch_sizes = [1]
quota_configs = []
client_name = "client1"
memory_limit = 16777216000

libhas_path = "/home/ubuntu/konton_ws/RaPP/libhas.so.1"

profiling_result = []



def get_end2end_latency_with_sm_quota(model_name, input_shape, sm_partition, prefix_dir, quota_config_path):
    model = get_model(name=model_name, weights="DEFAULT")
    model = model.eval()   
    device = torch.device("cuda")
    model.to(device)
    for bs in batch_sizes:
        for quota in quota_configs:
            quota_c = quota * 1.0 /100.0
            with open(quota_config_path, "w") as f:
                f.write("1\n")
                f.write(f'{client_name} {quota_c:.2g} {quota_c:.2g} {sm_partition} {memory_limit}')
            
            in_shape = (bs, *input_shape)
            in_data = torch.randn(in_shape)
            in_data = in_data.to(device) 
            
            ## warm-up
            for _ in range(20):
                _ = model(in_data)
            torch.cuda.synchronize()
            
            
            ## profiling
            new_in_data = in_data
            profile_times = 180
            times = []
            for i in range(profile_times):
                torch.cuda.synchronize()
                start_time = time.time()
                res = model(new_in_data)
                torch.cuda.synchronize() 
                end_time = time.time()
                times.append((end_time - start_time)*1000.0)
                time.sleep(0.005) ## interval to simulate independent profiling
            avg_time = sum(times)/len(times)
            
            print(f"model_name={model_name}, batch_size={bs}, sm_quota={sm_partition}:{quota}, avg_time={avg_time}")
            tmp_dict = {"batch": bs, "sm": sm_partition, "quota": quota, "latency": avg_time}
            profiling_result.append(tmp_dict)
    
    file_name = model_name + "_" + str(sm_partition) + ".json"
    file_path = os.path.join(prefix_dir, file_name)      
    with open(file_path, "w") as f:
        data = {
            "model_name": model_name,
            "profiling_result": profiling_result
        }
        json.dump(data, f, indent=4)
    os._exit(0)


def main():
    parser = argparse.ArgumentParser(description="extract graph runtime features.")
    parser.add_argument("--model_name", type=str, help="the name of the model for operators's feature extraction")
    parser.add_argument("--input_shape", type=str, help="the shape of the model input")
    parser.add_argument("--sm_partition", type=int, help="the sm percentage to use.")
    parser.add_argument("--quota_config_path", type=str, default="/home/ubuntu/konton_ws/HAS-FaST-Manager/config/clients_config.txt")
    parser.add_argument("--prefix_dir", type=str, default="./", help="the prefix path to store feature file.")
    args = parser.parse_args()
    
    os.environ["CUDA_MPS_ACTIVE_THREAD_PERCENTAGE"] = str(args.sm_partition)
    
    # GPU_CLIENT_PORT=56001 GPU_CLIENT_NAME="client1" LD_PRELOAD=
    os.environ["GPU_CLIENT_PORT"] = str(56001)
    os.environ["GPU_CLIENT_NAME"] = client_name
    os.environ["LD_PRELOAD"] = libhas_path
    
    in_shape = [int(item) for item in args.input_shape.split(',')]
    
    quota_interval = 10
    quota_val = 10
    while quota_val <= 100:
        quota_configs.append(quota_val)
        quota_val += quota_interval

    get_end2end_latency_with_sm_quota(args.model_name, in_shape, args.sm_partition, args.prefix_dir, args.quota_config_path)


if __name__ == "__main__":
    main()