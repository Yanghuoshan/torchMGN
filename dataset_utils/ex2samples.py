from datasets import *
import dataclasses
import time
import os

if __name__ == "__main__":
    datasets_name = "deforming_plate"
    split = "train"
    in_path = "D:\\project_summary\\Graduation Project\\tmp\\datasets_np"
    out_path = "D:\\project_summary\\Graduation Project\\tmp\\datasets_single_sample"

    in_path = os.path.join(in_path,datasets_name,split)
    out_path = os.path.join(out_path,datasets_name,split)

    print("in_path:",in_path)
    print("out_path:",out_path)
    start_time = time.time()

    if datasets_name == "deforming_plate":
        ds = deforming_datasets(in_path)
    elif datasets_name == "flag_simple":
        ds = cloth_datasets(in_path)
    elif datasets_name == "cylinder_flow":
        ds = flow_datasets(in_path)
    else:
        raise ValueError("datasets不支持")
    
    print(list(ds.files.items())[0][1] - 1)
    dl = torch.utils.data.DataLoader(
        ds,
        shuffle=True,
        batch_size=1
    )

    gen = iter(dl)
    os.makedirs(out_path, exist_ok=True)

    count = 0
    for i,s in enumerate(gen):
        conut = count + 1
        num_nodes = s["node_type"].size(1)
        print(f'Sample {i}: {num_nodes} nodes')
        batch_np = {
            k: v[0].numpy()
            for k, v in s.items()
        }

        np.savez_compressed(f'{out_path}/{i}.npz', **batch_np)

    print("The total number of samples:",count)
    end_time = time.time()
    print(f"运行时间: {execution_time} 秒")
    