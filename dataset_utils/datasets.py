"""
This file contains datasets that coverts trajectories in files of ex{i}.npz into samples.
Each sample contains inputs and targets of models. 
"""

import torch
import numpy as np
import json
import os
import time

import torch.utils
import torch.utils.data

class HyperEl_datasets(torch.utils.data.Dataset):
    def __init__(self, path):
        self.path = path
        self.meta = json.loads(open(os.path.join(path, 'metadata.json')).read())
        self.files = self.meta['files']
        self.num_samples = sum(self.files[f] - 1 for f in self.files)

    @property
    def avg_nodes_per_sample(self):
        total_nodes = 0
        total_samples = 0
        for fname, num_steps in self.files.items():
            data = np.load(os.path.join(self.path, fname))
            total_nodes += data['mesh_pos'].shape[1] * (num_steps - 1)
            total_samples += (num_steps - 1)

        return total_nodes / total_samples


    def idx_to_file(self, sample_id):
        for fname, num_steps in self.files.items():
            if sample_id < (num_steps - 1): return fname, sample_id
            else: sample_id -= (num_steps - 1)
        raise IndexError()

    def __len__(self): return self.num_samples

    def __getitem__(self, idx : int) -> dict:
        fname, sid = self.idx_to_file(idx)
        data = np.load(os.path.join(self.path, fname))

        return dict(
            cells=torch.LongTensor(data['cells'][sid, ...]),
            node_type=torch.LongTensor(data['node_type'][sid, ...]),
            mesh_pos=torch.Tensor(data['mesh_pos'][sid, ...]),
            world_pos=torch.Tensor(data['world_pos'][sid, ...]),
            target_world_pos=torch.Tensor(data['world_pos'][sid + 1, ...]),
            stress=torch.Tensor(data['stress'][sid, ...])
        )

class IncompNS_datasets(torch.utils.data.Dataset):
    def __init__(self, path):
        self.path = path
        self.meta = json.loads(open(os.path.join(path, 'metadata.json')).read())
        self.files = self.meta['files']
        self.num_samples = sum(self.files[f] - 1 for f in self.files)

    @property
    def avg_nodes_per_sample(self):
        total_nodes = 0
        total_samples = 0
        for fname, num_steps in self.files.items():
            data = np.load(os.path.join(self.path, fname))
            total_nodes += data['mesh_pos'].shape[1] * (num_steps - 1)
            total_samples += (num_steps - 1)

        return total_nodes / total_samples


    def idx_to_file(self, sample_id):
        for fname, num_steps in self.files.items():
            if sample_id < (num_steps - 1): return fname, sample_id
            else: sample_id -= (num_steps - 1)
        raise IndexError()

    def __len__(self): return self.num_samples

    def __getitem__(self, idx : int) -> dict:
        fname, sid = self.idx_to_file(idx)
        data = np.load(os.path.join(self.path, fname))

        return dict(
            cells=torch.LongTensor(data['cells'][sid, ...]),
            node_type=torch.LongTensor(data['node_type'][sid, ...]),
            mesh_pos=torch.Tensor(data['mesh_pos'][sid, ...]),
            velocity=torch.Tensor(data['velocity'][sid, ...]),
            target_velocity=torch.Tensor(data['velocity'][sid + 1, ...]),
            pressure=torch.Tensor(data['pressure'][sid, ...])
        )

class Cloth_datasets(torch.utils.data.Dataset):
    def __init__(self, path):
        self.path = path
        self.meta = json.loads(open(os.path.join(path, 'metadata.json')).read())
        self.files = self.meta['files']
        self.num_samples = sum(self.files[f] - 2 for f in self.files)

    @property
    def avg_nodes_per_sample(self):
        total_nodes = 0
        total_samples = 0
        for fname, num_steps in self.files.items():
            data = np.load(os.path.join(self.path, fname))
            total_nodes += data['mesh_pos'].shape[1] * (num_steps - 2)
            total_samples += (num_steps - 2)

        return total_nodes / total_samples


    def idx_to_file(self, sample_id):
        for fname, num_steps in self.files.items():
            if sample_id < (num_steps - 2): return fname, sample_id
            else: sample_id -= (num_steps - 2)
        raise IndexError()

    def __len__(self): return self.num_samples

    def __getitem__(self, idx : int) -> dict:
        fname, sid = self.idx_to_file(idx)
        data = np.load(os.path.join(self.path, fname))

        return dict(
            cells=torch.LongTensor(data['cells'][sid, ...]),
            node_type=torch.LongTensor(data['node_type'][sid, ...]),
            mesh_pos=torch.Tensor(data['mesh_pos'][sid, ...]),
            world_pos=torch.Tensor(data['world_pos'][sid + 1, ...]),
            prev_world_pos=torch.Tensor(data['world_pos'][sid, ...]),
            target_world_pos=torch.Tensor(data['world_pos'][sid + 2, ...])
        )
    
def get_dataloader(path, 
                   model = "Cloth",
                   split = "train",
                   shuffle = True,
                   pre_fetch = None):
    """
    根据不同的模型使用不同的数据类
    """
    path = os.path.join(path,split)
    if model == "Cloth":
        Datasets = Cloth_datasets
    elif model== "IncompNS":
        Datasets = IncompNS_datasets
    elif model == "HyperEl":
        Datasets = HyperEl_datasets
    else:
        raise ValueError("The dataset type doesn't exist.")
    
    ds = Datasets(path)
    if pre_fetch is None:
        return torch.utils.data.DataLoader(ds, batch_size=1, shuffle = shuffle)
    return torch.utils.data.DataLoader(ds, batch_size=1, shuffle = shuffle, prefetch_factor=pre_fetch, num_workers=4, pin_memory=True)

def  get_batch_dataloader(path, 
                   model = "Cloth",
                   split = "train",
                   batch_size = 1, 
                   shuffle = True,
                   pre_fetch = None):
    pass


if __name__ == "__main__":
    # ds = deforming_datasets("D:\project_summary\Graduation Project\\tmp\datasets_np\deforming_plate\\train")
    # ds = cloth_datasets("D:\project_summary\Graduation Project\\tmp\datasets_np\\flag_simple\\train")
    # ds = flow_datasets("D:\project_summary\Graduation Project\\tmp\datasets_np\\cylinder_flow\\train")
    dl = get_dataloader("D:\project_summary\Graduation Project\\tmp\datasets_np\deforming_plate",model="HyperEl",split="train",pre_fetch=2)
    dl = iter(dl)
    start_time = time.time()
    for _ in range(10):
        next(dl)["node_type"]
    end_time = time.time()
    
    execution_time = (end_time - start_time)/10
    print(f"运行时间: {execution_time} 秒")
