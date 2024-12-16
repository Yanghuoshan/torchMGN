"""
This file contains datasets that coverts trajectories in files of ex{i}.npz into samples.
Each sample contains inputs and targets of models. 
"""

import torch
import numpy as np
import json
import os
import time
import sys
sys.path.append('../')

import torch.utils
import torch.utils.data

from model_utils.common import build_graph_HyperEl

class HyperEl_datasets(torch.utils.data.Dataset):
    def __init__(self, path, is_data_graph = False):
        self.path = path
        self.meta = json.loads(open(os.path.join(path, 'metadata.json')).read())
        self.files = self.meta['files']
        self.num_samples = sum(self.files[f] - 1 for f in self.files)
        self.is_data_graph = is_data_graph

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
        if not self.is_data_graph:
            return dict(
                cells=torch.LongTensor(data['cells'][sid, ...]),
                node_type=torch.LongTensor(data['node_type'][sid, ...]),
                mesh_pos=torch.Tensor(data['mesh_pos'][sid, ...]),
                world_pos=torch.Tensor(data['world_pos'][sid, ...]),
                target_world_pos=torch.Tensor(data['world_pos'][sid + 1, ...]),
                stress=torch.Tensor(data['stress'][sid, ...])
            )
        else:
            d = dict(
                cells=torch.LongTensor(data['cells'][sid, ...]),
                node_type=torch.LongTensor(data['node_type'][sid, ...]),
                mesh_pos=torch.Tensor(data['mesh_pos'][sid, ...]),
                world_pos=torch.Tensor(data['world_pos'][sid, ...]),
                target_world_pos=torch.Tensor(data['world_pos'][sid + 1, ...]),
                stress=torch.Tensor(data['stress'][sid, ...])
            )
            graph = build_graph_HyperEl(d)

            world_pos = d['world_pos']
            target_world_pos = d['target_world_pos']
            target_stress = d['stress']
            cur_position = world_pos
            target_position = target_world_pos
            target_velocity = target_position - cur_position
            
            target = torch.concat((target_velocity, target_stress), dim=1) 

            return [graph, target, d['node_type']]
    

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
    
        
def my_collate_fn(batch): # cumstom collate fn
        # batch [data1, data2...]
        return batch
    
def get_dataloader(path, 
                   model = "Cloth",
                   split = "train",
                   shuffle = True,
                   prefetch = 0,
                   is_data_graph = False):
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
    
    ds = Datasets(path, is_data_graph)
    if prefetch == 0:
        return torch.utils.data.DataLoader(ds, batch_size=1, shuffle = shuffle, collate_fn=my_collate_fn)
    return torch.utils.data.DataLoader(ds, batch_size=1, shuffle = shuffle, prefetch_factor=prefetch, num_workers=4, pin_memory=True, collate_fn=my_collate_fn)



if __name__ == "__main__":
    # ds = deforming_datasets("D:\project_summary\Graduation Project\\tmp\datasets_np\deforming_plate\\train")
    # ds = cloth_datasets("D:\project_summary\Graduation Project\\tmp\datasets_np\\flag_simple\\train")
    # ds = flow_datasets("D:\project_summary\Graduation Project\\tmp\datasets_np\\cylinder_flow\\train")
    dl = get_dataloader("D:\project_summary\Graduation Project\\tmp\datasets_np\deforming_plate",model="HyperEl",split="train",prefetch=1,is_output_graph=True)
    dl = iter(dl)
    start_time = time.time()
    for _ in range(10):
        next(dl)
    end_time = time.time()
    
    execution_time = (end_time - start_time)/10
    print(f"运行时间: {execution_time} 秒")
