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

import h5pickle as h5py

from tqdm import trange

from model_utils.common import build_graph_HyperEl, build_graph_Cloth

class HyperEl_single_dataset(torch.utils.data.Dataset):
    def __init__(self, path, is_data_graph = False):
        self.path = path
        self.meta = json.loads(open(os.path.join(path, 'metadata.json')).read())
        self.files = self.meta['files']
        self.num_samples = sum(self.files[f] - 1 for f in self.files)
        if is_data_graph:
            self.return_item = self.return_graph
        else:
            self.return_item = self.return_dict

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
        return self.return_item(data, sid)
        
    def return_dict(self, data, sid):
        return dict(
                cells=torch.LongTensor(data['cells'][sid, ...]),
                node_type=torch.LongTensor(data['node_type'][sid, ...]),
                mesh_pos=torch.Tensor(data['mesh_pos'][sid, ...]),
                world_pos=torch.Tensor(data['world_pos'][sid, ...]),
                target_world_pos=torch.Tensor(data['world_pos'][sid + 1, ...]),
                stress=torch.Tensor(data['stress'][sid, ...])
            )

    def return_graph(self, data, sid):
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
    

class IncompNS_single_dataset(torch.utils.data.Dataset):
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


class Cloth_single_dataset(torch.utils.data.Dataset):
    def __init__(self, path, is_data_graph = False):
        self.path = path
        self.meta = json.loads(open(os.path.join(path, 'metadata.json')).read())
        self.files = self.meta['files']
        self.num_samples = sum(self.files[f] - 2 for f in self.files)
        if is_data_graph:
            self.return_item = self.return_graph
        else:
            self.return_item = self.return_dict

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

        return self.return_item(data, sid)
    
    def return_dict(self, data, sid):
        return dict(
            cells=torch.LongTensor(data['cells'][sid, ...]),
            node_type=torch.LongTensor(data['node_type'][sid, ...]),
            mesh_pos=torch.Tensor(data['mesh_pos'][sid, ...]),
            world_pos=torch.Tensor(data['world_pos'][sid + 1, ...]),
            prev_world_pos=torch.Tensor(data['world_pos'][sid, ...]),
            target_world_pos=torch.Tensor(data['world_pos'][sid + 2, ...])
        )

    def return_graph(self, data, sid):
        d = dict(
            cells=torch.LongTensor(data['cells'][sid, ...]),
            node_type=torch.LongTensor(data['node_type'][sid, ...]),
            mesh_pos=torch.Tensor(data['mesh_pos'][sid, ...]),
            world_pos=torch.Tensor(data['world_pos'][sid + 1, ...]),
            prev_world_pos=torch.Tensor(data['world_pos'][sid, ...]),
            target_world_pos=torch.Tensor(data['world_pos'][sid + 2, ...])
        )
        graph = build_graph_Cloth(d)

        world_pos = d['world_pos']
        prev_world_pos = d['prev_world_pos']
        target_world_pos = d['target_world_pos']

        cur_position = world_pos
        prev_position = prev_world_pos
        target_position = target_world_pos
        target = target_position - 2 * cur_position + prev_position

        return [graph, target, d['node_type']]
    
        
class Cloth_trajectory_dataset(torch.utils.data.Dataset):
    def __init__(self, path, is_data_graph = False, trajectory_index = 0):
        self.path = path
        self.meta = json.loads(open(os.path.join(path, 'metadata.json')).read())
        self.files = self.meta['files']
        self.trajectory_index = trajectory_index
        self.fname = list(self.files.keys())[trajectory_index]
        self.num_samples = self.files[self.fname] - 2
        if is_data_graph:
            self.return_item = self.return_graph
        else:
            self.return_item = self.return_dict

    def __len__(self): return self.num_samples

    def __getitem__(self, idx : int) -> dict:
        data = np.load(os.path.join(self.path, self.fname))
        return self.return_item(data, idx)
    
    def return_dict(self, data, sid):
        return dict(
            cells=torch.LongTensor(data['cells'][sid, ...]),
            node_type=torch.LongTensor(data['node_type'][sid, ...]),
            mesh_pos=torch.Tensor(data['mesh_pos'][sid, ...]),
            world_pos=torch.Tensor(data['world_pos'][sid + 1, ...]),
            prev_world_pos=torch.Tensor(data['world_pos'][sid, ...]),
            target_world_pos=torch.Tensor(data['world_pos'][sid + 2, ...])
        )

    def return_graph(self, data, sid):
        d = dict(
            cells=torch.LongTensor(data['cells'][sid, ...]),
            node_type=torch.LongTensor(data['node_type'][sid, ...]),
            mesh_pos=torch.Tensor(data['mesh_pos'][sid, ...]),
            world_pos=torch.Tensor(data['world_pos'][sid + 1, ...]),
            prev_world_pos=torch.Tensor(data['world_pos'][sid, ...]),
            target_world_pos=torch.Tensor(data['world_pos'][sid + 2, ...])
        )
        graph = build_graph_Cloth(d)

        world_pos = d['world_pos']
        prev_world_pos = d['prev_world_pos']
        target_world_pos = d['target_world_pos']

        cur_position = world_pos
        prev_position = prev_world_pos
        target_position = target_world_pos
        target = target_position - 2 * cur_position + prev_position

        return [graph, target, d['node_type']]



class Cloth_single_dataset_hdf5(torch.utils.data.Dataset):
    def __init__(self, path, is_data_graph = False):
        self.path = path
        self.meta = json.loads(open(os.path.join(path, 'metadata.json')).read())
        self.files = self.meta['files']
        self.num_samples = sum(self.files[f] - 2 for f in self.files)

        self.hdf5_dataset = h5py.File(os.path.join(path, 'dataset.h5'), 'r')

        if is_data_graph:
            self.return_item = self.return_graph
        else:
            self.return_item = self.return_dict

    @property
    def avg_nodes_per_sample(self):
        total_nodes = 0
        total_samples = 0
        for fname, num_steps in self.files.items():
            data = self.hdf5_dataset[fname]
            total_nodes += data['mesh_pos'][:].shape[1] * (num_steps - 2)
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
        data = self.hdf5_dataset[fname]

        return self.return_item(data, sid)
    
    def return_dict(self, data, sid):
        return dict(
            cells=torch.LongTensor(data['cells'][sid, ...]),
            node_type=torch.LongTensor(data['node_type'][sid, ...]),
            mesh_pos=torch.Tensor(data['mesh_pos'][sid, ...]),
            world_pos=torch.Tensor(data['world_pos'][sid + 1, ...]),
            prev_world_pos=torch.Tensor(data['world_pos'][sid, ...]),
            target_world_pos=torch.Tensor(data['world_pos'][sid + 2, ...])
        )

    def return_graph(self, data, sid):
        d = dict(
            cells=torch.LongTensor(data['cells'][sid, ...]),
            node_type=torch.LongTensor(data['node_type'][sid, ...]),
            mesh_pos=torch.Tensor(data['mesh_pos'][sid, ...]),
            world_pos=torch.Tensor(data['world_pos'][sid + 1, ...]),
            prev_world_pos=torch.Tensor(data['world_pos'][sid, ...]),
            target_world_pos=torch.Tensor(data['world_pos'][sid + 2, ...])
        )
        graph = build_graph_Cloth(d)

        world_pos = d['world_pos']
        prev_world_pos = d['prev_world_pos']
        target_world_pos = d['target_world_pos']

        cur_position = world_pos
        prev_position = prev_world_pos
        target_position = target_world_pos
        target = target_position - 2 * cur_position + prev_position

        return [graph, target, d['node_type']]


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
        Datasets = Cloth_single_dataset
    elif model== "IncompNS":
        Datasets = IncompNS_single_dataset
    elif model == "HyperEl":
        Datasets = HyperEl_single_dataset
    else:
        raise ValueError("The dataset type doesn't exist.")
    
    ds = Datasets(path, is_data_graph)
    if prefetch == 0:
        return torch.utils.data.DataLoader(ds, batch_size=1, shuffle = shuffle, collate_fn=my_collate_fn)
    return torch.utils.data.DataLoader(ds, batch_size=1, shuffle = shuffle, prefetch_factor=prefetch, num_workers=8, pin_memory=True, collate_fn=my_collate_fn)


def get_trajectory_dataloader(path,
                              model = "Cloth",
                              split = "test",
                              trajectory_index = 0,
                              shuffle = False,
                              prefetch = 0,
                              is_data_graph = False):
    path = os.path.join(path, split)
    if model == "Cloth":
        Datasets = Cloth_trajectory_dataset
    else:
        raise ValueError("The dataset type doesn't exist.")
    
    ds = Datasets(path, is_data_graph, trajectory_index)
    if prefetch == 0:
        return torch.utils.data.DataLoader(ds, batch_size=1, shuffle = shuffle, collate_fn=my_collate_fn)
    return torch.utils.data.DataLoader(ds, batch_size=1, shuffle = shuffle, prefetch_factor=prefetch, num_workers=8, pin_memory=True, collate_fn=my_collate_fn)


def get_dataloader_hdf5(path, 
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
        Datasets = Cloth_single_dataset_hdf5
    else:
        raise ValueError("The dataset type doesn't exist.")
    
    ds = Datasets(path, is_data_graph)
    if prefetch == 0:
        return torch.utils.data.DataLoader(ds, batch_size=1, shuffle = shuffle, collate_fn=my_collate_fn)
    return torch.utils.data.DataLoader(ds, batch_size=1, shuffle = shuffle, prefetch_factor=prefetch, num_workers=8, pin_memory=True, collate_fn=my_collate_fn)


if __name__ == "__main__":
    # ds = deforming_datasets("D:\project_summary\Graduation Project\\tmp\datasets_np\deforming_plate\\train")
    # ds = cloth_datasets("D:\project_summary\Graduation Project\\tmp\datasets_np\\flag_simple\\train")
    # ds = flow_datasets("D:\project_summary\Graduation Project\\tmp\datasets_np\\cylinder_flow\\train")
    prefetch = 0
    is_graph = False
    use_h5 = True
    print(f'prefetch: {prefetch}, is_graph: {is_graph}, is_useh5: {use_h5}')
    if use_h5:
        dl = get_dataloader_hdf5("D:\project_summary\Graduation Project\\tmp\datasets_hdf5\\flag_simple",model="Cloth",split="train",prefetch=prefetch,is_data_graph=is_graph)
    else:
        dl = get_dataloader("D:\project_summary\Graduation Project\\tmp\datasets_np\\flag_simple",model="Cloth",split="train",prefetch=prefetch,is_data_graph=is_graph)
    dl = iter(dl)
    start_time = time.time()
    # for _ in range(100):
    #     next(dl)
    end_time = time.time()
    a = next(dl)[0]
    print(a['mesh_pos'],a['world_pos'])
    print(sum( (a['world_pos'][0]-a['world_pos'][1]) ** 2))
    execution_time = (end_time - start_time)/100
    print(f"运行时间: {execution_time} 秒")
