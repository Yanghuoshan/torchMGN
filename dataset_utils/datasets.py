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

from model_utils.common import build_graph_HyperEl, build_graph_Cloth, NodeType

from dataclasses import replace


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

    def __len__(self): return next(iter(self.files.items()))[1]

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


class HyperEl_trajectory_dataset(torch.utils.data.Dataset):
    def __init__(self, path, prebuild_graph_fn = None, trajectory_index = 0):
        self.path = path
        self.meta = json.loads(open(os.path.join(path, 'metadata.json')).read())
        self.files = self.meta['files']
        self.num_samples = sum(self.files[f] - 1 for f in self.files)
        self.fname = list(self.files.keys())[trajectory_index]
        self.prebuild_graph_fn = prebuild_graph_fn

        self.hdf5_dataset = h5py.File(os.path.join(path, 'dataset.h5'), 'r')

        # if prebuild_graph_fn is not None:
        #     self.return_item = self.return_graph
        # else:
        self.return_item = self.return_dict


    def __len__(self): return next(iter(self.files.items()))[1]

    def __getitem__(self, idx : int) -> dict:
        data = self.hdf5_dataset[self.fname]
        return self.return_item(data, idx)
    
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
        new_dict = dict(
            cells=torch.LongTensor(data['cells'][sid, ...]),
            node_type=torch.LongTensor(data['node_type'][sid, ...]),
            mesh_pos=torch.Tensor(data['mesh_pos'][sid, ...]),
            world_pos=torch.Tensor(data['world_pos'][sid, ...]),
            target_world_pos=torch.Tensor(data['world_pos'][sid + 1, ...]),
            stress=torch.Tensor(data['stress'][sid, ...])
        ) 
             
        graph = self.prebuild_graph_fn(new_dict)

        world_pos = new_dict['world_pos']
        target_world_pos = new_dict['target_world_pos']
        # target_stress = new_dict['stress']

        cur_position = world_pos
        target_position = target_world_pos
        target = target_position - cur_position

        # target = torch.concat((target, target_stress), dim=1) 

        return [graph, target, new_dict['node_type']]



class Cloth_single_dataset_hdf5(torch.utils.data.Dataset):
    def __init__(self, path, prebuild_graph_fn = None, add_noise_fn = None):
        self.path = path
        self.meta = json.loads(open(os.path.join(path, 'metadata.json')).read())
        self.files = self.meta['files']
        self.num_samples = sum(self.files[f] - 2 for f in self.files)
        self.add_noise_fn = add_noise_fn
        self.prebuild_graph_fn = prebuild_graph_fn

        self.hdf5_dataset = h5py.File(os.path.join(path, 'dataset.h5'), 'r')

        if prebuild_graph_fn is not None:
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
        new_dict =  dict(
            cells=torch.LongTensor(data['cells'][sid, ...]),
            node_type=torch.LongTensor(data['node_type'][sid, ...]),
            mesh_pos=torch.Tensor(data['mesh_pos'][sid, ...]),
            world_pos=torch.Tensor(data['world_pos'][sid + 1, ...]),
            prev_world_pos=torch.Tensor(data['world_pos'][sid, ...]),
            target_world_pos=torch.Tensor(data['world_pos'][sid + 2, ...])
        )
        
        if self.add_noise_fn is not None:
            new_dict = self.add_noise_fn(new_dict)    
        
        return new_dict
        

    def return_graph(self, data, sid):
        new_dict = dict(
            cells=torch.LongTensor(data['cells'][sid, ...]),
            node_type=torch.LongTensor(data['node_type'][sid, ...]),
            mesh_pos=torch.Tensor(data['mesh_pos'][sid, ...]),
            world_pos=torch.Tensor(data['world_pos'][sid + 1, ...]),
            prev_world_pos=torch.Tensor(data['world_pos'][sid, ...]),
            target_world_pos=torch.Tensor(data['world_pos'][sid + 2, ...])
        )
        if self.add_noise_fn is not None:
            new_dict = self.add_noise_fn(new_dict)   
             
        graph = self.prebuild_graph_fn(new_dict)

        world_pos = new_dict['world_pos']
        prev_world_pos = new_dict['prev_world_pos']
        target_world_pos = new_dict['target_world_pos']

        cur_position = world_pos
        prev_position = prev_world_pos
        target_position = target_world_pos
        target = target_position - 2 * cur_position + prev_position

        return [graph, target, new_dict['node_type']]
    

class HyperEl_single_dataset_hdf5(torch.utils.data.Dataset):
    def __init__(self, path, prebuild_graph_fn = None, add_noise_fn = None):
        self.path = path
        self.meta = json.loads(open(os.path.join(path, 'metadata.json')).read())
        self.files = self.meta['files']
        self.num_samples = sum(self.files[f] - 1 for f in self.files)
        self.add_noise_fn = add_noise_fn
        self.prebuild_graph_fn = prebuild_graph_fn

        # self.alter=alter # 当alter为true时，障碍物取下一时刻作为输入

        self.hdf5_dataset = h5py.File(os.path.join(path, 'dataset.h5'), 'r')

        if prebuild_graph_fn is not None:
            self.return_item = self.return_graph
        else:
            self.return_item = self.return_dict

    @property
    def avg_nodes_per_sample(self):
        total_nodes = 0
        total_samples = 0
        for fname, num_steps in self.files.items():
            data = self.hdf5_dataset[fname]
            total_nodes += data['mesh_pos'][:].shape[1] * (num_steps - 1)# 一阶模型，即预测速度的模型-1，预测加速度的模型-2
            total_samples += (num_steps - 1)

        return total_nodes / total_samples


    def idx_to_file(self, sample_id):
        for fname, num_steps in self.files.items():
            if sample_id < (num_steps - 1): return fname, sample_id
            else: sample_id -= (num_steps - 1)
        raise IndexError()

    def __len__(self): return next(iter(self.files.items()))[1]

    def __getitem__(self, idx : int) -> dict:
        fname, sid = self.idx_to_file(idx)
        data = self.hdf5_dataset[fname]

        return self.return_item(data, sid)
    
    def return_dict(self, data, sid):
        new_dict =  dict(
            cells=torch.LongTensor(data['cells'][sid, ...]),
            node_type=torch.LongTensor(data['node_type'][sid, ...]),
            mesh_pos=torch.Tensor(data['mesh_pos'][sid, ...]),
            world_pos=torch.Tensor(data['world_pos'][sid, ...]),
            target_world_pos=torch.Tensor(data['world_pos'][sid + 1, ...]),
            stress=torch.Tensor(data['stress'][sid, ...])
        )

        # if self.alter:
        #     value = NodeType.OBSTACLE
        #     indices = torch.nonzero(new_dict["node_type"].squeeze() == value).squeeze()
        #     new_dict["world_pos"][indices] = torch.Tensor(data['world_pos'][sid + 10, ...][indices])
        
        if self.add_noise_fn is not None:
            new_dict = self.add_noise_fn(new_dict)    
        
        return new_dict
        

    def return_graph(self, data, sid):
        new_dict = dict(
            cells=torch.LongTensor(data['cells'][sid, ...]),
            node_type=torch.LongTensor(data['node_type'][sid, ...]),
            mesh_pos=torch.Tensor(data['mesh_pos'][sid, ...]),
            world_pos=torch.Tensor(data['world_pos'][sid, ...]),
            target_world_pos=torch.Tensor(data['world_pos'][sid + 1, ...]),
            stress=torch.Tensor(data['stress'][sid, ...])
        )
        # if self.alter:
        #     value = NodeType.OBSTACLE
        #     indices = torch.nonzero(new_dict["node_type"].squeeze() == value).squeeze()
        #     new_dict["world_pos"][indices] = torch.Tensor(data['world_pos'][sid + 10, ...][indices])

        if self.add_noise_fn is not None:
            new_dict = self.add_noise_fn(new_dict)   
             
        graph = self.prebuild_graph_fn(new_dict)

        world_pos = new_dict['world_pos']
        target_world_pos = new_dict['target_world_pos']
        # target_stress = new_dict['stress']

        cur_position = world_pos
        target_position = target_world_pos
        target = target_position - cur_position

        # target = torch.concat((target, target_stress), dim=1) 

        return [graph, target, new_dict['node_type']]


class Easy_HyperEl_single_dataset_hdf5(torch.utils.data.Dataset):
    def __init__(self, path, prebuild_graph_fn = None, add_noise_fn = None):
        self.path = path
        self.meta = json.loads(open(os.path.join(path, 'metadata.json')).read())
        self.files = self.meta['files']
        self.num_samples = sum(self.files[f] - 1 for f in self.files)
        self.add_noise_fn = add_noise_fn
        self.prebuild_graph_fn = prebuild_graph_fn

        # self.alter=alter # 当alter为true时，障碍物取下一时刻作为输入

        self.hdf5_dataset = h5py.File(os.path.join(path, 'dataset.h5'), 'r')

        if prebuild_graph_fn is not None:
            self.return_item = self.return_graph
        else:
            self.return_item = self.return_dict

    @property
    def avg_nodes_per_sample(self):
        total_nodes = 0
        total_samples = 0
        for fname, num_steps in self.files.items():
            data = self.hdf5_dataset[fname]
            total_nodes += data['mesh_pos'][:].shape[1] * (num_steps - 1)# 一阶模型，即预测速度的模型-1，预测加速度的模型-2
            total_samples += (num_steps - 1)

        return total_nodes / total_samples


    def idx_to_file(self, sample_id):
        for fname, num_steps in self.files.items():
            if sample_id < (num_steps - 1): return fname, sample_id
            else: sample_id -= (num_steps - 1)
        raise IndexError()

    def __len__(self): return next(iter(self.files.items()))[1]

    def __getitem__(self, idx : int) -> dict:
        fname, sid = self.idx_to_file(idx)
        data = self.hdf5_dataset[fname]

        return self.return_item(data, sid)
    
    def return_dict(self, data, sid):
        new_dict =  dict(
            cells=torch.LongTensor(data['cells'][sid, ...]),
            node_type=torch.LongTensor(data['node_type'][sid, ...]),
            mesh_pos=torch.Tensor(data['mesh_pos'][sid, ...]),
            world_pos=torch.Tensor(data['world_pos'][sid, ...]),
            target_world_pos=torch.Tensor(data['world_pos'][sid + 1, ...]),
            # stress=torch.Tensor(data['stress'][sid, ...]) Easy datasets only predict positions.
        )

        # if self.alter:
        #     value = NodeType.OBSTACLE
        #     indices = torch.nonzero(new_dict["node_type"].squeeze() == value).squeeze()
        #     new_dict["world_pos"][indices] = torch.Tensor(data['world_pos'][sid + 10, ...][indices])
        
        if self.add_noise_fn is not None:
            new_dict = self.add_noise_fn(new_dict)    
        
        return new_dict
        

    def return_graph(self, data, sid):
        new_dict = dict(
            cells=torch.LongTensor(data['cells'][sid, ...]),
            node_type=torch.LongTensor(data['node_type'][sid, ...]),
            mesh_pos=torch.Tensor(data['mesh_pos'][sid, ...]),
            world_pos=torch.Tensor(data['world_pos'][sid, ...]),
            target_world_pos=torch.Tensor(data['world_pos'][sid + 1, ...]),
            # stress=torch.Tensor(data['stress'][sid, ...])
        )
        # if self.alter:
        #     value = NodeType.OBSTACLE
        #     indices = torch.nonzero(new_dict["node_type"].squeeze() == value).squeeze()
        #     new_dict["world_pos"][indices] = torch.Tensor(data['world_pos'][sid + 10, ...][indices])

        if self.add_noise_fn is not None:
            new_dict = self.add_noise_fn(new_dict)   
             
        graph = self.prebuild_graph_fn(new_dict)

        world_pos = new_dict['world_pos']
        target_world_pos = new_dict['target_world_pos']
        # target_stress = new_dict['stress']

        cur_position = world_pos
        target_position = target_world_pos
        target = target_position - cur_position

        # target = torch.concat((target, target_stress), dim=1) 

        return [graph, target, new_dict['node_type']]


def my_collate_fn(batch): # cumstom collate fn
        # batch [data1, data2...]
        return batch


def graph_collate_fn(batch):
    """
    Collate datas which are graph type
    """
    new_graph = None
    new_target = None
    new_node_type = None
    ptr = [0]
    for data in batch:
        cumulative_node_num = ptr[-1]
        if new_graph is None:
            new_graph = data[0]
            new_target = data[1]
            new_node_type = data[2]
        else:
            new_graph.node_features = torch.concat((new_graph.node_features, data[0].node_features),dim=0)
            for i, es in enumerate(data[0].edge_sets):
                new_graph.edge_sets[i].features = torch.concat((new_graph.edge_sets[i].features, es.features),dim=0)
                new_graph.edge_sets[i].senders = torch.concat((new_graph.edge_sets[i].senders, es.senders + cumulative_node_num),dim=0)
                new_graph.edge_sets[i].receivers = torch.concat((new_graph.edge_sets[i].receivers, es.receivers + cumulative_node_num),dim=0)
            new_target = torch.concat((new_target, data[1]),dim=0)
            new_node_type = torch.concat((new_node_type, data[2]),dim=0)
        ptr.append(cumulative_node_num + data[0].node_features.shape[0])
    
    ptr = torch.tensor(ptr)

    return [[new_graph, new_target, new_node_type, ptr[:]]]
    # data = [graph, target, d['node_type']]
    # return [multi_graph_data]


def dict_collate_fn(batch):
    """
    Collate datas which are dict type
    """
    new_dict = None
    ptr = [0]
    for data in batch:
        cumulative_node_num = ptr[-1]
        if new_dict is None:
            new_dict = dict(**data)
        else:
            for k,v in data.items():
                if k == 'cells':
                    v += cumulative_node_num
                new_dict[k] = torch.concat((new_dict[k],v),dim=0)

        ptr.append(cumulative_node_num + data['mesh_pos'].shape[0])

    ptr = torch.tensor(ptr)
    
    new_dict['ptr'] = ptr[:]
    return [new_dict]

    
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
    elif model == "HyperEl":
        Datasets = HyperEl_trajectory_dataset
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


def get_dataloader_hdf5_batch(path, 
                   model = "Cloth",
                   split = "train",
                   shuffle = True,
                   prefetch = 0,
                   batch_size = 2,
                   add_noise_fn = None,
                   prebuild_graph_fn = None):
    path = os.path.join(path,split)
    if model == "Cloth":
        Datasets = Cloth_single_dataset_hdf5
    elif model == "HyperEl":
        Datasets = HyperEl_single_dataset_hdf5
    elif model == "Easy_HyperEl":
        Datasets == Easy_HyperEl_single_dataset_hdf5
    else:
        raise ValueError("The dataset type doesn't exist.")
    
    if prebuild_graph_fn is None:
        collate_fn = dict_collate_fn
    else:
        collate_fn = graph_collate_fn
    ds = Datasets(path, add_noise_fn=add_noise_fn, prebuild_graph_fn=prebuild_graph_fn)
    if prefetch == 0:
        return torch.utils.data.DataLoader(ds, batch_size=batch_size, shuffle = shuffle, collate_fn=collate_fn)
    return torch.utils.data.DataLoader(ds, batch_size=batch_size, shuffle = shuffle, prefetch_factor=prefetch, num_workers=8, pin_memory=True, collate_fn=collate_fn)


if __name__ == "__main__":
    # ds = deforming_datasets("D:\project_summary\Graduation Project\\tmp\datasets_np\deforming_plate\\train")
    # ds = cloth_datasets("D:\project_summary\Graduation Project\\tmp\datasets_np\\flag_simple\\train")
    # ds = flow_datasets("D:\project_summary\Graduation Project\\tmp\datasets_np\\cylinder_flow\\train")
    prefetch = 4
    is_graph = False
    use_h5 = True
    print(f'prefetch: {prefetch}, is_graph: {is_graph}, is_useh5: {use_h5}')
    if use_h5:
        dl = get_dataloader_hdf5_batch("D:\project_summary\Graduation Project\\tmp\datasets_hdf5\\deforming_plate",model="HyperEl",split="train",prefetch=prefetch,batch_size=1,shuffle=False)
    else:
        dl = get_dataloader("D:\project_summary\Graduation Project\\tmp\datasets_np\\flag_simple",model="Cloth",split="train",prefetch=prefetch)
    # dl = iter(dl)
    # start_time = time.time()
    # # for _ in range(100):
    # #     next(dl)
    # end_time = time.time()
    
    # a = next(dl)[0]
    # print(a['cells'].shape)
    
    # execution_time = (end_time - start_time)/100
    # print(f"运行时间: {execution_time} 秒")

    ds = HyperEl_single_dataset_hdf5("D:\project_summary\Graduation Project\\tmp\datasets_hdf5\\deforming_plate\\train")
    print(next(iter(ds))["node_type"].shape)

    # 三维渲染

    # import matplotlib.pyplot as plt
    # from mpl_toolkits.mplot3d import Axes3D
    # import numpy as np
    # os.environ["KMP_DUPLICATE_LIB_OK"]="TRUE"

    # for _ in range(40):
    #     a = next(dl)[0]
    #     points = a['world_pos']
    #     x = points[:, 0].numpy().astype(float)
    #     y = points[:, 1].numpy().astype(float)
    #     z = points[:, 2].numpy().astype(float)
    #     types = a['node_type'].numpy()
    #     # 定义颜色映射
    #     colors = ['c','m','y','k','r','b','g']
    #     print(types)
    #     point_colors = [colors[t[0]] for t in types]
    
    #     # 创建三维图像
    #     fig = plt.figure()
    #     ax = fig.add_subplot(111, projection='3d')

    #     # 绘制点
    #     ax.scatter(x, y, z, c=point_colors, marker='o')

    #     # 设置标签
    #     ax.set_xlabel('X 轴')
    #     ax.set_ylabel('Y 轴')
    #     ax.set_zlabel('Z 轴')

    #     # 显示图像
    #     plt.show()





