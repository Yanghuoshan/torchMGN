# Lint as: python3
# pylint: disable=g-bad-file-header
# Copyright 2020 DeepMind Technologies Limited. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#    http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or  implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ============================================================================
"""Commonly used data structures and functions."""

import enum
# import tensorflow.compat.v1 as tf
import torch
import torch.nn.functional as F
from dataclasses import replace,dataclass

import os
import logging


class NodeType(enum.IntEnum):
    NORMAL = 0
    OBSTACLE = 1
    AIRFOIL = 2
    HANDLE = 3
    INFLOW = 4
    OUTFLOW = 5
    WALL_BOUNDARY = 6
    SYMMETRY = 8
    SIZE = 9


@dataclass
class EdgeSet:
    name: str
    features: torch.Tensor
    senders: torch.Tensor
    receivers: torch.Tensor

    def to(self, device):
        self.features = self.features.to(device)
        self.senders = self.senders.to(device)
        self.receivers = self.receivers.to(device)
        return self

@dataclass
class MultiGraph:
    node_features:torch.Tensor
    edge_sets:list

    def to(self, device):
        self.node_features = self.node_features.to(device)
        for es in self.edge_sets:
            es = es.to(device)
        return self


def triangles_to_edges(faces, type=0):
    """Computes mesh two ways edges from triangles 0, rectangles 1 or tetrahedra 2, both triangles and rectangles 3"""
    if type==0:
        # collect edges from triangles
        edges = torch.cat((faces[:, 0:2],
                           faces[:, 1:3],
                           torch.stack((faces[:, 2], faces[:, 0]), dim=1)), dim=0)
        # those edges are sometimes duplicated (within the mesh) and sometimes
        # single (at the mesh boundary).
        # sort & pack edges as single tf.int64
        receivers, _ = torch.min(edges, dim=1)
        senders, _ = torch.max(edges, dim=1)

        edges = torch.stack((senders, receivers), dim=1)
        edges = torch.unique(edges, return_inverse=False, return_counts=False, dim=0)
        senders, receivers = torch.unbind(edges, dim=1)
        # senders = senders.to(torch.int64)
        # receivers = receivers.to(torch.int64)

        return torch.cat((senders, receivers), dim=0), torch.cat((receivers, senders), dim=0)
    elif type==1:
        edges = torch.cat((faces[:, 0:2],
                           faces[:, 1:3],
                           faces[:, 2:4],
                           torch.stack((faces[:, 3], faces[:, 0]), dim=1)), dim=0)
        # those edges are sometimes duplicated (within the mesh) and sometimes
        # single (at the mesh boundary).
        # sort & pack edges as single tf.int64
        receivers, _ = torch.min(edges, dim=1)
        senders, _ = torch.max(edges, dim=1)

        edges = torch.stack((senders, receivers), dim=1)
        edges = torch.unique(edges, return_inverse=False, return_counts=False, dim=0)
        senders, receivers = torch.unbind(edges, dim=1)
        # senders = senders.to(torch.int64)
        # receivers = receivers.to(torch.int64)

        return torch.cat((senders, receivers), dim=0), torch.cat((receivers, senders), dim=0)
    elif type==2:
        edges = torch.cat((faces[:, [0, 1]],
                           faces[:, [0, 2]],
                           faces[:, [0, 3]],
                           faces[:, [1, 2]],
                           faces[:, [1, 3]],
                           faces[:, [2, 3]]), dim=0)
        receivers, _ = torch.min(edges, dim=1)
        senders, _ = torch.max(edges, dim=1)

        edges = torch.stack((senders, receivers), dim=1)
        edges = torch.unique(edges, return_inverse=False, return_counts=False, dim=0)
        senders, receivers = torch.unbind(edges, dim=1)

        return torch.cat((senders, receivers), dim=0), torch.cat((receivers, senders), dim=0)
    else:
        triangles = faces[0]
        rectangles = faces[1]

        # collect edges from triangles
        edges = torch.cat((triangles[:, 0:2],
                           triangles[:, 1:3],
                           torch.stack((triangles[:, 2], triangles[:, 0]), dim=1), 
                           rectangles[:, 0:2],
                           rectangles[:, 1:3],
                           rectangles[:, 2:4],
                           torch.stack((rectangles[:, 3], rectangles[:, 0]), dim=1)),
                           dim=0)

        receivers, _ = torch.min(edges, dim=1)
        senders, _ = torch.max(edges, dim=1)
        edges = torch.stack((senders, receivers), dim=1)
        edges = torch.unique(edges, return_inverse=False, return_counts=False, dim=0)
        senders, receivers = torch.unbind(edges, dim=1)

        return torch.cat((senders, receivers), dim=0), torch.cat((receivers, senders), dim=0)

    
# def rectangles_to_edges(faces):
#     edges = torch.cat((faces[:, 0:2],
#                            faces[:, 1:3],
#                            faces[:, 2:4],
#                            torch.stack((faces[:, 3], faces[:, 0]), dim=1)), dim=0)
#     # those edges are sometimes duplicated (within the mesh) and sometimes
#     # single (at the mesh boundary).
#     # sort & pack edges as single tf.int64
#     receivers, _ = torch.min(edges, dim=1)
#     senders, _ = torch.max(edges, dim=1)
#     packed_edges = torch.stack((senders, receivers), dim=1)
#     unique_edges = torch.unique(packed_edges, return_inverse=False, return_counts=False, dim=0)
#     senders, receivers = torch.unbind(unique_edges, dim=1)
#     senders = senders.to(torch.int64)
#     receivers = receivers.to(torch.int64)
#     two_way_connectivity = (torch.cat((senders, receivers), dim=0), torch.cat((receivers, senders), dim=0))
#     return {'two_way_connectivity': two_way_connectivity, 'senders': senders, 'receivers': receivers}


def build_graph_HyperEl(inputs, rectangle=True):
    """Builds input graph."""
    world_pos = inputs['world_pos']

    node_type = inputs['node_type']

    ptr = inputs['ptr']

    one_hot_node_type = F.one_hot(node_type[:, 0].to(torch.int64), NodeType.SIZE).float()

    cells = inputs['cells']
    senders, receivers = triangles_to_edges(cells, rectangle=True)

    
    # find world edge
    # 原论文应选用最小的mesh域的距离
    # 且原论文也没有规定obstacle和其他种类的node只能作为sender或receiver
    world_distance_matrix = torch.cdist(world_pos, world_pos, p=2)

    radius = 0.03
    pre_i = ptr[0]
    world_connection_matrix = torch.zeros_like(world_distance_matrix, dtype=torch.bool)
    for next_i in ptr[1:]:
        world_connection_segment = torch.zeros_like(world_distance_matrix, dtype=torch.bool)[pre_i:next_i,pre_i:next_i]=True

        world_connection_segment = torch.where((world_distance_matrix < radius) & world_connection_segment, True, False)

        world_connection_matrix = world_connection_matrix | world_connection_segment

        pre_i = next_i

    # remove self connection
    world_connection_matrix = world_connection_matrix.fill_diagonal_(False)

    # remove world edge node pairs that already exist in mesh edge collection
    world_connection_matrix[senders, receivers] = torch.tensor(False, dtype=torch.bool, device=senders.device)


    world_senders, world_receivers = torch.nonzero(world_connection_matrix, as_tuple=True)

    relative_world_pos = (torch.index_select(input=world_pos, dim=0, index=world_receivers) -
                          torch.index_select(input=world_pos, dim=0, index=world_senders))

    world_edge_features = torch.cat((
        relative_world_pos,
        torch.norm(relative_world_pos, dim=-1, keepdim=True)), dim=-1)

    world_edges = EdgeSet(
        name='world_edges',
        features=world_edge_features,
        receivers=world_receivers,
        senders=world_senders)



    mesh_pos = inputs['mesh_pos']
    relative_mesh_pos = (torch.index_select(mesh_pos, 0, senders) -
                         torch.index_select(mesh_pos, 0, receivers))
    all_relative_world_pos = (torch.index_select(input=world_pos, dim=0, index=senders) -
                          torch.index_select(input=world_pos, dim=0, index=receivers))
    mesh_edge_features = torch.cat((
        relative_mesh_pos,
        torch.norm(relative_mesh_pos, dim=-1, keepdim=True),
        all_relative_world_pos,
        torch.norm(all_relative_world_pos, dim=-1, keepdim=True)), dim=-1)

    mesh_edges = EdgeSet(
        name='mesh_edges',
        features=mesh_edge_features,
        receivers=receivers,
        senders=senders)

    node_features = one_hot_node_type

        
    return (MultiGraph(node_features=node_features, edge_sets=[mesh_edges, world_edges]))


def build_graph_Cloth(inputs, rectangle=False):
        """Builds input graph."""
        node_type = inputs['node_type']
        velocity = inputs['world_pos'] - inputs['prev_world_pos']
        one_hot_node_type = F.one_hot(node_type[:, 0].to(torch.int64), NodeType.SIZE)

        node_features = torch.cat((velocity, one_hot_node_type), dim=-1)

        cells = inputs['cells']
        senders, receivers = triangles_to_edges(cells, rectangle=False)

        mesh_pos = inputs['mesh_pos']
        relative_world_pos = (torch.index_select(input=inputs['world_pos'], dim=0, index=senders) -
                              torch.index_select(input=inputs['world_pos'], dim=0, index=receivers))
        relative_mesh_pos = (torch.index_select(mesh_pos, 0, senders) -
                             torch.index_select(mesh_pos, 0, receivers))
        
        edge_features = torch.cat((
            relative_world_pos,
            torch.norm(relative_world_pos, dim=-1, keepdim=True),
            relative_mesh_pos,
            torch.norm(relative_mesh_pos, dim=-1, keepdim=True)), dim=-1)

        mesh_edges = EdgeSet(
            name='mesh_edges',
            features=edge_features,
            receivers=receivers,
            senders=senders)

        return (MultiGraph(node_features=node_features,edge_sets=[mesh_edges]))

def add_noise_fn(field, scale, gamma):
    """
    input = {
        "world_pos":[...],
        "mesh_pos":[...],
        "node_type":[...],
        ...
    }
    """
    
    def add_noise(input):

        zero_size = torch.zeros(input[field].size(), dtype=torch.float32)
        noise = torch.normal(zero_size, std=scale)
        mask = torch.eq(input["node_type"][:,0],torch.tensor([NodeType.NORMAL.value]).int())
        mask = mask.unsqueeze(1)
        mask = mask.repeat(1, noise.size(1) // mask.size(1))
        noise = torch.where(mask, noise, torch.zeros_like(noise))
        input[field] += noise
        input['target_'+field] += (1.0 - gamma) * noise
        return input
    
    def add_noise_mutifields(input):

        for one_field in field:
            zero_size = torch.zeros(input[one_field].size(), dtype=torch.float32)
            noise = torch.normal(zero_size, std=scale)
            mask = torch.eq(input["node_type"][:,0],torch.tensor([NodeType.NORMAL.value]).int())
            mask = mask.unsqueeze(1)
            mask = mask.repeat(1, noise.size(1) // mask.size(1))
            noise = torch.where(mask, noise, torch.zeros_like(noise))
            input[one_field] += noise
            input['target_'+one_field] += (1.0 - gamma) * noise
        return input
    
    if isinstance(field, list):
        return add_noise_mutifields
    else:
        return add_noise

