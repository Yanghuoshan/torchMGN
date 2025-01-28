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
"""Model for FlagSimple."""

import torch
from torch import nn as nn
import torch.nn.functional as F
# from torch_cluster import random_walk
import functools

from model_utils import common
from model_utils import normalization
from model_utils import encode_process_decode

from dataclasses import replace




class Model(nn.Module):
    """Model for static cloth simulation."""

    def __init__(self, output_size, message_passing_aggregator='sum', message_passing_steps=15, is_use_world_edge=False, mesh_type=0):
        super(Model, self).__init__()
        self._output_normalizer = normalization.Normalizer(size=3, name='output_normalizer')
        self._node_normalizer = normalization.Normalizer(size=3 + common.NodeType.SIZE, name='node_normalizer')
        # self._node_dynamic_normalizer = normalization.Normalizer(size=1, name='node_dynamic_normalizer')
        self._mesh_edge_normalizer = normalization.Normalizer(size=7, name='mesh_edge_normalizer')  # 2D coord + 3D coord + 2*length = 7
        # self._world_edge_normalizer = normalization.Normalizer(size=4, name='world_edge_normalizer')

        self.message_passing_steps = message_passing_steps
        self.message_passing_aggregator = message_passing_aggregator

        self.mesh_type = mesh_type
        
        self.learned_model = encode_process_decode.EncodeProcessDecode(
            output_size=output_size,
            latent_size=128,
            num_layers=2,
            message_passing_steps=self.message_passing_steps,
            message_passing_aggregator=self.message_passing_aggregator,
            is_use_world_edge = is_use_world_edge)
        
        self.noise_scale = 0.003
        self.noise_gamma = 0.1
        self.noise_field = "world_pos"

        

    def graph_normalization(self, graph):
        new_node_features = self._node_normalizer(graph.node_features)
        new_mesh_edges = replace(graph.edge_sets[0],features = self._mesh_edge_normalizer(graph.edge_sets[0].features))
        
        graph = replace(graph, node_features=new_node_features, edge_sets=[new_mesh_edges])
        return graph
    

    def build_graph(self, inputs):
        """Builds input graph."""
        # node_type = inputs['node_type']
        # velocity = inputs['world_pos'] - inputs['prev_world_pos']
        # one_hot_node_type = F.one_hot(node_type[:, 0].to(torch.int64), common.NodeType.SIZE)
        # node_features = torch.cat((velocity, one_hot_node_type), dim=-1)
        node_features = torch.cat((inputs['world_pos'] - inputs['prev_world_pos'], F.one_hot(inputs['node_type'][:, 0].to(torch.int64), common.NodeType.SIZE)), dim=-1)

        cells = inputs['cells']
        senders, receivers = common.triangles_to_edges(cells, type=self.mesh_type)

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

        mesh_edges = common.EdgeSet(
            name='mesh_edges',
            features=edge_features,
            receivers=receivers,
            senders=senders)

        return (common.MultiGraph(node_features=node_features, edge_sets=[mesh_edges]))

    def forward(self, inputs, is_training, prebuild_graph=False):
        if is_training:
            if not prebuild_graph:
                inputs = self.build_graph(inputs)
            graph = self.graph_normalization(inputs)
            return self.learned_model(graph)
        else:
            graph = self.build_graph(inputs)
            graph = self.graph_normalization(graph)
            return self._update(inputs, self.learned_model(graph))
        
    def forward_with_graph(self, graph, is_training):
        # graph features normalization
        # new_node_features = self._node_normalizer(graph.node_features)
        # new_mesh_edges = replace(graph.edge_sets[0],features = self._mesh_edge_normalizer(graph.edge_sets[0].features))
        
        # graph = replace(graph, node_features=new_node_features, edge_sets=[new_mesh_edges])

        graph = self.graph_normalization(graph)

        if is_training:
            return self.learned_model(graph)

    def _update(self, inputs, per_node_network_output):
        """
        Integrate model outputs.
        Ouput next position 
        """

        acceleration = self._output_normalizer.inverse(per_node_network_output)

        # integrate forward
        cur_position = inputs['world_pos']
        prev_position = inputs['prev_world_pos']
        position = 2 * cur_position + acceleration - prev_position
        return position

    def get_output_normalizer(self):
        return self._output_normalizer

    def save_model(self, path):
        torch.save(self.learned_model, path + "_learned_model.pth")
        torch.save(self._output_normalizer, path + "_output_normalizer.pth")
        torch.save(self._mesh_edge_normalizer, path + "_mesh_edge_normalizer.pth")
        # torch.save(self._world_edge_normalizer, path + "_world_edge_normalizer.pth")
        torch.save(self._node_normalizer, path + "_node_normalizer.pth")
        # torch.save(self._node_dynamic_normalizer, path + "_node_dynamic_normalizer.pth")

    def load_model(self, path):
        self.learned_model = torch.load(path + "_learned_model.pth", map_location='cpu')
        self._output_normalizer = torch.load(path + "_output_normalizer.pth", map_location='cpu')
        self._mesh_edge_normalizer = torch.load(path + "_mesh_edge_normalizer.pth", map_location='cpu')
        # self._world_edge_normalizer = torch.load(path + "_world_edge_normalizer.pth")
        self._node_normalizer = torch.load(path + "_node_normalizer.pth", map_location='cpu')
        # self._node_dynamic_normalizer = torch.load(path + "_node_dynamic_normalizer.pth")

    def evaluate(self):
        self.eval()
        self.learned_model.eval()

    def to(self, device):
        super().to(device)
        self._output_normalizer.to(device)
        self._mesh_edge_normalizer.to(device)
        self._node_normalizer.to(device)
        self.learned_model.to(device)
        return self


def loss_fn(inputs, network_output, model):
    world_pos = inputs['world_pos']
    prev_world_pos = inputs['prev_world_pos']
    target_world_pos = inputs['target_world_pos']

    cur_position = world_pos
    prev_position = prev_world_pos
    target_position = target_world_pos
    target_acceleration = target_position - 2 * cur_position + prev_position
    target_acceleration = target_acceleration.to(network_output.device)
    target_normalized = model.get_output_normalizer()(target_acceleration)

    # build loss
    node_type = inputs['node_type'].to(network_output.device)
    loss_mask = torch.eq(node_type[:, 0], torch.tensor([common.NodeType.NORMAL.value], device=network_output.device).int())
    error = torch.sum((target_normalized - network_output) ** 2, dim=1)
    loss = torch.mean(error[loss_mask])  
    return loss


def loss_fn_alter(target, network_output, node_type, model):
    target_normalizer = model.get_output_normalizer()
    target_normalized = target_normalizer(target)
    loss_mask = torch.eq(node_type[:, 0], torch.tensor([common.NodeType.NORMAL.value], device=node_type.device).int())
    error = torch.sum((target_normalized - network_output) ** 2, dim=1)
    loss = torch.mean(error[loss_mask])
    return loss


def rollout(model, initial_state, num_steps):
    """
    Rolls out a trajectory.
    initial state: a dict of initial state
    """
    node_type = initial_state['node_type']
    mask = torch.eq(node_type[:, 0], torch.tensor([common.NodeType.NORMAL.value], device=node_type.device))
    mask = torch.stack((mask, mask, mask), dim=1)

    def step_fn(prev_pos, cur_pos, trajectory, cells):

        with torch.no_grad():
            prediction = model({**initial_state, # cells, node_type, mesh_pos
                                'prev_world_pos': prev_pos,
                                'world_pos': cur_pos}, is_training=False)

        next_pos = torch.where(mask, prediction, cur_pos)

        trajectory.append(cur_pos)
        cells.append(initial_state['cells'])
        return cur_pos, next_pos, trajectory, cells

    prev_pos = initial_state['prev_world_pos']
    cur_pos = initial_state['world_pos']
    trajectory = []
    cells = []
    for step in range(num_steps):
        prev_pos, cur_pos, trajectory, cells = step_fn(prev_pos, cur_pos, trajectory, cells)

    return dict(
        world_pos = torch.stack(trajectory),
        cells = torch.stack(cells)
        )


def evaluate(model, trajectory, num_steps=None):
    """
    Performs model rollouts and create stats.
    trajectory: a iterable dataloader
    num_steps: num of rollout steps 
    """
    # initial_state = next(trajectory)[0]
    # if num_steps is None:
    #     num_steps = initial_state['cells'].shape[0]
    # prediction = rollout(model, initial_state, num_steps)

    # # error = tf.reduce_mean((prediction - trajectory['world_pos'])**2, axis=-1)
    # # scalars = {'mse_%d_steps' % horizon: tf.reduce_mean(error[1:horizon+1])
    # #            for horizon in [1, 10, 20, 50, 100, 200]}

    # scalars = None
    # traj_ops = {
    #     'faces': trajectory['cells'],
    #     'mesh_pos': trajectory['mesh_pos'],
    #     'gt_pos': trajectory['world_pos'],
    #     'pred_pos': prediction
    # }
    # return scalars, traj_ops
    pass


