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
"""Model for Waterballoon."""

from networkx import MultiGraph
from pyparsing import C
from model_utils import common
from model_utils import normalization
from model_utils import encode_process_decode

import torch
import torch.nn as nn
import torch.nn.functional as F
from dataclasses import replace


class Model(nn.Module):
    """Model for fluid simulation."""

    def __init__(self, output_size, message_passing_aggregator='sum', message_passing_steps=15, latent_size=128,
                 is_use_world_edge=False, 
                 mesh_type=3, 
                 use_global_features=False):
        super(Model, self).__init__()
        self.output_size = output_size
        self._output_normalizer = normalization.Normalizer(size=output_size, name='output_normalizer')
        self._mesh_edge_normalizer = normalization.Normalizer(size=output_size*2+2, name='mesh_edge_normalizer')
        # self._world_edge_normalizer = normalization.Normalizer(size=4, name='world_edge_normalizer') # abandon temporarily
        self._node_normalizer = normalization.Normalizer(size=1 + common.NodeType.SIZE, name='node_normalizer')

        self.message_passing_steps = message_passing_steps
        self.message_passing_aggregator = message_passing_aggregator
        
        self.mesh_type = mesh_type

        self.use_global_features = use_global_features
        self.latent_size = latent_size
        
        if self.use_global_features:
            self.learned_model = encode_process_decode.EncodeProcessDecodeAlter(
                output_size=output_size,# 在deforming_plate中是4
                latent_size=latent_size,
                num_layers=2,
                message_passing_steps=self.message_passing_steps,
                message_passing_aggregator=self.message_passing_aggregator,
                is_use_world_edge=is_use_world_edge)
        else:
            self.learned_model = encode_process_decode.EncodeProcessDecode(
                output_size=output_size,# 在deforming_plate中是4
                latent_size=latent_size,
                num_layers=2,
                message_passing_steps=self.message_passing_steps,
                message_passing_aggregator=self.message_passing_aggregator,
                is_use_world_edge=is_use_world_edge)
            
        self.noise_scale = 0.003
        self.noise_gamma = 1
        self.noise_field = "world_pos"

    def graph_normalization(self, graph):
        new_mesh_edges = replace(graph.edge_sets[0],features = self._mesh_edge_normalizer(graph.edge_sets[0].features))
        # new_world_edges = replace(graph.edge_sets[1],features = self._world_edge_normalizer(graph.edge_sets[1].features))
        new_node_features = self._node_normalizer(graph.node_features)

        # graph = replace(graph, node_features=new_node_features, edge_sets=[new_mesh_edges, new_world_edges])
        graph = replace(graph, node_features=new_node_features, edge_sets=[new_mesh_edges])
        return graph

    def build_graph(self, inputs):
        """Builds input graph."""
        node_type = inputs['node_type']
        
        pressure_increase = inputs['target_pressure']-inputs['pressure']
        num_nodes = node_type.shape[0]
        expanded_pressure_increase = pressure_increase.expand(num_nodes, -1)

        node_type = F.one_hot(node_type[:, 0].to(torch.int64), common.NodeType.SIZE)

        node_features = torch.cat((expanded_pressure_increase, node_type), dim=-1)

        global_features = torch.zeros(1,self.latent_size, device=node_features.device)

        cells = [inputs['triangles'],inputs['rectangles']]
        senders, receivers = common.triangles_to_edges(cells, type=3)

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
        if self.use_global_features:
            return (common.MultiGraph(node_features=node_features, global_features=global_features, edge_sets=[mesh_edges]))
        else:
            return (common.MultiGraph(node_features=node_features, edge_sets=[mesh_edges]))
    
    
    def forward(self, inputs, is_trainning, prebuild_graph=False):
        if is_trainning:
            if not prebuild_graph:
                inputs = self.build_graph(inputs)
            graph = self.graph_normalization(inputs)
            return self.learned_model(graph)
        else:
            graph = self.build_graph(inputs)
            graph = self.graph_normalization(graph)
            return self._update(inputs, self.learned_model(graph))

        
    def _update(self, inputs, per_node_network_output):
        """Integrate model outputs."""
        update_tensor = self._output_normalizer.inverse(per_node_network_output)
        # integrate forward
        cur_world_pos = inputs["world_pos"]
        return cur_world_pos + update_tensor

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

    def to(self, device):
        super().to(device)
        self._output_normalizer.to(device)
        self._mesh_edge_normalizer.to(device)
        self._node_normalizer.to(device)
        self.learned_model.to(device)
        return self

    def evaluate(self):
        self.eval()
        self.learned_model.eval()

def loss_fn(inputs, network_output, model):
    world_pos = inputs['world_pos']
    target_world_pos = inputs['target_world_pos']
        
    cur_position = world_pos
    target_position = target_world_pos
    target = target_position - cur_position

    target = target.to(network_output.device)
    target_normalized = model.get_output_normalizer()(target)

    
    # build loss
    node_type = inputs['node_type'].to(network_output.device)
    loss_mask1 = torch.eq(node_type[:, 0], torch.tensor([common.NodeType.NORMAL.value], device=network_output.device).int())
    loss_mask2 = torch.eq(node_type[:, 0], torch.tensor([common.NodeType.WALL_BOUNDARY.value], device=network_output.device).int())
    # combine_loss_mark = loss_mask1 | loss_mask2
    # 新增的条件：node_type 为 symmetry 的点
    loss_mask3 = torch.eq(node_type[:, 0], torch.tensor([common.NodeType.SYMMETRY.value], device=network_output.device).int())
    combine_loss_mark = loss_mask1 | loss_mask2 | loss_mask3
    # 原始 error 计算
    error = torch.sum((target_normalized - network_output) ** 2, dim=1)

    # 对于 node_type 为 symmetry 的点，只计算第二维的 loss
    # special_error = (target_normalized[loss_mask3, 1] - network_output[loss_mask3, 1]) ** 2

    # 将 special_error 应用于 error 中对应的位置
    # error[loss_mask3] = special_error

    # 在loss中加入抑制变形项

    
    # update_tensor = model.get_output_normalizer().inverse(network_output)
    # new_world_pos = world_pos + update_tensor
    # relative_world_pos = (torch.index_select(input=world_pos, dim=0, index=senders) -
    #                       torch.index_select(input=world_pos, dim=0, index=receivers))
    # new_relative_world_pos = (torch.index_select(input=new_world_pos, dim=0, index=senders) -
    #                           torch.index_select(input=new_world_pos, dim=0, index=receivers))
    # edge_length = torch.norm(relative_world_pos, dim=-1, keepdim=True)
    # new_edge_length = torch.norm(new_relative_world_pos, dim=-1, keepdim=True)
    # R = torch.sum(new_edge_length/edge_length)

    loss = torch.mean(error[combine_loss_mark]) 
    return loss


def loss_fn_alter(init_graph:common.MultiGraph, target, network_output, node_type, model):
    target_normalizer = model.get_output_normalizer()
    target_normalized = target_normalizer(target)
    loss_mask1 = torch.eq(node_type[:, 0], torch.tensor([common.NodeType.NORMAL.value], device=network_output.device).int())
    loss_mask2 = torch.eq(node_type[:, 0], torch.tensor([common.NodeType.WALL_BOUNDARY.value], device=network_output.device).int())
    # combine_loss_mark = loss_mask1 | loss_mask2
    # 新增的条件：node_type 为 symmetry 的点
    loss_mask3 = torch.eq(node_type[:, 0], torch.tensor([common.NodeType.SYMMETRY.value], device=network_output.device).int())
    combine_loss_mark = loss_mask1 | loss_mask2 | loss_mask3
    # 原始 error 计算
    error = torch.sum((target_normalized - network_output) ** 2, dim=1)

    # 对于 node_type 为 symmetry 的点，只计算第二维和第三维的 loss
    # special_error = (target_normalized[loss_mask3, 1] - network_output[loss_mask3, 1]) ** 2

    # 将 special_error 应用于 error 中对应的位置
    # error[loss_mask3] = special_error

    # 在loss中加入抑制变形项
    # update_tensor = model.get_output_normalizer().inverse(network_output)
    # relative_world_pos = init_graph.edge_sets[0].features[:,0:2]
    # relative_mesh_pos = init_graph.edge_sets[0].features[:,3:5]
    # senders = init_graph.edge_sets[0].senders
    # receivers = init_graph.edge_sets[0].receivers
    # new_relative_world_pos = relative_world_pos + (torch.index_select(input=update_tensor, dim=0, index=senders) - torch.index_select(input=update_tensor, dim=0, index=receivers))
    
    # mesh_edge_length = torch.norm(relative_mesh_pos, dim=-1, keepdim=True)
    # edge_length = torch.norm(relative_world_pos, dim=-1, keepdim=True)
    # new_edge_length = torch.norm(new_relative_world_pos, dim=-1, keepdim=True)
    
    # R1 = torch.mean(new_edge_length/edge_length)
    # R2 = torch.mean(new_edge_length/mesh_edge_length)
    # alpha = 1
    # beta = 1

    # loss = torch.mean(error[combine_loss_mark]) + alpha*(R1-1)**2 + beta*(R2-1)**2

    # 在loss中加入与目标边长的比较
    senders = init_graph.edge_sets[0].senders
    receivers = init_graph.edge_sets[0].receivers
    relative_world_pos = init_graph.edge_sets[0].features[:,0:2]
    target_relative_world_pos = relative_world_pos + (torch.index_select(input=target, dim=0, index=senders) - torch.index_select(input=target, dim=0, index=receivers))
    target_edge_length = torch.norm(target_relative_world_pos, dim=-1, keepdim=True)

    update_tensor = model.get_output_normalizer().inverse(network_output)
    new_relative_world_pos = relative_world_pos + (torch.index_select(input=update_tensor, dim=0, index=senders) - torch.index_select(input=update_tensor, dim=0, index=receivers))
    new_edge_length = torch.norm(new_relative_world_pos, dim=-1, keepdim=True)

    R3 = torch.mean((new_edge_length - target_edge_length)**2)
    gamma = 1

    loss = torch.mean(error[combine_loss_mark]) + gamma * R3
    return loss


def rollout(model, trajectory, num_steps, device='cuda'):
    cur_state = next(trajectory)[0]
    node_type = cur_state['node_type']
    mask_normal = torch.eq(node_type[:, 0], torch.tensor([common.NodeType.NORMAL.value], device=node_type.device))

    mask_symmetry = torch.eq(node_type[:, 0], torch.tensor([common.NodeType.SYMMETRY.value], device=node_type.device))
    # x_zero_mask = torch.abs(cur_state['world_pos'][:, 0] - 0) < 1e-6
    # mask_symmetry = mask_symmetry|x_zero_mask

    mask_inflow = torch.eq(node_type[:, 0], torch.tensor([common.NodeType.INFLOW.value], device=node_type.device))
    
    mask_obstacle = torch.eq(node_type[:, 0], torch.tensor([common.NodeType.OBSTACLE.value], device=node_type.device))

    mask_wallboundary = torch.eq(node_type[:, 0], torch.tensor([common.NodeType.WALL_BOUNDARY.value], device=node_type.device))

    pred_trajectory = []
    triangles = []
    rectangles = []

    for k in cur_state:
        cur_state[k] = cur_state[k].to(device)

    pred_trajectory.append(cur_state['world_pos'])
    triangles.append(cur_state['triangles'])
    rectangles.append(cur_state['rectangles'])

    for _ in range(num_steps):
        
        with torch.no_grad():
            prediction = model(cur_state,is_trainning=False) # prediction三个维度分别是x, y, w

        # 选取普通点和边界点正常更新world_pos
        next_step_world_pos = prediction[:,0:2]

        # 对称轴点只更新 y 轴坐标（索引 1）
        next_step_world_pos[mask_symmetry, 1] = prediction[mask_symmetry, 1]
        next_step_world_pos[mask_symmetry, 0] = cur_state['world_pos'][mask_symmetry, 0]


        cur_state = next(trajectory)[0]
        for k in cur_state:
            cur_state[k] = cur_state[k].to(device)

        next_step_world_pos[mask_obstacle] = cur_state['world_pos'][mask_obstacle]
        next_step_world_pos[mask_inflow] = cur_state['world_pos'][mask_inflow]

        cur_state['world_pos'] = next_step_world_pos


        pred_trajectory.append(cur_state['world_pos'])
        triangles.append(cur_state['triangles'])
        rectangles.append(cur_state['rectangles'])

    return dict(
        world_pos = torch.stack(pred_trajectory),
        triangles = torch.stack(triangles),
        rectangles = torch.stack(rectangles)
        )

def evaluate(model, trajectory, num_steps=None):
    pass