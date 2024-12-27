"""Model for DeformingPlate."""

import torch
from torch import nn as nn
import torch.nn.functional as F

from model_utils import common
from model_utils import normalization
from model_utils import encode_process_decode

import torch_scatter
from dataclasses import replace


class Model(nn.Module):
    """Model for static cloth simulation."""

    def __init__(self, output_size, message_passing_aggregator='sum', message_passing_steps=15, is_use_world_edge=True, device='cuda'):
        super(Model, self).__init__()
        self._output_normalizer = normalization.Normalizer(size=output_size, name='output_normalizer')
        # self._stress_output_normalizer = normalization.Normalizer(size=3, name='stress_output_normalizer')# NOT USED ACTUALLY
        # self._node_normalizer = normalization.Normalizer(size=9, name='node_normalizer')# NOT USED ACTUALLY
        # self._node_dynamic_normalizer = normalization.Normalizer(size=1, name='node_dynamic_normalizer')# NOT USED ACTUALLY
        self._mesh_edge_normalizer = normalization.Normalizer(size=8, name='mesh_edge_normalizer')
        self._world_edge_normalizer = normalization.Normalizer(size=4, name='world_edge_normalizer') 
        self._displacement_base = None

        self.core_model = encode_process_decode
        self.message_passing_steps = message_passing_steps
        self.message_passing_aggregator = message_passing_aggregator

        
        self.learned_model = self.core_model.EncodeProcessDecode(
            output_size=output_size,# 在deforming_plate中是4
            latent_size=128,
            num_layers=2,
            message_passing_steps=self.message_passing_steps,
            message_passing_aggregator=self.message_passing_aggregator,
            is_use_world_edge=is_use_world_edge)

    def _build_graph(self, inputs):
        """Builds input graph."""
        world_pos = inputs['world_pos']

        node_type = inputs['node_type']

        one_hot_node_type = F.one_hot(node_type[:, 0].to(torch.int64), common.NodeType.SIZE).float()

        cells = inputs['cells']
        senders, receivers = common.triangles_to_edges(cells, rectangle=True)


        # find world edge
        # 原论文应选用最小的mesh域的距离
        # 且原论文也没有规定obstacle和其他种类的node只能作为sender或receiver
        radius = 0.03
        world_distance_matrix = torch.cdist(world_pos, world_pos, p=2)
        world_connection_matrix = torch.where(world_distance_matrix < radius, True, False)

        # remove self connection
        world_connection_matrix = world_connection_matrix.fill_diagonal_(False)

        # remove world edge node pairs that already exist in mesh edge collection
        world_connection_matrix[senders, receivers] = torch.tensor(False, dtype=torch.bool, device=senders.device)

        # only obstacle and handle node as sender and normal node as receiver
        '''no_connection_mask = torch.eq(node_type[:, 0], torch.tensor([common.NodeType.OBSTACLE.value], device=device))
        no_connection_mask = torch.logical_or(no_connection_mask, torch.eq(node_type[:, 0], torch.tensor([common.NodeType.HANDLE.value], device=device)))
        no_connection_mask = torch.stack([no_connection_mask] * world_pos.shape[0], dim=1)
        no_connection_mask_t = torch.transpose(no_connection_mask, 0, 1)
        world_connection_matrix = torch.where(no_connection_mask_t, torch.tensor(0., dtype=torch.float32, device=device),
                                              world_connection_matrix)
        world_connection_matrix = torch.where(no_connection_mask, world_connection_matrix, torch.tensor(0., dtype=torch.float32, device=device))'''

        # remove receivers whose node type is obstacle
        '''no_connection_mask = torch.eq(node_type[:, 0], torch.tensor([common.NodeType.OBSTACLE.value], device=device))
        no_connection_mask_t = torch.transpose(torch.stack([no_connection_mask] * world_pos.shape[0], dim=1), 0, 1)
        world_connection_matrix = torch.where(no_connection_mask_t, torch.tensor(False, dtype=torch.bool, device=device), world_connection_matrix)'''
        # remove senders whose node type is handle and normal
        '''connection_mask = torch.eq(node_type[:, 0], torch.tensor([common.NodeType.OBSTACLE.value], device=device))
        connection_mask = torch.stack([no_connection_mask] * world_pos.shape[0], dim=1)
        world_connection_matrix = torch.where(connection_mask, world_connection_matrix, torch.tensor(False, dtype=torch.bool, device=device))'''
        '''no_connection_mask_t = torch.transpose(torch.stack([no_connection_mask] * world_pos.shape[0], dim=1), 0, 1)
        world_connection_matrix = torch.where(no_connection_mask_t,
                                              torch.tensor(0., dtype=torch.float32, device=device),
                                              world_connection_matrix)'''
        '''world_connection_matrix = torch.where(no_connection_mask,
                                              torch.tensor(0., dtype=torch.float32, device=device),
                                              world_connection_matrix)'''
        # remove senders whose type is normal or handle
        '''no_connection_mask = torch.eq(node_type[:, 0], torch.tensor([common.NodeType.NORMAL.value], device=device))
        no_connection_mask = torch.logical_or(no_connection_mask, torch.eq(node_type[:, 0], torch.tensor([common.NodeType.HANDLE.value], device=device)))
        no_connection_mask = torch.stack([no_connection_mask] * world_pos.shape[0], dim=1)
        world_connection_matrix = torch.where(no_connection_mask, torch.tensor(0., dtype=torch.float32, device=device),
                                              world_connection_matrix)'''
        # select the closest sender
        '''world_distance_matrix = torch.where(world_connection_matrix, world_distance_matrix, torch.tensor(float('inf'), device=device))
        min_values, indices = torch.min(world_distance_matrix, 1)
        world_senders = torch.arange(0, world_pos.shape[0], dtype=torch.int32, device=device)
        world_s_r_tuple = torch.stack((world_senders, indices), dim=1)
        world_senders_mask = torch.eq(node_type[:, 0], torch.tensor([common.NodeType.OBSTACLE.value], device=device))
        world_senders_mask_value = torch.logical_not(torch.isinf(min_values))
        world_senders_mask = torch.logical_and(world_senders_mask, world_senders_mask_value)
        world_s_r_tuple = world_s_r_tuple[world_senders_mask]
        world_senders, world_receivers = torch.unbind(world_s_r_tuple, dim=1)'''
        # print(world_senders.shape[0])
        world_senders, world_receivers = torch.nonzero(world_connection_matrix, as_tuple=True)

        relative_world_pos = (torch.index_select(input=world_pos, dim=0, index=world_receivers) -
                              torch.index_select(input=world_pos, dim=0, index=world_senders))

        '''relative_world_velocity = (torch.index_select(input=inputs['target|world_pos'], dim=0, index=world_senders) -
                              torch.index_select(input=inputs['world_pos'], dim=0, index=world_senders))'''


        world_edge_features = torch.cat((
            relative_world_pos,
            torch.norm(relative_world_pos, dim=-1, keepdim=True)), dim=-1)

        '''world_edge_features = torch.cat((
            relative_world_pos,
            torch.norm(relative_world_pos, dim=-1, keepdim=True),
            relative_world_velocity,
            torch.norm(relative_world_velocity, dim=-1, keepdim=True)), dim=-1)'''

        world_edges = self.core_model.EdgeSet(
            name='world_edges',
            features=self._world_edge_normalizer(world_edge_features),
            # features=world_edge_features,
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

        mesh_edges = self.core_model.EdgeSet(
            name='mesh_edges',
            features=self._mesh_edge_normalizer(mesh_edge_features),
            # features=mesh_edge_features,
            receivers=receivers,
            senders=senders)

        '''obstacle_mask = torch.eq(node_type[:, 0], torch.tensor([common.NodeType.OBSTACLE.value], device=device))
        obstacle_mask = torch.stack([obstacle_mask] * 3, dim=1)
        masked_target_world_pos = torch.where(obstacle_mask, target_world_pos, torch.tensor(0., dtype=torch.float32, device=device))
        masked_world_pos = torch.where(obstacle_mask, world_pos, torch.tensor(0., dtype=torch.float32, device=device))
        # kinematic_nodes_features = self._node_normalizer(masked_target_world_pos - masked_world_pos)
        kinematic_nodes_features = masked_target_world_pos - masked_world_pos
        normal_node_features = torch.cat((torch.zeros_like(world_pos), one_hot_node_type), dim=-1)
        kinematic_node_features = torch.cat((kinematic_nodes_features, one_hot_node_type), dim=-1)
        obstacle_mask = torch.eq(node_type[:, 0], torch.tensor([common.NodeType.OBSTACLE.value], device=device))
        obstacle_mask = torch.stack([obstacle_mask] * 12, dim=1)
        node_features = torch.where(obstacle_mask, kinematic_node_features, normal_node_features)'''
        node_features = one_hot_node_type

        
        return (self.core_model.MultiGraph(node_features=node_features,
                                              edge_sets=[mesh_edges, world_edges]))

    def forward(self, inputs, is_training):
        graph = self._build_graph(inputs)
        if is_training:
            return self.learned_model(graph)
        else:
            return self._update(inputs, self.learned_model(graph))
    
    def forward_with_graph(self, graph, is_training):
        # graph features normalization
        new_mesh_edges = replace(graph.edge_sets[0],features = self._mesh_edge_normalizer(graph.edge_sets[0].features))
        new_world_edges = replace(graph.edge_sets[1],features = self._world_edge_normalizer(graph.edge_sets[1].features))
        
        new_graph = replace(graph, edge_sets=[new_mesh_edges, new_world_edges])

        if is_training:
            return self.learned_model(new_graph)
        

    def _update(self, inputs, per_node_network_output):
        """Integrate model outputs."""
        '''output_mask = torch.eq(inputs['node_type'][:, 0], torch.tensor([common.NodeType.NORMAL.value], device=device))
        output_mask = torch.stack([output_mask] * inputs['world_pos'].shape[-1], dim=1)
        velocity = self._output_normalizer.inverse(torch.where(output_mask, per_node_network_output, torch.tensor(0., device=device)))'''
        output = self._output_normalizer.inverse(per_node_network_output)
        velocity = output[...,0:3]
        stress = output[...,3]

        node_type = inputs['node_type']
        '''scripted_node_mask = torch.eq(node_type[:, 0], torch.tensor([common.NodeType.OBSTACLE.value], device=device))
        scripted_node_mask = torch.stack([scripted_node_mask] * 3, dim=1)'''

        # integrate forward
        cur_position = inputs['world_pos']
        position = cur_position + velocity
        # position = torch.where(scripted_node_mask, position + inputs['target|world_pos'] - inputs['world_pos'], position)
        return (position, cur_position, velocity, stress)

    def get_output_normalizer(self):
        return self._output_normalizer

    def save_model(self, path):
        torch.save(self.learned_model, path + "_learned_model.pth")
        torch.save(self._output_normalizer, path + "_output_normalizer.pth")
        # torch.save(self._node_dynamic_normalizer, path + "_node_dynamic_normalizer.pth")
        # torch.save(self._stress_output_normalizer, path + "_stress_output_normalizer.pth")
        torch.save(self._mesh_edge_normalizer, path + "_mesh_edge_normalizer.pth")
        torch.save(self._world_edge_normalizer, path + "_world_edge_normalizer.pth")
        # torch.save(self._node_normalizer, path + "_node_normalizer.pth")

    def load_model(self, path):
        self.learned_model = torch.load(path + "_learned_model.pth", map_location='cpu')
        self._output_normalizer = torch.load(path + "_output_normalizer.pth", map_location='cpu')
        # self._node_dynamic_normalizer = torch.load(path + "_node_dynamic_normalizer.pth")
        # self._stress_output_normalizer = torch.load(path + "_stress_output_normalizer.pth")
        self._mesh_edge_normalizer = torch.load(path + "_mesh_edge_normalizer.pth", map_location='cpu')
        self._world_edge_normalizer = torch.load(path + "_world_edge_normalizer.pth", map_location='cpu')
        # self._node_normalizer = torch.load(path + "_node_normalizer.pth")

    def evaluate(self):
        self.eval()
        self.learned_model.eval()

    def to(self, device):
        super().to(device)
        self._output_normalizer.to(device)
        self._mesh_edge_normalizer.to(device)
        self._world_edge_normalizer.to(device)
        self.learned_model.to(device)


def loss_fn(inputs, network_output, model):
    """L2 loss on position."""
    # build target acceleration
    
    world_pos = inputs['world_pos']
    target_world_pos = inputs['target_world_pos']
    target_stress = inputs['stress']
    

    cur_position = world_pos
    target_position = target_world_pos
    target_velocity = target_position - cur_position

    target = torch.concat((target_velocity, target_stress), dim=1)
    '''scripted_node_mask = torch.eq(node_type[:, 0], torch.tensor([common.NodeType.NORMAL.value], device=device))
    scripted_node_mask = torch.logical_not(scripted_node_mask)
    scripted_node_mask = torch.stack([scripted_node_mask] * 3, dim=1)
    target_velocity = torch.where(scripted_node_mask, torch.tensor(0., device=device), target_velocity)'''

    target_normalizer = model.get_output_normalizer()
    target_normalized = target_normalizer(target)

    '''node_type = inputs['node_type']
    scripted_node_mask = torch.eq(node_type[:, 0], torch.tensor([common.NodeType.OBSTACLE.value], device=device))
    scripted_node_mask = torch.stack([scripted_node_mask] * 3, dim=1)
    target_normalized = torch.where(scripted_node_mask, torch.tensor(0., device=device), target_normalized)'''

    # build loss
    # print(network_output[187])
    node_type = inputs['node_type']
    loss_mask = torch.eq(node_type[:, 0], torch.tensor([common.NodeType.NORMAL.value], device=node_type.device).int())
    # loss_mask = torch.logical_not(loss_mask)
    # loss_mask = torch.eq(node_type[:, 0], torch.tensor([common.NodeType.OBSTACLE.value], device=device).int())
    # loss_mask = torch.eq(node_type[:, 0], torch.tensor([common.NodeType.NORMAL.value], device=device).int())
    # loss_mask = torch.logical_not(loss_mask)
    error = torch.sum((target_normalized - network_output) ** 2, dim=1)
    loss = torch.mean(error[loss_mask])

    # error = torch.sum((target_normalized - network_output) ** 2, dim=1)
    # error += torch.sum((target_normalized_stress - network_output) ** 2, dim=1)
    # loss = torch.mean(error)
    return loss

def loss_fn_alter(target, network_output, node_type, model):
    """L2 loss on position."""
    # build target acceleration
    
    # world_pos = inputs['world_pos']
    # target_world_pos = inputs['target_world_pos']
    # target_stress = inputs['stress']
    

    # cur_position = world_pos
    # target_position = target_world_pos
    # target_velocity = target_position - cur_position

    # target = torch.concat((target_velocity, target_stress), dim=1)
    '''scripted_node_mask = torch.eq(node_type[:, 0], torch.tensor([common.NodeType.NORMAL.value], device=device))
    scripted_node_mask = torch.logical_not(scripted_node_mask)
    scripted_node_mask = torch.stack([scripted_node_mask] * 3, dim=1)
    target_velocity = torch.where(scripted_node_mask, torch.tensor(0., device=device), target_velocity)'''

    target_normalizer = model.get_output_normalizer()
    target_normalized = target_normalizer(target)

    '''node_type = inputs['node_type']
    scripted_node_mask = torch.eq(node_type[:, 0], torch.tensor([common.NodeType.OBSTACLE.value], device=device))
    scripted_node_mask = torch.stack([scripted_node_mask] * 3, dim=1)
    target_normalized = torch.where(scripted_node_mask, torch.tensor(0., device=device), target_normalized)'''

    # build loss
    # print(network_output[187])
    loss_mask = torch.eq(node_type[:, 0], torch.tensor([common.NodeType.NORMAL.value], device=node_type.device).int())
    # loss_mask = torch.logical_not(loss_mask)
    # loss_mask = torch.eq(node_type[:, 0], torch.tensor([common.NodeType.OBSTACLE.value], device=device).int())
    # loss_mask = torch.eq(node_type[:, 0], torch.tensor([common.NodeType.NORMAL.value], device=device).int())
    # loss_mask = torch.logical_not(loss_mask)
    error = torch.sum((target_normalized - network_output) ** 2, dim=1)
    loss = torch.mean(error[loss_mask])

    return loss

def rollout():
    pass

def evaluate():
    pass