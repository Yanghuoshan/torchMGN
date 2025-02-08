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

    def __init__(self, output_size, message_passing_aggregator='sum', message_passing_steps=15, is_use_world_edge=True, mesh_type=2):
        super(Model, self).__init__()
        self.output_size = output_size
        self._output_normalizer = normalization.Normalizer(size=output_size, name='output_normalizer')
        # self._stress_output_normalizer = normalization.Normalizer(size=3, name='stress_output_normalizer')# NOT USED ACTUALLY
        # self._node_normalizer = normalization.Normalizer(size=9, name='node_normalizer')# NOT USED ACTUALLY
        # self._node_dynamic_normalizer = normalization.Normalizer(size=1, name='node_dynamic_normalizer')# NOT USED ACTUALLY
        self._mesh_edge_normalizer = normalization.Normalizer(size=output_size*2+2, name='mesh_edge_normalizer')
        self._world_edge_normalizer = normalization.Normalizer(size=output_size+1, name='world_edge_normalizer') 

        self.message_passing_steps = message_passing_steps
        self.message_passing_aggregator = message_passing_aggregator

        self.mesh_type = mesh_type

        
        self.learned_model = encode_process_decode.EncodeProcessDecode(
            output_size=output_size,# 在deforming_plate中是4
            latent_size=128,
            num_layers=2,
            message_passing_steps=self.message_passing_steps,
            message_passing_aggregator=self.message_passing_aggregator,
            is_use_world_edge=is_use_world_edge)
        
        self.noise_scale = 0.003
        self.noise_gamma = 1
        self.noise_field = "world_pos"

    
    def graph_normalization(self, graph):
        new_mesh_edges = replace(graph.edge_sets[0],features = self._mesh_edge_normalizer(graph.edge_sets[0].features))
        new_world_edges = replace(graph.edge_sets[1],features = self._world_edge_normalizer(graph.edge_sets[1].features))
        
        graph = replace(graph, edge_sets=[new_mesh_edges, new_world_edges])
        return graph

    
    def build_graph(self, inputs):
        """Builds input graph."""
        world_pos = inputs['world_pos']

        node_type = inputs['node_type']

        one_hot_node_type = F.one_hot(node_type[:, 0].to(torch.int64), common.NodeType.SIZE).float()

        cells = inputs['cells']

        # ptr = inputs['ptr']

        senders, receivers = common.triangles_to_edges(cells, type=self.mesh_type)


        # find world edge
        # 原论文应选用最小的mesh域的距离
        # 且原论文也没有规定obstacle和其他种类的node只能作为sender或receiver
        world_distance_matrix = torch.cdist(world_pos, world_pos, p=2)

        radius = 0.03
        # pre_i = ptr[0]
        # world_connection_matrix = torch.zeros_like(world_distance_matrix, dtype=torch.bool)

        # world_connection_segment = torch.zeros_like(world_distance_matrix, dtype=torch.bool)

        # for next_i in ptr[1:]:
        #     world_connection_segment[:,:] = False

        #     world_connection_segment = world_connection_segment[pre_i:next_i,pre_i:next_i]=True

        #     world_connection_segment = torch.where((world_distance_matrix < radius) & world_connection_segment, True, False)

        #     world_connection_matrix = world_connection_matrix | world_connection_segment

        #     pre_i = next_i

        world_connection_matrix = world_distance_matrix < radius

        # remove self connection
        world_connection_matrix = world_connection_matrix.fill_diagonal_(False)

        # remove world edge node pairs that already exist in mesh edge collection
        world_connection_matrix[senders, receivers] = torch.tensor(False, dtype=torch.bool, device=senders.device)

        # print(world_senders.shape[0])
        world_senders, world_receivers = torch.nonzero(world_connection_matrix, as_tuple=True)

        relative_world_pos = (torch.index_select(input=world_pos, dim=0, index=world_receivers) -
                              torch.index_select(input=world_pos, dim=0, index=world_senders))

        world_edge_features = torch.cat((
            relative_world_pos,
            torch.norm(relative_world_pos, dim=-1, keepdim=True)), dim=-1)

        '''world_edge_features = torch.cat((
            relative_world_pos,
            torch.norm(relative_world_pos, dim=-1, keepdim=True),
            relative_world_velocity,
            torch.norm(relative_world_velocity, dim=-1, keepdim=True)), dim=-1)'''

        world_edges = common.EdgeSet(
            name='world_edges',
            features=world_edge_features,
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

        mesh_edges = common.EdgeSet(
            name='mesh_edges',
            features=mesh_edge_features,
            # features=mesh_edge_features,
            receivers=receivers,
            senders=senders)

        node_features = one_hot_node_type
        
        return (common.MultiGraph(node_features=node_features,
                                              edge_sets=[mesh_edges, world_edges]))

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
    
    def forward_with_graph(self, graph, is_trainning):
        # graph features normalization
        # new_mesh_edges = replace(graph.edge_sets[0],features = self._mesh_edge_normalizer(graph.edge_sets[0].features))
        # new_world_edges = replace(graph.edge_sets[1],features = self._world_edge_normalizer(graph.edge_sets[1].features))
        
        # new_graph = replace(graph, edge_sets=[new_mesh_edges, new_world_edges])
        graph = self.graph_normalization(graph)

        if is_trainning:
            return self.learned_model(graph)
        

    def _update(self, inputs, per_node_network_output):
        """Integrate model outputs."""
        '''output_mask = torch.eq(inputs['node_type'][:, 0], torch.tensor([common.NodeType.NORMAL.value], device=device))
        output_mask = torch.stack([output_mask] * inputs['world_pos'].shape[-1], dim=1)
        velocity = self._output_normalizer.inverse(torch.where(output_mask, per_node_network_output, torch.tensor(0., device=device)))'''
        output = self._output_normalizer.inverse(per_node_network_output)
        velocity = output[...,0:self.output_size]
        # stress = output[...,3]
        '''scripted_node_mask = torch.eq(node_type[:, 0], torch.tensor([common.NodeType.OBSTACLE.value], device=device))
        scripted_node_mask = torch.stack([scripted_node_mask] * 3, dim=1)'''

        # integrate forward
        cur_position = inputs['world_pos']
        position = cur_position + velocity
        # position = torch.where(scripted_node_mask, position + inputs['target|world_pos'] - inputs['world_pos'], position)
        return position

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
    """L1 loss on position."""
    # build target acceleration
    
    world_pos = inputs['world_pos']
    target_world_pos = inputs['target_world_pos']
    # target_stress = inputs['stress']
    

    cur_position = world_pos
    target_position = target_world_pos
    target_velocity = target_position - cur_position

    # target = torch.concat((target_velocity, target_stress), dim=1)
    target = target_velocity
    '''scripted_node_mask = torch.eq(node_type[:, 0], torch.tensor([common.NodeType.NORMAL.value], device=device))
    scripted_node_mask = torch.logical_not(scripted_node_mask)
    scripted_node_mask = torch.stack([scripted_node_mask] * 3, dim=1)
    target_velocity = torch.where(scripted_node_mask, torch.tensor(0., device=device), target_velocity)'''
    target = target.to(network_output.device)
    target_normalized = model.get_output_normalizer()(target)

    '''node_type = inputs['node_type']
    scripted_node_mask = torch.eq(node_type[:, 0], torch.tensor([common.NodeType.OBSTACLE.value], device=device))
    scripted_node_mask = torch.stack([scripted_node_mask] * 3, dim=1)
    target_normalized = torch.where(scripted_node_mask, torch.tensor(0., device=device), target_normalized)'''

    # build loss
    # print(network_output[187])
    node_type = inputs['node_type'].to(network_output.device)
    loss_mask = torch.eq(node_type[:, 0], torch.tensor([common.NodeType.NORMAL.value], device=network_output.device).int())
    symmetry_mask = torch.eq(node_type[:, 0], torch.tensor([common.NodeType.SYMMETRY.value], device=network_output.device).int())
    
    network_output[symmetry_mask,0] = target_normalized[symmetry_mask,0] # Points on the symmerty only use pos_y
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
    symmetry_mask = torch.eq(node_type[:, 0], torch.tensor([common.NodeType.SYMMETRY.value], device=network_output.device).int())
    
    network_output[symmetry_mask,0] = target_normalized[symmetry_mask,0] # Points on the symmerty only use pos_y
    error = torch.sum((target_normalized - network_output) ** 2, dim=1)
    loss = torch.mean(error[loss_mask])

    return loss

def rollout(model, trajectory, num_steps, device='cuda'):
    cur_state = next(trajectory)[0]
    node_type = cur_state['node_type']
    mask_normal = torch.eq(node_type[:, 0], torch.tensor([common.NodeType.NORMAL.value], device=node_type.device))
    mask_normal = torch.stack((mask_normal, mask_normal), dim=1).to(device)

    mask_symmetry = torch.eq(node_type[:, 0], torch.tensor([common.NodeType.SYMMETRY.value], device=node_type.device))

    mask_initiative = torch.eq(node_type[:, 0], torch.tensor([common.NodeType.INITIATIVE.value], device=node_type.device))
    mask_initiative = torch.stack((mask_initiative, mask_initiative), dim=1).to(device)
    # mask_obstacle = torch.eq(node_type[:, 0], torch.tensor([common.NodeType.OBSTACLE.value], device=node_type.device))
    # mask_obstacle = torch.stack((mask_obstacle, mask_obstacle, mask_obstacle), dim=1).to(device)

    new_trajectory = []
    cells = []

    for k in cur_state:
        cur_state[k] = cur_state[k].to(device)

    new_trajectory.append(cur_state['world_pos'])
    cells.append(cur_state['cells'])

    for _ in range(num_steps):
        
        with torch.no_grad():
            prediction = model(cur_state,is_trainning=False)

        next_pos = torch.where(mask_normal|mask_initiative, prediction, cur_state['world_pos']) # select normal and initiative points
        next_pos[mask_symmetry]=prediction[mask_symmetry]


        cur_state['world_pos'] = next_pos

        new_trajectory.append(cur_state['world_pos'])
        cells.append(cur_state['cells'])

    return dict(
        world_pos = torch.stack(new_trajectory),
        cells = torch.stack(cells)
        )


def evaluate():
    pass