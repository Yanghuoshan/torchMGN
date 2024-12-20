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
"""Core learned graph net model."""

import collections
from math import ceil
from collections import OrderedDict
import functools
import torch
from torch import nn as nn
import torch_scatter
from torch_scatter.composite import scatter_softmax
import torch.nn.functional as F
from dataclasses import dataclass,replace
from model_utils.common import EdgeSet,MultiGraph

# EdgeSet = collections.namedtuple('EdgeSet', ['name', 'features', 'senders',
#                                              'receivers'])
# MultiGraph = collections.namedtuple('Graph', ['node_features', 'edge_sets'])
# MultiGraphWithPos = collections.namedtuple('Graph', ['node_features', 'edge_sets', 'target_feature', 'model_type', 'node_dynamic'])

def init_weights(m): ## Init lazylinear layers
    if isinstance(m, nn.LazyLinear):
        nn.init.kaiming_uniform_(m.weight, a=0, mode='fan_in', nonlinearity='relu')
        if m.bias is not None:
            nn.init.constant_(m.bias, 0)

class LazyMLP(nn.Module):
    def __init__(self, output_sizes, is_Sigmoid=False):
        super().__init__()                        
        num_layers = len(output_sizes)
        self._layers_ordered_dict = OrderedDict()
        for index, output_size in enumerate(output_sizes):
            self._layers_ordered_dict["linear_" + str(index)] = nn.LazyLinear(output_size)
            if index < (num_layers - 1):
                # self._layers_ordered_dict["relu_" + str(index)] = nn.ReLU()
                if is_Sigmoid:
                    self._layers_ordered_dict["relu_" + str(index)] = nn.Sigmoid()
                else:
                    self._layers_ordered_dict["relu_" + str(index)] = nn.ReLU()
        self.layers = nn.Sequential(self._layers_ordered_dict)

    def forward(self, input):
        y = self.layers(input)
        return y

'''
class AttentionModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.linear_layer = nn.LazyLinear(1)
        self.leaky_relu = nn.LeakyReLU(negative_slope=0.2)
        self.to(device)

    def forward(self, input, index):

        latent = self.linear_layer(input)
        latent = self.leaky_relu(latent)
        result = torch.zeros(*latent.shape)

        result = scatter_softmax(latent.float(), index, dim=0)
        result = result.type(result.dtype)
        return result
'''

class GraphNetBlock(nn.Module):
    """Multi-Edge Interaction Network with residual connections."""

    def __init__(self, model_fn, output_size, message_passing_aggregator, is_use_world_edge):
        super().__init__()
        self.mesh_edge_model = model_fn(output_size)
        if is_use_world_edge:
            self.world_edge_model = model_fn(output_size)
        self.node_model = model_fn(output_size)
        self.message_passing_aggregator = message_passing_aggregator

        # self.linear_layer = nn.LazyLinear(1)
        # self.leaky_relu = nn.LeakyReLU(negative_slope=0.2)

    def _update_edge_features(self, node_features, edge_set):
        """Aggregrates node features, and applies edge function."""
        senders = edge_set.senders
        receivers = edge_set.receivers
        sender_features = torch.index_select(input=node_features, dim=0, index=senders)
        receiver_features = torch.index_select(input=node_features, dim=0, index=receivers)
        features = [sender_features, receiver_features, edge_set.features]
        features = torch.cat(features, dim=-1)
        if edge_set.name == "mesh_edges":
            return self.mesh_edge_model(features)
        else:
            return self.world_edge_model(features)

    def unsorted_segment_operation(self, data, segment_ids, num_segments, operation):
        """
        Computes the sum along segments of a tensor. Analogous to tf.unsorted_segment_sum.

        :param data: A tensor whose segments are to be summed.
        :param segment_ids: The segment indices tensor.
        :param num_segments: The number of segments.
        :return: A tensor of same data type as the data argument.
        """
        assert all([i in data.shape for i in segment_ids.shape]), "segment_ids.shape should be a prefix of data.shape"

        # segment_ids is a 1-D tensor repeat it to have the same shape as data
        if len(segment_ids.shape) == 1:
            s = torch.prod(torch.tensor(data.shape[1:])).long().to(segment_ids.device)
            segment_ids = segment_ids.repeat_interleave(s).view(segment_ids.shape[0], *data.shape[1:])

        assert data.shape == segment_ids.shape, "data.shape and segment_ids.shape should be equal"

        shape = [num_segments] + list(data.shape[1:])
        result = torch.zeros(*shape)
        if operation == 'sum':
            result = torch_scatter.scatter_add(data.float(), segment_ids, dim=0, dim_size=num_segments)
        elif operation == 'max':
            result, _ = torch_scatter.scatter_max(data.float(), segment_ids, dim=0, dim_size=num_segments)
        elif operation == 'mean':
            result = torch_scatter.scatter_mean(data.float(), segment_ids, dim=0, dim_size=num_segments)
        elif operation == 'min':
            result, _ = torch_scatter.scatter_min(data.float(), segment_ids, dim=0, dim_size=num_segments)
        elif operation == 'std':
            result = torch_scatter.scatter_std(data.float(), segment_ids, out=result, dim=0, dim_size=num_segments)
        else:
            raise Exception('Invalid operation type!')
        result = result.type(data.dtype)
        return result

    def _update_node_features(self, node_features, edge_sets):
        """Aggregrates edge features, and applies node function."""
        num_nodes = node_features.shape[0]
        features = [node_features]
        for edge_set in edge_sets:
            if self.message_passing_aggregator == 'pna':
                features.append(
                    self.unsorted_segment_operation(edge_set.features, edge_set.receivers,
                                                    num_nodes, operation='sum'))
                features.append(
                    self.unsorted_segment_operation(edge_set.features, edge_set.receivers,
                                                    num_nodes, operation='mean'))
                features.append(
                    self.unsorted_segment_operation(edge_set.features, edge_set.receivers,
                                                    num_nodes, operation='max'))
                features.append(
                    self.unsorted_segment_operation(edge_set.features, edge_set.receivers,
                                                    num_nodes, operation='min'))
            else:
                features.append(
                    self.unsorted_segment_operation(edge_set.features, edge_set.receivers, num_nodes,
                                                    operation=self.message_passing_aggregator))
        features = torch.cat(features, dim=-1)
        return self.node_model(features)

    def forward(self, graph, mask=None):
        """Applies GraphNetBlock and returns updated MultiGraph."""
        # apply edge functions
        new_edge_sets = []
        for edge_set in graph.edge_sets:
            updated_features = self._update_edge_features(graph.node_features, edge_set)
            # new_edge_sets.append(edge_set._replace(features=updated_features)) # namedtuple
            new_edge_sets.append(replace(edge_set, features=updated_features))

        # apply node function
        new_node_features = self._update_node_features(graph.node_features, new_edge_sets)

        # add residual connections
        new_node_features = new_node_features + graph.node_features
        if mask is not None:
            mask = mask.repeat(new_node_features.shape[-1])
            mask = mask.view(new_node_features.shape[0], new_node_features.shape[1])
            new_node_features = torch.where(mask, new_node_features, graph.node_features)
        new_edge_sets = [replace(es, features=es.features + old_es.features)
                         for es, old_es in zip(new_edge_sets, graph.edge_sets)]
        # new_edge_sets = [es._replace(features=es.features + old_es.features)
        #                  for es, old_es in zip(new_edge_sets, graph.edge_sets)] # namedtuple
        return MultiGraph(new_node_features, new_edge_sets)


class Encoder(nn.Module):
    """Encodes node and edge features into latent features."""

    def __init__(self, make_mlp, latent_size, is_use_world_edge):
        super().__init__()
        self._make_mlp = make_mlp
        self._latent_size = latent_size
        self.node_model = self._make_mlp(latent_size)
        self.mesh_edge_model = self._make_mlp(latent_size)
        if is_use_world_edge:
            self.world_edge_model = self._make_mlp(latent_size)

    def forward(self, graph):
        node_latents = self.node_model(graph.node_features)
        new_edges_sets = []

        for index, edge_set in enumerate(graph.edge_sets):
            if edge_set.name == "mesh_edges":
                feature = edge_set.features
                latent = self.mesh_edge_model(feature)
                # new_edges_sets.append(edge_set._replace(features=latent)) # namedtuple
                new_edges_sets.append(replace(edge_set, features=latent))
            else:
                feature = edge_set.features
                latent = self.world_edge_model(feature)
                # new_edges_sets.append(edge_set._replace(features=latent))
                new_edges_sets.append(replace(edge_set, features=latent))
        return MultiGraph(node_latents, new_edges_sets)


class Decoder(nn.Module):
    """Decodes node features from graph."""

    """Encodes node and edge features into latent features."""

    def __init__(self, make_mlp, output_size):
        super().__init__()
        self.model = make_mlp(output_size)

    def forward(self, graph):
        return self.model(graph.node_features)

class Processor(nn.Module):
    '''
    This class takes the nodes with the most influential feature (sum of square)
    The the chosen numbers of nodes in each ripple will establish connection(features and distances) with the most influential nodes and this connection will be learned
    Then the result is add to output latent graph of encoder and the modified latent graph will be feed into original processor

    Option: choose whether to normalize the high rank node connection
    '''

    def __init__(self, make_mlp, output_size, message_passing_steps, message_passing_aggregator, is_use_world_edge):
        super().__init__()
        self.graphnet_blocks = nn.ModuleList()
        for index in range(message_passing_steps):
            self.graphnet_blocks.append(GraphNetBlock(model_fn=make_mlp, output_size=output_size,
                                                      message_passing_aggregator=message_passing_aggregator,
                                                      is_use_world_edge=is_use_world_edge
                                                    ))

    def forward(self, latent_graph, normalized_adj_mat=None, mask=None):
        for graphnet_block in self.graphnet_blocks:
            if mask is not None:
                latent_graph = graphnet_block(latent_graph, mask)
            else:
                latent_graph = graphnet_block(latent_graph)
        return latent_graph

class EncodeProcessDecode(nn.Module):
    """Encode-Process-Decode GraphNet model."""

    def __init__(self,
                 output_size,
                 latent_size,
                 num_layers,
                 message_passing_aggregator, 
                 message_passing_steps,
                 is_use_world_edge):
        super().__init__()
        self._latent_size = latent_size
        self._output_size = output_size
        self._num_layers = num_layers
        self._message_passing_steps = message_passing_steps
        self._message_passing_aggregator = message_passing_aggregator
        
        self.encoder = Encoder(make_mlp=self._make_mlp, latent_size=self._latent_size, is_use_world_edge=is_use_world_edge)
        self.processor = Processor(make_mlp=self._make_mlp, output_size=self._latent_size,
                                   message_passing_steps=self._message_passing_steps,
                                   message_passing_aggregator=self._message_passing_aggregator,
                                   is_use_world_edge=is_use_world_edge)
        self.decoder = Decoder(make_mlp=functools.partial(self._make_mlp, layer_norm=False),
                               output_size=self._output_size)

    def _make_mlp(self, output_size, layer_norm=True):
        """Builds an MLP."""
        widths = [self._latent_size] * self._num_layers + [output_size]
        network = LazyMLP(widths)
        if layer_norm:
            network = nn.Sequential(network, nn.LayerNorm(normalized_shape=widths[-1]))
        return network

    def forward(self, graph):
        """Encodes and processes a multigraph, and returns node features."""
        latent_graph = self.encoder(graph)
        latent_graph = self.processor(latent_graph)
        return self.decoder(latent_graph)
