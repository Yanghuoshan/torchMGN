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


def triangles_to_edges(faces, rectangle=False):
    """Computes mesh two ways edges from triangles and rectangles."""
    if not rectangle:
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
    else:
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

def generate_world_edges(world_pos, radius, mesh_senders=None, mesh_receivers=None):
    """
    world_pos:     The world field positions of mesh points
    radius:        Threshold of the max edge length
    mesh_senders:  The mesh field edges' senders
    mesh_receiver: The mesh field edges's receivers
    """
    radius = 0.03
    world_distance_matrix = torch.cdist(world_pos, world_pos, p=2)
    world_connection_matrix = torch.where(world_distance_matrix < radius, True, False)

    # remove self connection
    world_connection_matrix = world_connection_matrix.fill_diagonal_(False)

    # remove world edge node pairs that already exist in mesh edge collection
    if mesh_senders is not None:
        world_connection_matrix[mesh_senders, mesh_receivers] = torch.tensor(False, dtype=torch.bool, device=mesh_senders.device)

    world_senders, world_receivers = torch.nonzero(world_connection_matrix, as_tuple=True)

    return world_senders, world_receivers


def build_graph_HyperEl(inputs, rectangle=True):
    """Builds input graph."""
    world_pos = inputs['world_pos']

    node_type = inputs['node_type']

    one_hot_node_type = F.one_hot(node_type[:, 0].to(torch.int64), NodeType.SIZE).float()

    cells = inputs['cells']
    senders, receivers = triangles_to_edges(cells, rectangle=True)


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


def save_checkpoint(model, optimizer, scheduler, step, losses, run_step_config):
    # save checkpoint to prevent interruption
    model.save_model(os.path.join(run_step_config['checkpoint_dir'], f"model_checkpoint"))
    torch.save(optimizer.state_dict(), os.path.join(run_step_config['checkpoint_dir'], f"optimizer_checkpoint.pth"))
    torch.save(scheduler.state_dict(), os.path.join(run_step_config['checkpoint_dir'], f"scheduler_checkpoint.pth"))
    # save the steps that have been already trained
    torch.save({'trained_step': step}, os.path.join(run_step_config['checkpoint_dir'], "step_checkpoint.pth"))
    # save the previous losses
    torch.save({'losses': losses}, os.path.join(run_step_config['checkpoint_dir'], "losses_checkpoint.pth"))


def learner(model, loss_fn, run_step_config, device, datasets, init_weights):
    root_logger = logging.getLogger()
    root_logger.info(f"Use gpu {run_step_config['gpu_id']}")
    optimizer = torch.optim.Adam(model.parameters(), lr=run_step_config['lr_init'])
    scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, 0.1 + 1e-6, last_epoch=-1)

    losses = []
    loss_save_interval = 1000
    loss_save_cnt = 0
    running_loss = 0.0
    trained_epoch = 0
    trained_step = 0

    if run_step_config['last_run_dir'] is not None:
        optimizer.load_state_dict(torch.load(os.path.join(run_step_config['last_run_step_dir'], 'checkpoint', "optimizer_checkpoint.pth")))
        scheduler.load_state_dict(torch.load(os.path.join(run_step_config['last_run_step_dir'], 'checkpoint', "scheduler_checkpoint.pth")))
        trained_step = torch.load(os.path.join(run_step_config['last_run_step_dir'], 'checkpoint', "step_checkpoint.pth"))['trained_step'] + 1
        losses = torch.load(os.path.join(run_step_config['last_run_step_dir'], 'checkpoint', "losses_checkpoint.pth"))['losses'][:]
        root_logger.info("Loaded optimizer, scheduler and model epoch checkpoint\n")

    # pre run for normalizer
    fixed_pass_count = 500 # equal to pass_count but doesn't change
    pass_count = 500
    if run_step_config['last_run_dir'] is not None:
        pass_count = 0

    # run the left steps or not
    not_reached_max_steps = True
    step = 0
    if run_step_config['last_run_dir'] is not None and not run_step_config['start_new_trained_step']:
        step = trained_step
    
    # dry run for lazy linear layers initialization
    is_dry_run = True
    if run_step_config['last_run_dir'] is not None:
        is_dry_run = False
        root_logger.info("No Dry run")

    while not_reached_max_steps:
        for epoch in range(run_step_config['epochs'])[trained_epoch:]:
            # model will train itself with the whole dataset
            if run_step_config['use_hdf5']:
                if run_step_config['batch_size'] > 1:
                    ds_loader = datasets.get_dataloader_hdf5_batch(run_step_config['dataset_dir'],
                                                        model=run_step_config['model'],
                                                        split='train',
                                                        shuffle=True,
                                                        prefetch=run_step_config['prefetch'], 
                                                        batch_size=run_step_config['batch_size'],
                                                        is_data_graph=run_step_config['is_data_graph'])
                else:
                    ds_loader = datasets.get_dataloader_hdf5(run_step_config['dataset_dir'],
                                                        model=run_step_config['model'],
                                                        split='train',
                                                        shuffle=True,
                                                        prefetch=run_step_config['prefetch'], 
                                                        is_data_graph=run_step_config['is_data_graph'])
            else:
                ds_loader = datasets.get_dataloader(run_step_config['dataset_dir'],
                                                    model=run_step_config['model'],
                                                    split='train',
                                                    shuffle=True,
                                                    prefetch=run_step_config['prefetch'], 
                                                    is_data_graph=run_step_config['is_data_graph'])
            root_logger.info("Epoch " + str(epoch + 1) + "/" + str(run_step_config['epochs']))
            ds_iterator = iter(ds_loader)

            # dry run
            if is_dry_run:
                if run_step_config['is_data_graph']:
                    input = next(ds_iterator)
                    graph =input[0][0].to(device)
                    target = input[0][1].to(device)
                    node_type = input[0][2].to(device)

                    model.forward_with_graph(graph,True)
                    
                else:
                    input = next(ds_iterator)[0]
                    for k in input:
                        input[k]=input[k].to(device)

                    model(input,is_training=True)

                model.apply(init_weights)
                    
                is_dry_run = False
                root_logger.info("Dry run finished")
            
            # start to train
            for input in ds_iterator:
                if run_step_config['is_data_graph']:
                    graph =input[0][0].to(device)
                    target = input[0][1].to(device)
                    node_type = input[0][2].to(device)

                    out = model.forward_with_graph(graph,True)
                    loss = loss_fn(target,out,node_type,model)
                else:
                    input = input[0]
                    for k in input:
                        input[k]=input[k].to(device)

                    out = model(input,is_training=True)
                    loss = loss_fn(input,out,model)

                if pass_count > 0:
                    pass_count -= 1
                else:
                    optimizer.zero_grad()
                    loss.backward()
                    optimizer.step()

                    running_loss += loss.cpu().item()
                    loss_save_cnt += 1
                    if loss_save_cnt % loss_save_interval == 0:
                        avg_loss = running_loss / loss_save_cnt
                        losses.append(avg_loss)
                        running_loss = 0.0
                        loss_save_cnt = 0
                        root_logger.info(f"Step [{step+1}], Loss: {avg_loss:.4f}")

                # Save the model state between steps
                if (step + 1- fixed_pass_count) % run_step_config['nsave_steps'] == 0:
                    save_checkpoint(model, optimizer, scheduler, step, losses, run_step_config)
                
                # Break if step reaches the maximun
                if (step+1) >= run_step_config['max_steps']:
                    not_reached_max_steps = False
                    break
                
                # memory cleaning
                # if step % 100 == 0:
                #     gc.collect()
                #     torch.cuda.empty_cache()

                step += 1

            # Break if step reaches the maximun
            if not_reached_max_steps == False:
                break

            # Save the model state between epochs
            save_checkpoint(model, optimizer, scheduler, step, losses, run_step_config)
            
            if epoch == 13:
                scheduler.step()
                root_logger.info("Call scheduler in epoch " + str(epoch))

    # Save the model state in the end
    save_checkpoint(model, optimizer, scheduler, step, losses, run_step_config)