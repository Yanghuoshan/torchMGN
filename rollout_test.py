import sys
import os
import pathlib
from pathlib import Path
import pickle
from absl import app
from absl import flags
import torch
from model_utils import HyperEl,Cloth
from model_utils import deform_eval
from model_utils import common
from model_utils.encode_process_decode import init_weights
from render_utils import Cloth_render
import logging
import numpy as np
import json
from model_utils.common import NodeType

from torch.nn.parallel import DistributedDataParallel as DDP
import torch.distributed as dist

import matplotlib
import matplotlib.pyplot as plt

from dataset_utils import datasets

device = torch.device('cuda')

os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'

# PARAMETERS = {
#     'deform': dict(noise=0.003, gamma=1.0, field='world_pos', history=False,
#                   size=3, batch=2, model=HyperEl, evaluator=deform_eval, loss_type='deform',
#                   stochastic_message_passing_used='False')
# }
# params = PARAMETERS['deform']
# model = HyperEl.Model(4, message_passing_steps=7).to(device)
last_run_step_dir = ''

M = Cloth
model = M.Model(3, message_passing_steps=7).to(device)
# model.load_model(os.path.join(last_run_step_dir, 'checkpoint', "model_checkpoint"))

rollout = M.rollout
render = Cloth_render.render

is_data_graph = False

# dl = datasets.get_dataloader("D:\project_summary\Graduation Project\\tmp\datasets_np\\flag_simple",model="Cloth",is_data_graph=is_data_graph)
# dl = iter(dl)
if is_data_graph:
    loss_fn = M.loss_fn_alter
    input = next(dl)
    graph =input[0][0]
    target = input[0][1]
    node_type = input[0][2]

    graph = graph.to(device)
    target = target.to(device)
    node_type = node_type.to(device)

# print(input)
# out = model(input,is_training=True)
    out = model.forward_with_graph(graph,True)
    model.apply(init_weights)
    input = next(dl)
    graph =input[0][0]
    target = input[0][1]
    node_type = input[0][2]

    graph = graph.to(device)
    target = target.to(device)
    node_type = node_type.to(device)

# print(input)
# out = model(input,is_training=True)
    out = model.forward_with_graph(graph,True)


    print(out)
else:
    dl = datasets.get_trajectory_dataloader("D:\project_summary\Graduation Project\\tmp\datasets_np\\flag_simple",
                                            model="Cloth",
                                            is_data_graph=is_data_graph, 
                                            trajectory_index=0)
    trajectory = iter(dl)
    origin_trajectory = None
    world_pos = []
    cells = []
    for _ in range(2):
        init_state = next(trajectory)[0]
        world_pos.append(init_state['world_pos'])
        cells.append(init_state['cells'])

    
    # for k in init_state:
    #     init_state[k] = init_state[k].to(device)
    trajectory = dict(
        world_pos = torch.stack(world_pos),
        cells = torch.stack(cells)
        )
    # new_trajectory =rollout(model,init_state,20)
    render(trajectory)

