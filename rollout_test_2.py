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
from run_utils.utils import *
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
last_run_dir = "D:\project_summary\Graduation Project\\torchMGN\output\Cloth\Thu-Dec-26-21-11-24-2024"
last_run_step_dir = find_nth_latest_run_step(last_run_dir, 1)
run_step_config = pickle_load(os.path.join(last_run_step_dir, 'log', 'config.pkl'))
run_step_config['last_run_dir'] = last_run_dir
run_step_config['last_run_step_dir'] = last_run_step_dir

M = Cloth
model = eval(run_step_config['model']).Model(run_step_config['output_size'], 
                                                 run_step_config['message_passing_aggregator'],
                                                 run_step_config['message_passing_steps'],
                                                )
last_run_step_dir = find_nth_latest_run_step(last_run_dir, 1)
model.load_model(os.path.join(last_run_step_dir, 'checkpoint', "model_checkpoint"))
print("Load model:",os.path.join(last_run_step_dir, 'checkpoint', "model_checkpoint"))
# model.load_model(os.path.join(last_run_step_dir, 'checkpoint', "model_checkpoint"))

model.to(device)
rollout = M.rollout
render = Cloth_render.render

is_data_graph = False


dl = datasets.get_trajectory_dataloader("D:\project_summary\Graduation Project\\tmp\datasets_np\\flag_simple",
                                        model="Cloth",
                                        is_data_graph=is_data_graph, 
                                        trajectory_index=0)
trajectory = iter(dl)
init_state = next(trajectory)[0]
for k in init_state:
    init_state[k] = init_state[k].to(device)
new_trajectory =rollout(model,init_state,200)
render(new_trajectory)

