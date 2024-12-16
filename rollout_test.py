import sys
import os
import pathlib
from pathlib import Path
import pickle
from absl import app
from absl import flags
import torch
from model_utils import HyperEl
from model_utils import deform_eval
from model_utils import common
import logging
import numpy as np
import json
from model_utils.common import NodeType

from torch.nn.parallel import DistributedDataParallel as DDP
import torch.distributed as dist

import matplotlib
matplotlib.use('AGG')
import matplotlib.pyplot as plt

from dataset_utils import datasets

device = torch.device('cuda')

PARAMETERS = {
    'deform': dict(noise=0.003, gamma=1.0, field='world_pos', history=False,
                  size=3, batch=2, model=HyperEl, evaluator=deform_eval, loss_type='deform',
                  stochastic_message_passing_used='False')
}
params = PARAMETERS['deform']
model = HyperEl.Model(4, message_passing_steps=7).to(device)
is_data_graph = False

dl = datasets.get_dataloader("D:\project_summary\Graduation Project\\tmp\datasets_np\deforming_plate",model="HyperEl",is_data_graph=is_data_graph)
dl = iter(dl)
if is_data_graph:
    loss_fn = HyperEl.loss_fn_alter
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
    loss = loss_fn(target,out,node_type,model)
    print(loss)
else:
    loss_fn = HyperEl.loss_fn
    input = next(dl)[0]
    for k in input:
        input[k]=input[k].to(device)

    out = model(input,is_training=True)
    loss = loss_fn(input,out,model)
    print(loss)


