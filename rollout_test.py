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
loss_fn = HyperEl.loss_fn
dl = datasets.get_dataloader("D:\project_summary\Graduation Project\\tmp\datasets_np\deforming_plate",model="HyperEl")
dl = iter(dl)
input = next(dl) 
for k,v in input.items():
    input[k] = input[k].squeeze(0).to(device)
# print(input)
out = model(input,is_training=True)
loss = loss_fn(input,out,model)
print(loss)
