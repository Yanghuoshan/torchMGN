import sys
import os
import pathlib
from pathlib import Path

import pickle
from absl import app
from absl import flags

import torch

from dataset_utils import datasets 
from model_utils import HyperEl_model
from model_utils.common import NodeType

import time
import datetime

import csv

from torch.nn.parallel import DistributedDataParallel as DDP
import torch.distributed as dist

import matplotlib
matplotlib.use('AGG')
import matplotlib.pyplot as plt

device = torch.device('cuda')

FLAGS = flags.FLAGS

# common run configuration
flags.DEFINE_enum('model', 'HyperEl_model', ['HyperEl_model'],
                  'Select model to run.')
flags.DEFINE_enum('mode', 'all', ['train', 'eval', 'all'],
                  'Train model, or run evaluation, or run both.')
flags.DEFINE_enum('rollout_split', 'valid', ['train', 'test', 'valid'],
                  'Dataset split to use for rollouts.')
flags.DEFINE_string('dataset', 'deforming_plate', ['deforming_plate'])
flags.DEFINE_integer('epochs', 2, 'No. of training epochs')
flags.DEFINE_integer('trajectories', 2, 'No. of training trajectories')
flags.DEFINE_integer('num_rollouts', 1, 'No. of rollout trajectories')

# core model configuration
flags.DEFINE_enum('core_model', 'encode_process_decode',
                  ['encode_process_decode'],
                  'Core model to be used')
flags.DEFINE_enum('message_passing_aggregator', 'sum', ['sum', 'max', 'min', 'mean', 'pna'], 'No. of training epochs')
flags.DEFINE_integer('message_passing_steps', 5, 'No. of training epochs')
flags.DEFINE_string('model_last_run_dir',
                    None,
                    # os.path.join('E:\\meshgraphnets\\output\\deforming_plate', 'Sat-Feb-12-12-14-04-2022'),
                    # os.path.join('/home/i53/student/ruoheng_ma/meshgraphnets/output/deforming_plate', 'Mon-Jan--3-15-18-53-2022'),
                    'Path to the checkpoint file of a network that should continue training')

# decide whether to use the configuration from last run step
flags.DEFINE_boolean('use_prev_config', True, 'Decide whether to use the configuration from last run step')

# hpc max run time setting
flags.DEFINE_integer('hpc_default_max_time', (48 - 4) * 60 * 60, 'Max run time on hpc')
# flags.DEFINE_integer('hpc_default_max_time', 1500, 'Max run time on hpc')
