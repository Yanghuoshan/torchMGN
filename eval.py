import sys
import os
import pathlib
from pathlib import Path
import logging

import pickle
from absl import app
from absl import flags

import torch
import gc

from dataset_utils import datasets
from model_utils import HyperEl,Cloth
from model_utils.common import NodeType
from model_utils.encode_process_decode import init_weights
from run_utils.utils import *

import time
import datetime

from torch.nn.parallel import DistributedDataParallel as DDP
import torch.distributed as dist

import matplotlib
import matplotlib.pyplot as plt
from tqdm import tqdm, trange


def evaluator(model, run_step_config):
    root_logger = logging.getLogger()
    model = run_step_config['model']
    """Run a model rollout trajectory."""
    ds_loader = datasets.get_trajectory_dataloader(run_step_config['dataset_dir'],
                                                   model=run_step_config['model'],
                                                   split='roll',
                                                   shuffle=False)
    ds_iterator = iter(ds_loader)
    trajectories = []
    mse_losses = []
    l1_losses = []

    for index in range(run_step_config['num_rollouts']):
        root_logger.info("Evaluating trajectory " + str(index + 1))
        ds_loader = datasets.get_trajectory_dataloader(run_step_config['dataset_dir'],
                                                   model=run_step_config['model'],
                                                   split='roll',
                                                   shuffle=False)
        trajectory = iter(ds_loader)
        
        _, prediction_trajectory = params['evaluator'].evaluate(model, trajectory)
        mse_loss_fn = torch.nn.MSELoss()
        l1_loss_fn = torch.nn.L1Loss()
        if model == 'cloth':
            mse_loss = mse_loss_fn(torch.squeeze(trajectory['world_pos'], dim=0), prediction_trajectory['pred_pos'])
            l1_loss = l1_loss_fn(torch.squeeze(trajectory['world_pos'], dim=0), prediction_trajectory['pred_pos'])
        elif model == 'deform':
            mse_loss = mse_loss_fn(torch.squeeze(trajectory['world_pos'], dim=0), prediction_trajectory['pred_pos'])
            l1_loss = l1_loss_fn(torch.squeeze(trajectory['world_pos'], dim=0), prediction_trajectory['pred_pos'])
        mse_losses.append(mse_loss.cpu())
        l1_losses.append(l1_loss.cpu())
        root_logger.info("    trajectory evaluation mse loss")
        root_logger.info("    " + str(mse_loss))
        root_logger.info("    trajectory evaluation l1 loss")
        root_logger.info("    " + str(l1_loss))
        trajectories.append(prediction_trajectory)
        # scalars.append(scalar_data)
    root_logger.info("mean mse loss of " + str(run_step_config['num_rollouts']) + " rollout trajectories")
    root_logger.info(torch.mean(torch.stack(mse_losses)))
    root_logger.info("mean l1 loss " + str(run_step_config['num_rollouts']) + " rollout trajectories")
    root_logger.info(torch.mean(torch.stack(l1_losses)))
    pickle_save(os.path.join(run_step_config['rollout_dir'], "rollout.pkl"), trajectories)
    loss_record = {}
    loss_record['eval_total_mse_loss'] = torch.sum(torch.stack(mse_losses)).item()
    loss_record['eval_total_l1_loss'] = torch.sum(torch.stack(l1_losses)).item()
    loss_record['eval_mean_mse_loss'] = torch.mean(torch.stack(mse_losses)).item()
    loss_record['eval_max_mse_loss'] = torch.max(torch.stack(mse_losses)).item()
    loss_record['eval_min_mse_loss'] = torch.min(torch.stack(mse_losses)).item()
    loss_record['eval_mean_l1_loss'] = torch.mean(torch.stack(l1_losses)).item()
    loss_record['eval_max_l1_loss'] = torch.max(torch.stack(l1_losses)).item()
    loss_record['eval_min_l1_loss'] = torch.min(torch.stack(l1_losses)).item()
    loss_record['eval_mse_losses'] = mse_losses
    loss_record['eval_l1_losses'] = l1_losses
    return loss_record


def main(argv):
    pass


if __name__ == '__main__':
    app.run(main)