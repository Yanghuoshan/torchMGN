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


FLAGS = flags.FLAGS

# common run configuration
flags.DEFINE_enum('model', 'Cloth', ['HyperEl','Cloth'], 'Select model to run.')
flags.DEFINE_string('output_dir','D:\\project_summary\\Graduation Project\\torchMGN\\','path to output_dir')

flags.DEFINE_string('datasets_dir','D:\\project_summary\\Graduation Project\\tmp\\datasets_np','path to datasets')
flags.DEFINE_enum('dataset', 'flag_simple', ['deforming_plate','flag_simple'], 'Select dataset to run')
flags.DEFINE_boolean('is_data_graph',False,'is dataloader output graph')
flags.DEFINE_integer('prefetch',1,'prefetch size')

# flags.DEFINE_float('lr_init',1e-4,'Initial learning rate')
# flags.DEFINE_integer('epochs', 2, 'Num of training epochs')
# flags.DEFINE_integer('max_steps', 10 ** 6, 'Num of training steps')
# flags.DEFINE_integer('nsave_steps', int(5000), help='Number of steps at which to save the model.')

flags.DEFINE_integer('gpu_id', 0, help='choose which gpu to use')

# core model configuration
flags.DEFINE_integer('output_size', 3, 'Num of output_size')
flags.DEFINE_enum('core_model', 'encode_process_decode', ['encode_process_decode'], 'Core model to be used')
flags.DEFINE_enum('message_passing_aggregator', 'sum', ['sum', 'max', 'min', 'mean', 'pna'], 'No. of training epochs')
flags.DEFINE_integer('message_passing_steps', 5, 'No. of training epochs')
# flags.DEFINE_boolean('is_use_world_edge', False, 'Is the model use world edges') 

# decide whether to use the previous model state
flags.DEFINE_string('model_last_run_dir', None, 'Path to the checkpoint file of a network that should continue training')
# decide whether to use the configuration from last run step. If not use the previous
flags.DEFINE_boolean('use_prev_config', True, 'Decide whether to use the configuration from last run step')

device = None


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


def n_step_evaluator(params, model, run_step_config, n_step_list, n_traj=1):
    model_type = run_step_config['model']

    ds_loader = dataset.load_dataset(run_step_config['dataset_dir'], run_step_config['rollout_split'], add_targets=True)
    ds_iterator = iter(ds_loader)

    n_step_mse_losses = {}
    n_step_l1_losses = {}

    # Take n_traj trajectories from valid set for n_step loss calculation
    for i in range(n_traj):
        trajectory = next(ds_iterator)
        trajectory = process_trajectory(trajectory, params, model_type, run_step_config['dataset_dir'], True)
        for n_step in n_step_list:
            mse_losses = []
            l1_losses = []
            for step in range(len(trajectory['world_pos']) - n_step):
                eval_traj = {}
                for k, v in trajectory.items():
                    eval_traj[k] = v[step:step + n_step + 1]
                _, prediction_trajectory = params['evaluator'].evaluate(model, eval_traj, n_step + 1)
                mse_loss_fn = torch.nn.MSELoss()
                l1_loss_fn = torch.nn.L1Loss()
                if model_type == 'cloth':
                    mse_loss = mse_loss_fn(torch.squeeze(eval_traj['world_pos'], dim=0), prediction_trajectory['pred_pos'])
                    l1_loss = l1_loss_fn(torch.squeeze(eval_traj['world_pos'], dim=0), prediction_trajectory['pred_pos'])
                elif model_type == 'cfd':
                    mse_loss = mse_loss_fn(torch.squeeze(eval_traj['velocity'], dim=0), prediction_trajectory['pred_velocity'])
                    l1_loss = l1_loss_fn(torch.squeeze(eval_traj['velocity'], dim=0), prediction_trajectory['pred_velocity'])
                elif model_type == 'deform':
                    mse_loss = mse_loss_fn(torch.squeeze(eval_traj['world_pos'], dim=0), prediction_trajectory['pred_pos'])
                    l1_loss = l1_loss_fn(torch.squeeze(eval_traj['world_pos'], dim=0), prediction_trajectory['pred_pos'])
                mse_losses.append(mse_loss.cpu())
                l1_losses.append(l1_loss.cpu())
            if n_step not in n_step_mse_losses and n_step not in n_step_l1_losses:
                n_step_mse_losses[n_step] = torch.stack(mse_losses)
                n_step_l1_losses[n_step] = torch.stack(l1_losses)
            elif n_step in n_step_mse_losses and n_step in n_step_l1_losses:
                n_step_mse_losses[n_step] = n_step_mse_losses[n_step] + torch.stack(mse_losses)
                n_step_l1_losses[n_step] = n_step_l1_losses[n_step] + torch.stack(l1_losses)
            else:
                raise Exception('Error when computing n step losses!')
    for (kmse, vmse), (kl1, vl1) in zip(n_step_mse_losses.items(), n_step_l1_losses.items()):
        n_step_mse_losses[kmse] = torch.div(vmse, i + 1)
        n_step_l1_losses[kl1] = torch.div(vl1, i + 1)

    return {'n_step_mse_loss': n_step_mse_losses, 'n_step_l1_loss': n_step_l1_losses}


def main(argv):
    global device
    device = torch.device(f'cuda:{FLAGS.gpu_id}')
    # record start time
    run_step_start_time = time.time()
    run_step_start_datetime = datetime.datetime.fromtimestamp(run_step_start_time).strftime('%c')
    
    # load config from previous run step if last run dir is specified
    # 如果 last_run_dir 没有指定，则模型和文件都会新建
    # 如果 last_run_dir 指定了目录，且use_prev_config为True，则会用最后一次的run_step_config
    # 如果 last_run_dir 指定了目录，但use_prev_config为False，则会用新建的run_step_config
    # 如果 last_run_dir 指定了目录，不管use_prev_config，模型会加载最后一次的checkpoint
    last_run_dir = FLAGS.model_last_run_dir
    use_prev_config = FLAGS.use_prev_config
    run_step_config = {'model': FLAGS.model, 
                       'dataset': FLAGS.dataset, 
                       'epochs': FLAGS.epochs, 
                       'max_steps':FLAGS.max_steps,
                       'nsave_steps':FLAGS.nsave_steps,
                       'output_size':FLAGS.output_size,
                       'core_model': FLAGS.core_model,
                       'message_passing_aggregator': FLAGS.message_passing_aggregator,
                       'message_passing_steps': FLAGS.message_passing_steps,
                    #    'is_use_world_edge':FLAGS.is_use_world_edge,
                       'dataset_dir': os.path.join(FLAGS.datasets_dir, FLAGS.dataset),
                       'last_run_dir': None}    
    if last_run_dir is not None and use_prev_config:
        last_run_step_dir = find_nth_latest_run_step(last_run_dir, 1)
        run_step_config = pickle_load(os.path.join(last_run_step_dir, 'log', 'config.pkl'))
        run_step_config['last_run_dir'] = last_run_dir
        run_step_config['last_run_step_dir'] = last_run_step_dir


    # setup directories
    output_dir = os.path.join(FLAGS.output_dir, 'output', run_step_config['model']) # 如果last_run_dir没有指定。则在output文件夹里创建新的run_dir
    run_step_dir = prepare_files_and_directories(last_run_dir, output_dir)
    checkpoint_dir = os.path.join(run_step_dir, 'checkpoint')
    log_dir = os.path.join(run_step_dir, 'log')
    rollout_dir = os.path.join(run_step_dir, 'rollout')

    run_step_config['checkpoint_dir'] = checkpoint_dir
    run_step_config['rollout_dir'] = rollout_dir

    # setup logger
    root_logger = logger_setup(os.path.join(log_dir, 'log.log'))
    run_step_config_save_path = os.path.join(log_dir, 'config.pkl')
    Path(run_step_config_save_path).touch()
    pickle_save(run_step_config_save_path, run_step_config)

    if last_run_dir is not None and use_prev_config:
        root_logger.info("=========================================================")
        root_logger.info("Continue run in " + str(run_step_dir))
        root_logger.info("=========================================================")
    else:
        root_logger.info("=========================================================")
        root_logger.info("Start new run in " + str(run_step_dir))
        root_logger.info("=========================================================")
    

    root_logger.info("Program started at time " + str(run_step_start_datetime))
    root_logger.info("Start evaluating......")

    # create or load model
    model = eval(run_step_config['model']).Model(run_step_config['output_size'], 
                                                 run_step_config['message_passing_aggregator'],
                                                 run_step_config['message_passing_steps'],
                                                #  run_step_config['is_use_world_edge'],
                                                )

    if last_run_dir is not None:
        last_run_step_dir = find_nth_latest_run_step(last_run_dir, 2)
        model.load_model(os.path.join(last_run_step_dir, 'checkpoint', "model_checkpoint"))
        root_logger.info(
            "Loaded checkpoint file in " + str(
                os.path.join(last_run_step_dir, 'checkpoint')) + " and starting retraining...")

    model.evaluate()  
    model.to(device)

    # evaluating process
    eval_loss_record = evaluator(params, model, run_step_config)
    step_loss = n_step_evaluator(params, model, run_step_config, n_step_list=[1, 3, 5, 7, 10], n_traj=1)


    root_logger.info("--------------------eval loss record---------------------")
    
    eval_loss_pkl_file = os.path.join(log_dir, 'eval_loss.pkl')
    Path(eval_loss_pkl_file).touch()
    pickle_save(eval_loss_pkl_file, eval_loss_record)
    for item in eval_loss_record.items():
        root_logger.info(item)
    step_loss_mse_pkl_file = os.path.join(log_dir, 'step_loss_mse.pkl')
    Path(step_loss_mse_pkl_file).touch()
    pickle_save(step_loss_mse_pkl_file, step_loss['n_step_mse_loss'])
    step_loss_l1_pkl_file = os.path.join(log_dir, 'step_loss_l1.pkl')
    Path(step_loss_l1_pkl_file).touch()
    pickle_save(step_loss_l1_pkl_file, step_loss['n_step_l1_loss'])
    root_logger.info("---------------------------------------------------------")


    # run summary
    log_run_summary(root_logger, run_step_config, run_step_dir)


if __name__ == '__main__':
    app.run(main)