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
from model_utils.common import *
from run_utils.utils import *

import time
import datetime

from torch.nn.parallel import DistributedDataParallel as DDP
import torch.distributed as dist

import matplotlib
import matplotlib.pyplot as plt
from tqdm import trange

os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "0, 1"

FLAGS = flags.FLAGS

# common run configuration
flags.DEFINE_enum('model', 'Cloth', ['HyperEl','Cloth','Easy_HyperEl'], 'Select model to run.')
flags.DEFINE_string('output_dir','D:\\project_summary\\Graduation Project\\torchMGN\\','path to output_dir')

flags.DEFINE_string('datasets_dir','D:\\project_summary\\Graduation Project\\tmp\\datasets_hdf5','path to datasets')
flags.DEFINE_enum('dataset', 'flag_simple', ['deforming_plate','flag_simple','my_dataset'], 'Select dataset to run')
flags.DEFINE_boolean('prebuild_graph', False, 'is dataloader output graph')
flags.DEFINE_integer('prefetch', 1, 'prefetch size')
flags.DEFINE_integer('batch_size', 1, 'batch size')

flags.DEFINE_float('lr_init',1e-4,'Initial learning rate')
flags.DEFINE_integer('epochs', 2, 'Num of training epochs')
flags.DEFINE_integer('max_steps', 10 ** 6, 'Num of training steps')
flags.DEFINE_integer('nsave_steps', int(5000), help='Number of steps at which to save the model.')

# devices selection
flags.DEFINE_integer('gpu_id', 0, help='choose which gpu to use')
flags.DEFINE_integer('gpu_num', 1, help='choose how many gpus to use')

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
# decide whether continue the last trained steps or start new steps
flags.DEFINE_bool('start_new_trained_step', True, 'Decide whether continue the last trained steps or start new steps')
device = None

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
                       'lr_init':FLAGS.lr_init,
                       'epochs': FLAGS.epochs, 
                       'max_steps':FLAGS.max_steps,
                       'nsave_steps':FLAGS.nsave_steps,
                       'output_size':FLAGS.output_size,
                       'core_model': FLAGS.core_model,
                       'message_passing_aggregator': FLAGS.message_passing_aggregator,
                       'message_passing_steps': FLAGS.message_passing_steps,
                    #    'is_use_world_edge':FLAGS.is_use_world_edge,
                       'dataset_dir': os.path.join(FLAGS.datasets_dir, FLAGS.dataset),
                       'last_run_dir': last_run_dir}    
        
    if last_run_dir is not None and use_prev_config:
        last_run_step_dir = find_nth_latest_run_step(last_run_dir, 1)
        run_step_config = pickle_load(os.path.join(last_run_step_dir, 'log', 'config.pkl'))
        run_step_config['last_run_dir'] = last_run_dir
        run_step_config['last_run_step_dir'] = last_run_step_dir

    # The run_step_config above will be fixed in continuous trainings
    # The configurations will be changed relying on every training settings 

    if FLAGS.datasets_dir[-4:]=='hdf5':
        run_step_config['use_hdf5'] = True
    else:
        run_step_config['use_hdf5'] = False

    run_step_config['batch_size'] = FLAGS.batch_size
    run_step_config['prefetch'] = FLAGS.prefetch
    run_step_config['prebuild_graph'] = FLAGS.prebuild_graph
    run_step_config['gpu_id'] = FLAGS.gpu_id
    run_step_config['gpu_num'] = FLAGS.gpu_num

    run_step_config['start_new_trained_step'] = FLAGS.start_new_trained_step


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

    # save the run_step_config
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
    root_logger.info("Start training......")

    # create or load model
    model = eval(run_step_config['model']).Model(run_step_config['output_size'], 
                                                 run_step_config['message_passing_aggregator'],
                                                 run_step_config['message_passing_steps'],
                                                #  run_step_config['is_use_world_edge'],
                                                )
    run_step_config['noise_scale'] = model.noise_scale
    run_step_config['noise_gamma'] = model.noise_gamma
    run_step_config['noise_field'] = model.noise_field
    run_step_config['build_graph'] = model.build_graph
    
    if FLAGS.prebuild_graph:
        loss_fn = eval(run_step_config['model']).loss_fn_alter
    else:
        loss_fn = eval(run_step_config['model']).loss_fn

    if last_run_dir is not None:
        last_run_step_dir = find_nth_latest_run_step(last_run_dir, 2)
        model.load_model(os.path.join(last_run_step_dir, 'checkpoint', "model_checkpoint"))
        root_logger.info(
            "Loaded checkpoint file in " + str(
                os.path.join(last_run_step_dir, 'checkpoint')) + " and starting retraining...")
        
    # if run_step_config['gpu_num'] > 1:
    #     device = 'cuda'
    
    model.to(device)


    # run summary
    log_run_summary(root_logger, run_step_config, run_step_dir)
    
    # training part

    train_start = time.time()
    learner(model, loss_fn, run_step_config, device)# <--- The training progress is in run_utils  
    train_end = time.time()
    
    # save the running time
    train_elapsed_time_in_second = train_end - train_start
    train_elapsed_time_in_second_pkl_file = os.path.join(log_dir, 'train_elapsed_time_in_second.pkl')
    Path(train_elapsed_time_in_second_pkl_file).touch()
    pickle_save(train_elapsed_time_in_second_pkl_file, train_elapsed_time_in_second)

    root_logger.info("Finished training......")
    
if __name__ == '__main__':
    app.run(main)