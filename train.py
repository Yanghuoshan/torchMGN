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
flags.DEFINE_boolean('is_data_graph', False, 'is dataloader output graph')
flags.DEFINE_integer('prefetch', 1, 'prefetch size')

flags.DEFINE_float('lr_init',1e-4,'Initial learning rate')
flags.DEFINE_integer('epochs', 2, 'Num of training epochs')
flags.DEFINE_integer('max_steps', 10 ** 6, 'Num of training steps')
flags.DEFINE_integer('nsave_steps', int(5000), help='Number of steps at which to save the model.')

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
# decide whether continue the last trained steps or start new steps
flags.DEFINE_bool('start_new_trained_step', True, 'Decide whether continue the last trained steps or start new steps')
device = None


def save_checkpoint(model, optimizer, scheduler, step, losses, run_step_config):
    # save checkpoint to prevent interruption
    model.save_model(os.path.join(run_step_config['checkpoint_dir'], f"model_checkpoint"))
    torch.save(optimizer.state_dict(), os.path.join(run_step_config['checkpoint_dir'], f"optimizer_checkpoint.pth"))
    torch.save(scheduler.state_dict(), os.path.join(run_step_config['checkpoint_dir'], f"scheduler_checkpoint.pth"))
    # save the steps that have been already trained
    torch.save({'trained_step': step}, os.path.join(run_step_config['checkpoint_dir'], "step_checkpoint.pth"))
    # save the previous losses
    torch.save({'losses': losses}, os.path.join(run_step_config['checkpoint_dir'], "losses_checkpoint.pth"))


def learner(model, loss_fn, run_step_config):
    root_logger = logging.getLogger()
    root_logger.info(f"Use gpu {FLAGS.gpu_id}")
    optimizer = torch.optim.Adam(model.parameters(), lr=FLAGS.lr_init)
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
    if run_step_config['last_run_dir'] is not None and not FLAGS.start_new_trained_step:
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
                ds_loader = datasets.get_dataloader_hdf5(run_step_config['dataset_dir'],
                                                    model=run_step_config['model'],
                                                    split='train',
                                                    shuffle=True,
                                                    prefetch=FLAGS.prefetch, 
                                                    is_data_graph=FLAGS.is_data_graph)
            else:
                ds_loader = datasets.get_dataloader(run_step_config['dataset_dir'],
                                                    model=run_step_config['model'],
                                                    split='train',
                                                    shuffle=True,
                                                    prefetch=FLAGS.prefetch, 
                                                    is_data_graph=FLAGS.is_data_graph)
            root_logger.info("Epoch " + str(epoch + 1) + "/" + str(run_step_config['epochs']))
            ds_iterator = iter(ds_loader)

            # dry run
            if is_dry_run:
                if FLAGS.is_data_graph:
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
                if FLAGS.is_data_graph:
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
    if FLAGS.datasets_dir[-4:]=='hdf5':
        run_step_config['use_hdf5'] = True
    else:
        run_step_config['use_hdf5'] = False

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
    root_logger.info("Start training......")

    # create or load model
    model = eval(run_step_config['model']).Model(run_step_config['output_size'], 
                                                 run_step_config['message_passing_aggregator'],
                                                 run_step_config['message_passing_steps'],
                                                #  run_step_config['is_use_world_edge'],
                                                )
    if FLAGS.is_data_graph:
        loss_fn = eval(run_step_config['model']).loss_fn_alter
    else:
        loss_fn = eval(run_step_config['model']).loss_fn

    if last_run_dir is not None:
        last_run_step_dir = find_nth_latest_run_step(last_run_dir, 2)
        model.load_model(os.path.join(last_run_step_dir, 'checkpoint', "model_checkpoint"))
        root_logger.info(
            "Loaded checkpoint file in " + str(
                os.path.join(last_run_step_dir, 'checkpoint')) + " and starting retraining...")
        
    model.to(device)

    # run summary
    log_run_summary(root_logger, run_step_config, run_step_dir)
    
    # training part

    train_start = time.time()
    learner(model, loss_fn, run_step_config)# <--- Training progress 
    train_end = time.time()
    
    # save the running time
    train_elapsed_time_in_second = train_end - train_start
    train_elapsed_time_in_second_pkl_file = os.path.join(log_dir, 'train_elapsed_time_in_second.pkl')
    Path(train_elapsed_time_in_second_pkl_file).touch()
    pickle_save(train_elapsed_time_in_second_pkl_file, train_elapsed_time_in_second)

    root_logger.info("Finished training......")
    
if __name__ == '__main__':
    app.run(main)