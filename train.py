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
flags.DEFINE_string('model_last_run_dir', None, 
                    # os.path.join('E:\\meshgraphnets\\output\\deforming_plate', 'Sat-Feb-12-12-14-04-2022'),
                    # os.path.join('/home/i53/student/ruoheng_ma/meshgraphnets/output/deforming_plate', 'Mon-Jan--3-15-18-53-2022'),
                    'Path to the checkpoint file of a network that should continue training')
# decide whether to use the configuration from last run step
flags.DEFINE_boolean('use_prev_config', True, 'Decide whether to use the configuration from last run step')
device = None


def learner(model, loss_fn, run_step_config):
    root_logger = logging.getLogger()
    root_logger.info(f"Use gpu {FLAGS.gpu_id}")
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
    scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, 0.1 + 1e-6, last_epoch=-1)
    trained_epoch = 0
    if run_step_config['last_run_dir'] is not None:
        optimizer.load_state_dict(
            torch.load(os.path.join(run_step_config['last_run_step_dir'], 'checkpoint', "optimizer_checkpoint.pth")))
        scheduler.load_state_dict(
            torch.load(os.path.join(run_step_config['last_run_step_dir'], 'checkpoint', "scheduler_checkpoint.pth")))
        # epoch_checkpoint = torch.load(
        #     os.path.join(run_step_config['last_run_step_dir'], 'checkpoint', "epoch_checkpoint.pth"))
        # trained_epoch = epoch_checkpoint['epoch'] + 1
        root_logger.info("Loaded optimizer, scheduler and model epoch checkpoint\n")

    # model training
    # epoch_training_losses = []
    fixed_pass_count = 500 # equal to pass_count but doesn't change
    pass_count = 500
    if run_step_config['last_run_dir'] is not None:
        pass_count = 0

    not_reached_max_steps = True
    step = 0
    loss_report_step = 1
    loss_save_interval = 1000
    running_loss = 0.0
    loss_count = 0
    losses = []

    is_dry_run = True

    while not_reached_max_steps:
        for epoch in range(run_step_config['epochs'])[trained_epoch:]:
            # model will train itself with the whole dataset
            ds_loader = datasets.get_dataloader(run_step_config['dataset_dir'],
                                                model=run_step_config['model'],
                                                split='train',
                                                shuffle=True,
                                                prefetch=FLAGS.prefetch, 
                                                is_data_graph=FLAGS.is_data_graph)
            root_logger.info("Epoch " + str(epoch + 1) + "/" + str(run_step_config['epochs']))
            epoch_training_loss = 0.0
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
                    if (step + 1 - fixed_pass_count) % loss_save_interval == 0:
                        avg_loss = running_loss / loss_save_interval
                        losses.append(avg_loss)
                        running_loss = 0.0
                        root_logger.info(f"Step [{step+1}], Loss: {avg_loss:.4f}")

                # Reprot loss
                # if (step+1) % loss_report_step == 0:
                #     root_logger.info(f"Training step: {step+1}/{run_step_config['max_steps']}. Loss: {loss}.")

                # Save model state
                if (step+1) % run_step_config['nsave_steps'] == 0:
                    model.save_model(os.path.join(run_step_config['checkpoint_dir'], f"model_checkpoint"))
                    torch.save(optimizer.state_dict(), os.path.join(run_step_config['checkpoint_dir'], f"optimizer_checkpoint.pth"))
                    torch.save(scheduler.state_dict(), os.path.join(run_step_config['checkpoint_dir'], f"scheduler_checkpoint.pth"))
                    loss_record = {}

                # Break if step reaches the maximun
                if (step+1) >= run_step_config['max_steps']:
                    not_reached_max_steps = False
                    break
                
                # 清理内存
                if step % 100 == 0:
                    gc.collect()
                    torch.cuda.empty_cache()

                step += 1

            # Break if step reaches the maximun
            if not_reached_max_steps == False:
                break

            # epoch_training_losses.append(epoch_training_loss)
            # root_logger.info("Current mean of epoch training losses")
            # root_logger.info(torch.mean(torch.stack(epoch_training_losses)))
            model.save_model(os.path.join(run_step_config['checkpoint_dir'], "epoch_model_checkpoint"))
            torch.save(optimizer.state_dict(), os.path.join(run_step_config['checkpoint_dir'], "epoch_optimizer_checkpoint" + ".pth"))
            torch.save(scheduler.state_dict(), os.path.join(run_step_config['checkpoint_dir'], "epoch_scheduler_checkpoint" + ".pth"))
            if epoch == 13:
                scheduler.step()
                root_logger.info("Call scheduler in epoch " + str(epoch))
            # torch.save({'epoch': epoch}, os.path.join(run_step_config['checkpoint_dir'], "epoch_checkpoint.pth"))


    model.save_model(os.path.join(run_step_config['checkpoint_dir'], "model_checkpoint"))
    torch.save(optimizer.state_dict(), os.path.join(run_step_config['checkpoint_dir'], "optimizer_checkpoint.pth"))
    torch.save(scheduler.state_dict(), os.path.join(run_step_config['checkpoint_dir'], "scheduler_checkpoint.pth"))
    loss_record = {}
    loss_record["avglosses_per_100_steps"] = losses[:]
    '''
    loss_record['train_total_loss'] = torch.sum(torch.stack(epoch_training_losses))
    loss_record['train_mean_epoch_loss'] = torch.mean(torch.stack(epoch_training_losses)).item()
    loss_record['train_max_epoch_loss'] = torch.max(torch.stack(epoch_training_losses)).item()
    loss_record['train_min_epoch_loss'] = torch.min(torch.stack(epoch_training_losses)).item()
    loss_record['train_epoch_losses'] = epoch_training_losses
    '''
    return loss_record


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
    root_logger.info("Start training......")

    # create or load model
    model = eval(run_step_config['model']).Model(run_step_config['output_size'], 
                                                 run_step_config['message_passing_aggregator'],
                                                 run_step_config['message_passing_steps'],
                                                #  run_step_config['is_use_world_edge'],
                                                 device = device)
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
    train_loss_record = None
    
    train_start = time.time()
    train_loss_record = learner(model, loss_fn, run_step_config)# <--- Training progress 
    train_end = time.time()
    train_elapsed_time_in_second = train_end - train_start

    # load train loss if exist and combine the previous and current train loss
    if last_run_dir is not None:
        # 将本次中途开始的训练的loss和之前的未完成的训练的loss进行整合，得到完整训练的loss
        saved_train_loss_record = pickle_load(os.path.join(last_run_step_dir, 'log', 'train_loss.pkl'))
        train_loss_record["avglosses_per_100_steps"] = saved_train_loss_record["avglosses_per_100_steps"] + train_loss_record["avglosses_per_100_steps"]
        show_loss_graph(train_loss_record["avglosses_per_100_steps"],log_dir)
        '''
        将本次中途开始的训练的loss和之前的未完成的训练的loss进行整合，得到完整训练的loss
        saved_train_loss_record = pickle_load(os.path.join(last_run_step_dir, 'log', 'train_loss.pkl'))
        train_loss_record['train_epoch_losses'] = saved_train_loss_record['train_epoch_losses'] + \
                                                      train_loss_record['train_epoch_losses']
        train_loss_record['train_total_loss'] = torch.sum(torch.stack(train_loss_record['train_epoch_losses']))
        train_loss_record['train_mean_epoch_loss'] = torch.mean(
                torch.stack(train_loss_record['train_epoch_losses'])).item()
        train_loss_record['train_max_epoch_loss'] = torch.max(
                torch.stack(train_loss_record['train_epoch_losses'])).item()
        train_loss_record['train_min_epoch_loss'] = torch.min(
                torch.stack(train_loss_record['train_epoch_losses'])).item()
        train_loss_record['all_trajectory_train_losses'] = saved_train_loss_record['all_trajectory_train_losses'] + \
                                                               train_loss_record['all_trajectory_train_losses']
        load train elapsed time if exist and combine the previous and current train loss
        '''
        saved_train_elapsed_time_in_second = pickle_load(os.path.join(last_run_step_dir, 'log', 'train_elapsed_time_in_second.pkl'))
        train_elapsed_time_in_second += saved_train_elapsed_time_in_second
    train_elapsed_time_in_second_pkl_file = os.path.join(log_dir, 'train_elapsed_time_in_second.pkl')
    Path(train_elapsed_time_in_second_pkl_file).touch()
    pickle_save(train_elapsed_time_in_second_pkl_file, train_elapsed_time_in_second)

    # save train loss
    train_loss_pkl_file = os.path.join(log_dir, 'train_loss.pkl')
    Path(train_loss_pkl_file).touch()
    pickle_save(train_loss_pkl_file, train_loss_record)

    root_logger.info("Finished training......")
    
if __name__ == '__main__':
    app.run(main)