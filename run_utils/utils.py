import sys
import os
import pathlib
import gc
from pathlib import Path
import logging
import time
import datetime
import pickle
from absl import app
from absl import flags
import torch
from matplotlib import pyplot as plt

from dataset_utils import datasets 
from model_utils.encode_process_decode import init_weights
from model_utils.common import *

os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'


def pickle_save(path, data):
    with open(path, 'wb') as f:
        pickle.dump(data, f)

def pickle_load(path):
    with open(path, 'rb') as f:
        return pickle.load(f)

def find_nth_latest_run_step(run_dir, n):
    """
    找到最新的运行目录
    """
    all_run_step_dirs = os.listdir(run_dir)
    all_run_step_dirs = map(lambda d: os.path.join(run_dir, d), all_run_step_dirs)
    all_run_step_dirs = [d for d in all_run_step_dirs if os.path.isdir(d)]
    nth_latest_run_step_dir = sorted(all_run_step_dirs, key=os.path.getmtime)[-n]
    return nth_latest_run_step_dir

def prepare_files_and_directories(last_run_dir, output_dir):
    '''
        The following code is about creating all the necessary files and directories for the run
        If last_run_dir is None, use output_dir to create new run_dir within new run_step_dir
    '''
    # if last run dir is not specified, then new run dir should be created, otherwise use run specified by argument
    if last_run_dir is not None:
        run_dir = last_run_dir
    else:
        run_create_time = time.time()
        run_create_datetime = datetime.datetime.fromtimestamp(run_create_time).strftime('%c')
        run_create_datetime_datetime_dash = run_create_datetime.replace(" ", "-").replace(":", "-")
        run_dir = os.path.join(output_dir, run_create_datetime_datetime_dash)
        Path(run_dir).mkdir(parents=True, exist_ok=True)

    # check for last run step dir and if exists, create a new run step dir with incrementing dir name, otherwise create the first run step dir
    all_run_step_dirs = os.listdir(run_dir)
    if not all_run_step_dirs:
        run_step_dir = os.path.join(run_dir, '1')
    else:
        latest_run_step_dir = find_nth_latest_run_step(run_dir, 1)
        run_step_dir = str(int(Path(latest_run_step_dir).name) + 1)
        run_step_dir = os.path.join(run_dir, run_step_dir)

    # make all the necessary directories
    checkpoint_dir = os.path.join(run_step_dir, 'checkpoint')
    log_dir = os.path.join(run_step_dir, 'log')
    rollout_dir = os.path.join(run_step_dir, 'rollout')
    Path(checkpoint_dir).mkdir(parents=True, exist_ok=True)
    Path(log_dir).mkdir(parents=True, exist_ok=True)
    Path(rollout_dir).mkdir(parents=True, exist_ok=True)

    return run_step_dir

def logger_setup(log_path):
    # set log configuration
    root_logger = logging.getLogger()
    root_logger.setLevel(logging.INFO)
    # console_output_handler = logging.StreamHandler(sys.stdout)
    # console_output_handler.setLevel(logging.INFO)
    file_log_handler = logging.FileHandler(filename=log_path, mode='w', encoding='utf-8')
    file_log_handler.setLevel(logging.INFO)
    formatter = logging.Formatter(fmt='%(asctime)s - %(message)s')
    # console_output_handler.setFormatter(formatter)
    file_log_handler.setFormatter(formatter)
    # root_logger.addHandler(console_output_handler)
    root_logger.addHandler(file_log_handler)
    return root_logger

def log_run_summary(root_logger, run_step_config, run_step_dir):
    root_logger.info("")
    root_logger.info("=======================Run Summary=======================")
    root_logger.info("Simulation task is " + str(run_step_config['model']) + " simulation")
    
    root_logger.info("Core model is " + run_step_config['core_model'])
    root_logger.info("Message passing aggregator is " + run_step_config['message_passing_aggregator'])
    root_logger.info("Message passing steps are " + str(run_step_config['message_passing_steps']))
    root_logger.info("Graphs prebuilding is " + str(run_step_config['prebuild_graph']))
    
    root_logger.info("Run output directory is " + run_step_dir)
    root_logger.info("=========================================================")
    root_logger.info("")

def show_loss_graph(losses, save_path):
    plt.plot(losses)
    plt.xlabel('Interval (x100 steps)')
    plt.ylabel('Loss')
    plt.title('Loss Over Time')
    plt.savefig(os.path.join(save_path,"train_loss.png"))


def save_checkpoint(model, optimizer, scheduler, epoch, step, losses, run_step_config):
    # save checkpoint to prevent interruption
    model.save_model(os.path.join(run_step_config['checkpoint_dir'], f"model_checkpoint"))
    torch.save(optimizer.state_dict(), os.path.join(run_step_config['checkpoint_dir'], f"optimizer_checkpoint.pth"))
    torch.save(scheduler.state_dict(), os.path.join(run_step_config['checkpoint_dir'], f"scheduler_checkpoint.pth"))
    # save the steps that have been already trained
    torch.save({'trained_epoch': epoch,'trained_step': step}, os.path.join(run_step_config['checkpoint_dir'], "epoch_n_step_checkpoint.pth"))
    # save the previous losses
    torch.save({'losses': losses}, os.path.join(run_step_config['checkpoint_dir'], "losses_checkpoint.pth"))


def learner(model, loss_fn, run_step_config, device):
    root_logger = logging.getLogger()
    root_logger.info(f"Use gpu {run_step_config['gpu_id']}")
    optimizer = torch.optim.Adam(model.parameters(), lr=run_step_config['lr_init'])
    scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, 0.1 + 1e-6, last_epoch=-1)

    losses = []
    loss_save_interval = 1000
    loss_save_cnt = 0
    running_loss = 0.0
    trained_epoch = 0
    trained_step = 0
    scheduler_flag = True

    if run_step_config['last_run_dir'] is not None:
        optimizer.load_state_dict(torch.load(os.path.join(run_step_config['last_run_step_dir'], 'checkpoint', "optimizer_checkpoint.pth")))
        scheduler.load_state_dict(torch.load(os.path.join(run_step_config['last_run_step_dir'], 'checkpoint', "scheduler_checkpoint.pth")))
        trained_step = torch.load(os.path.join(run_step_config['last_run_step_dir'], 'checkpoint', "step_checkpoint.pth"))['trained_step'] + 1
        trained_epoch = torch.load(os.path.join(run_step_config['last_run_step_dir'], 'checkpoint', "step_checkpoint.pth"))['trained_epoch']
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
    start_epoch = 0
    if run_step_config['last_run_dir'] is not None and not run_step_config['start_new_trained_step']:
        start_epoch = trained_epoch
        step = trained_step
    
    # dry run for lazy linear layers initialization
    is_dry_run = True
    if run_step_config['last_run_dir'] is not None:
        is_dry_run = False
        root_logger.info("No Dry run")

    while not_reached_max_steps:
        for epoch in range(run_step_config['epochs'])[start_epoch:]:
            # model will train itself with the whole dataset
            if run_step_config['use_hdf5']:
                if run_step_config['prebuild_graph'] == True:
                    ds_loader = datasets.get_dataloader_hdf5_batch(run_step_config['dataset_dir'],
                                                            model=run_step_config['model'],
                                                            split='train',
                                                            shuffle=True,
                                                            prefetch=run_step_config['prefetch'], 
                                                            batch_size=run_step_config['batch_size'],
                                                            add_noise_fn=add_noise_fn(model.noise_field, model.noise_scale, model.noise_gamma),
                                                            prebuild_graph_fn=model.build_graph)
                else:
                    ds_loader = datasets.get_dataloader_hdf5_batch(run_step_config['dataset_dir'],
                                                            model=run_step_config['model'],
                                                            split='train',
                                                            shuffle=True,
                                                            prefetch=run_step_config['prefetch'], 
                                                            batch_size=run_step_config['batch_size'],
                                                            add_noise_fn=add_noise_fn(model.noise_field, model.noise_scale, model.noise_gamma))
            else:
                ds_loader = datasets.get_dataloader(run_step_config['dataset_dir'],
                                                    model=run_step_config['model'],
                                                    split='train',
                                                    shuffle=True,
                                                    prefetch=run_step_config['prefetch'])
            root_logger.info("Epoch " + str(epoch + 1) + "/" + str(run_step_config['epochs']))
            ds_iterator = iter(ds_loader)

            # dry run
            if is_dry_run:
                input = next(ds_iterator)[0]
                if run_step_config['prebuild_graph']:
                    input[0]=input[0].to(device)
                    input[1]=input[1].to(device)
                    input[2]=input[2].to(device)
                    model(input[0], is_training = True, prebuild_graph = True)
                else:
                    for k in input:
                        input[k]=input[k].to(device)
                    model(input, is_training = True, prebuild_graph = False)

                model.apply(init_weights)
                    
                is_dry_run = False
                root_logger.info("Dry run finished")
            
            # start to train
            for input in ds_iterator:
                input = input[0]
                
                if run_step_config['prebuild_graph']:
                    input[0]=input[0].to(device)
                    input[1]=input[1].to(device)
                    input[2]=input[2].to(device)
                    out = model(input[0], is_training = True, prebuild_graph = True)

                    loss = loss_fn(input[1],out,input[2],model)
                else:
                    for k in input:
                        input[k]=input[k].to(device)
                    out = model(input, is_training = True, prebuild_graph = False)
                
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

                # Adjust the lr when the training step is larger then 0.6*max_steps
                if ((step+1) > (run_step_config['max_steps']*0.6)) and scheduler_flag:
                    scheduler.step()
                    root_logger.info("Call scheduler in step " + str(step + 1))
                    scheduler_flag = False
                
                # memory cleaning
                if ((step+1) % 10000) == 0:
                    gc.collect()
                    torch.cuda.empty_cache()

                step += 1

            # Break if step reaches the maximun
            if not_reached_max_steps == False:
                break

            # Save the model state between epochs
            save_checkpoint(model, optimizer, scheduler, step, losses, run_step_config)
            
            # if epoch == 3:
            #     scheduler.step()
            #     root_logger.info("Call scheduler in epoch " + str(epoch + 1))

    # Save the model state in the end
    save_checkpoint(model, optimizer, scheduler, step, losses, run_step_config)