import sys
import os
import pathlib
from pathlib import Path
import logging
import time
import datetime
import pickle
from absl import app
from absl import flags
import torch
from matplotlib import pyplot as plt


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
    
    root_logger.info("Run output directory is " + run_step_dir)
    root_logger.info("=========================================================")
    root_logger.info("")

def show_loss_graph(losses):
    plt.plot(losses)
    plt.xlabel('Interval (x100 steps)')
    plt.ylabel('Loss')
    plt.title('Loss Over Time')
    plt.show()