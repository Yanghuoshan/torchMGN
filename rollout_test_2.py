import sys
import os
import pathlib
from pathlib import Path
import pickle
from absl import app
from absl import flags
import torch
from model_utils import HyperEl,Cloth,Easy_HyperEl,IncompNS,Inflaction,HyperEl2d
from model_utils import deform_eval
from model_utils import common
from model_utils.encode_process_decode import init_weights
from render_utils import Cloth_render, HyperEl_render, Easy_HyperEl_render, IncompNS_render, Inflaction_render, HyperEl2d_render
from run_utils.utils import *
import logging
import numpy as np
import json
from model_utils.common import NodeType

from torch.nn.parallel import DistributedDataParallel as DDP
import torch.distributed as dist

import matplotlib
import matplotlib.pyplot as plt

from dataset_utils import datasets

device = torch.device('cuda')

os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'


last_run_dir = "D:\project_summary\Graduation Project\\torchMGN\output\HyperEl2d\Mon-Mar--3-14-20-27-2025"
# ds_dir = "D:\project_summary\Graduation Project\\tmp\datasets_np\\flag_simple"
ds_dir ="D:\project_summary\Graduation Project\\tmp\datasets_hdf5\\vessel2d"
trajectory_index = 20
split = "test"

last_run_step_dir = find_nth_latest_run_step(last_run_dir, 1)
run_step_config = pickle_load(os.path.join(last_run_step_dir, 'log', 'config.pkl'))
run_step_config['last_run_dir'] = last_run_dir
run_step_config['last_run_step_dir'] = last_run_step_dir

if run_step_config['model'] == 'Cloth':
    render = Cloth_render.render
elif run_step_config['model'] == 'HyperEl':
    render = HyperEl_render.render
    run_step_config['latent_size'] = 128
elif run_step_config['model']== 'Easy_HyperEl':
    render = Easy_HyperEl_render.render
elif run_step_config['model']== 'IncompNS':
    render = IncompNS_render.render
elif run_step_config['model']== 'Inflaction':
    render = Inflaction_render.render
elif run_step_config['model'] == 'HyperEl2d':
    render = HyperEl2d_render.render3d


M = eval(run_step_config['model'])
model = M.Model(run_step_config['output_size'], 
                run_step_config['message_passing_aggregator'],
                run_step_config['message_passing_steps'],
                run_step_config['latent_size']
                )
last_run_step_dir = find_nth_latest_run_step(last_run_dir, 1)
model.load_model(os.path.join(last_run_step_dir, 'checkpoint', "model_checkpoint"))
print("Load model:",os.path.join(last_run_step_dir, 'checkpoint', "model_checkpoint"))
# model.load_model(os.path.join(last_run_step_dir, 'checkpoint', "model_checkpoint"))

model.to(device)
# print(run_step_config['latent_size'])
rollout = M.rollout

is_data_graph = False
steps = 399


dl = datasets.get_trajectory_dataloader(ds_dir,
                                        model=run_step_config['model'],
                                        is_data_graph=is_data_graph, 
                                        trajectory_index=trajectory_index,
                                        split=split)
if run_step_config['model']=="Cloth":
    trajectory = iter(dl)
    init_state = next(trajectory)[0]
    for k in init_state:
        init_state[k] = init_state[k].to(device)
    new_trajectory =rollout(model,init_state,399)
    anim = render(new_trajectory, skip=5)
    anim.save('animation4.gif', writer='pillow')
elif run_step_config['model']=="HyperEl":
    for i in range(6):
        dl = datasets.get_trajectory_dataloader(ds_dir,
                                            model=run_step_config['model'],
                                            is_data_graph=is_data_graph, 
                                            trajectory_index=i + 20,
                                            split=split)
        trajectory = iter(dl)
        new_trajectory =rollout(model,trajectory,300)
        print(new_trajectory["world_pos"].size(),new_trajectory["cells"].size())
        anim = render(new_trajectory, skip=5)
        anim.save(f'HyperEl{i + 20}.gif', writer='pillow')
elif run_step_config['model'] == 'Easy_HyperEl':
    trajectory = iter(dl)
    new_trajectory =rollout(model,trajectory,99)
    print(new_trajectory["world_pos"].size(),new_trajectory["cells"].size())
    print(new_trajectory["world_pos"])
    print(new_trajectory["cells"])
    for i in range(99):
        plt.scatter(new_trajectory["world_pos"][i][:,0].to('cpu'),new_trajectory["world_pos"][i][:,1].to('cpu'))
        plt.show()
    # anim = render(new_trajectory, skip=5)
    # anim.save('animation5.gif', writer='pillow')
elif run_step_config['model'] == 'IncompNS':
    for i in range(1):
        dl = datasets.get_trajectory_dataloader(ds_dir,
                                            model=run_step_config['model'],
                                            is_data_graph=is_data_graph, 
                                            trajectory_index=i+4,
                                            split=split)
        trajectory = iter(dl)
        new_trajectory =rollout(model,trajectory,590)
        print(new_trajectory["mesh_pos"].size(),new_trajectory["cells"].size(),new_trajectory["velocity"].size(),new_trajectory["pressure"].size())
        anim = render(new_trajectory,skip=5, color_field ='pressure')
        anim.save(f'IncompNSCylinderflowPressure{i+4}.gif', writer='pillow')
elif run_step_config['model'] == 'Inflaction':
    trajectory = iter(dl)
    new_trajectory =rollout(model,trajectory,140)
    print(new_trajectory["world_pos"].size(),new_trajectory["triangles"].size(),new_trajectory["rectangles"].size())
    anim = render(new_trajectory,skip=1)
    anim.save('InflactionWithR3.gif', writer='pillow')
elif run_step_config['model'] == 'HyperEl2d':
    trajectory = iter(dl)
    new_trajectory =rollout(model,trajectory,399)
    print(new_trajectory["world_pos"].size(),new_trajectory["cells"].size(),new_trajectory["stress"].size(),new_trajectory["node_type"].size())
    anim = render(new_trajectory,skip=1)
    anim.save('vessel2dcolor3dNew7.gif', writer='pillow')


