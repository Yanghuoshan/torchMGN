import pickle

from absl import app
from absl import flags

from matplotlib import animation
import matplotlib.pyplot as plt

import torch

# FLAGS = flags.FLAGS
# flags.DEFINE_string('rollout_path', None, 'Path to rollout pickle file')


# def main(unused_argv):
#   with open(FLAGS.rollout_path, 'rb') as fp:
#     rollout_data = pickle.load(fp)

#   fig = plt.figure(figsize=(8, 8))
#   ax = fig.add_subplot(111, projection='3d')
#   skip = 10
#   num_steps = rollout_data[0]['gt_pos'].shape[0] # total steps
#   num_frames = len(rollout_data) * num_steps // skip

#   # compute bounds
#   bounds = []
#   for trajectory in rollout_data:
#     bb_min = trajectory['gt_pos'].min(axis=(0, 1))
#     bb_max = trajectory['gt_pos'].max(axis=(0, 1))
#     bounds.append((bb_min, bb_max))

#   def animate(num):
#     step = (num*skip) % num_steps
#     traj = (num*skip) // num_steps
#     ax.cla()
#     bound = bounds[traj]
#     ax.set_xlim([bound[0][0], bound[1][0]])
#     ax.set_ylim([bound[0][1], bound[1][1]])
#     ax.set_zlim([bound[0][2], bound[1][2]])
#     pos = rollout_data[traj]['pred_pos'][step]
#     faces = rollout_data[traj]['faces'][step]
#     ax.plot_trisurf(pos[:, 0], pos[:, 1], faces, pos[:, 2], shade=True)
#     ax.set_title('Trajectory %d Step %d' % (traj, step))
#     return fig,

#   _ = animation.FuncAnimation(fig, animate, frames=num_frames, interval=100)
#   plt.show(block=True)


# if __name__ == '__main__':
#   app.run(main)

def render(trajectory):
    trajectory['world_pos'] = trajectory['world_pos'].to('cpu')
    trajectory['cells'] = trajectory['cells'].to('cpu')
    fig = plt.figure(figsize=(8, 8))
    ax = fig.add_subplot(111, projection='3d')
    skip = 1
    num_steps = trajectory['world_pos'].shape[0] # total steps
    # num_frames = len(rollout_data) * num_steps // skip

    num_frames = num_steps // skip

  # compute the bound
    
    bb_min = torch.min(trajectory['world_pos'],dim=0).values
    bb_min = torch.min(bb_min,dim=0).values
    bb_max = torch.max(trajectory['world_pos'],dim=0).values
    bb_max = torch.max(bb_max,dim=0).values
    
    bound = (bb_min, bb_max)

    def animate(num):
        step = num*skip
        ax.cla()
        ax.set_xlim([bound[0][0], bound[1][0]])
        ax.set_ylim([bound[0][1], bound[1][1]])
        ax.set_zlim([bound[0][2], bound[1][2]])
        pos = trajectory['world_pos'][step]
        faces = trajectory['cells'][step]
        ax.plot_trisurf(pos[:, 0], pos[:, 1], faces, pos[:, 2], shade=True)
        ax.set_title('Step %d' % (step))
        return fig,

    anim = animation.FuncAnimation(fig, animate, frames=num_frames, interval=100)
    plt.show(block=True)