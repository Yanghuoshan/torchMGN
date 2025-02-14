from os import name
import pickle

from absl import app
from absl import flags

from matplotlib import animation
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d.art3d import Poly3DCollection,PolyCollection
import torch

# import vtk
import numpy as np


def render(trajectory, skip=1):# 2d projection
    trajectory['world_pos'] = trajectory['world_pos'].to('cpu')
    trajectory['triangles'] = trajectory['triangles'].to('cpu')
    trajectory['rectangles'] = trajectory['rectangles'].to('cpu')
    fig = plt.figure(figsize=(8, 8))
    ax = fig.add_subplot(111)
    skip = skip
    num_steps = trajectory['world_pos'].shape[0] # total steps
    # num_frames = len(rollout_data) * num_steps // skip

    num_frames = num_steps // skip

  # compute the bound
    
    bb_min = torch.min(trajectory['world_pos'],dim=0).values
    bb_min = torch.min(bb_min,dim=0).values
    bb_max = torch.max(trajectory['world_pos'],dim=0).values
    bb_max = torch.max(bb_max,dim=0).values
    
    bound = (bb_min, bb_max)

    ## core function modified
    def animate(num):
        step = num * skip
        ax.cla()
        ax.set_xlim([bound[0][0], bound[1][0]])
        ax.set_ylim([bound[0][1], bound[1][1]])
        
        pos = trajectory['world_pos'][step]
        rectangles = trajectory['rectangles'][step]
        triangles = trajectory['triangles'][step]
        
        for quad in rectangles:
            verts = [pos[quad[i]].numpy() for i in range(4)]
            face = [[verts[0], verts[1], verts[2], verts[3]]]
            poly2d = PolyCollection(face, alpha=0.2, edgecolor='k')
            ax.add_collection(poly2d)

        for tri in triangles:
            verts = [pos[tri[i]].numpy() for i in range(3)]
            face = [[verts[0], verts[1], verts[2]]]
            poly2d = PolyCollection(face, alpha=0.2, edgecolor='k')
            ax.add_collection(poly2d)
        
        ax.set_title('Step %d' % (step))
        return fig,



    anim = animation.FuncAnimation(fig, animate, frames=num_frames, interval=100)
    # plt.show(block=True)
    return anim



