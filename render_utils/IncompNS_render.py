from os import name
import pickle

from absl import app
from absl import flags

from matplotlib import animation
from matplotlib import cm
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d.art3d import Poly3DCollection,PolyCollection
import torch

# import vtk
import numpy as np


import torch
import matplotlib.pyplot as plt
from matplotlib.collections import PolyCollection
from matplotlib import cm
import numpy as np

import torch
import matplotlib.pyplot as plt
from matplotlib.collections import PolyCollection
from matplotlib import cm
import numpy as np

def render(trajectory, skip=1, color_field ='velocity'):  # 2d projection
    trajectory['mesh_pos'] = trajectory['mesh_pos'].to('cpu')
    trajectory['cells'] = trajectory['cells'].to('cpu')
    trajectory['pressure'] = trajectory['pressure'].to('cpu')
    trajectory['velocity'] = trajectory['velocity'].to('cpu')
    
    fig, ax = plt.subplots(figsize=(10, 8))
    skip = skip
    num_steps = trajectory['mesh_pos'].shape[0]  # total steps
    num_frames = num_steps // skip
    
    # compute the bound
    bb_min = torch.min(trajectory['mesh_pos'], dim=0).values
    bb_min = torch.min(bb_min, dim=0).values
    bb_max = torch.max(trajectory['mesh_pos'], dim=0).values
    bb_max = torch.max(bb_max, dim=0).values
    
    bound = (bb_min, bb_max)
    
    # Adjust aspect ratio to ensure equal units in both axes
    ax.set_aspect('equal', adjustable='datalim')
    
    if color_field == 'veocity':
        # Create a colormap and normalization object
        cmap = cm.viridis
        norm_velocity = torch.sqrt(torch.sum(trajectory['velocity']**2, dim=2))
        norm = plt.Normalize(vmin=norm_velocity.min().item(), vmax=norm_velocity.max().item())
        
        ## core function modified
        def animate(num):
            step = num * skip
            print(f"render step: {step}")
            ax.cla()
            ax.set_xlim([bound[0][0], bound[1][0]])
            ax.set_ylim([bound[0][1], bound[1][1]])
            
            pos = trajectory['mesh_pos'][step]
            triangles = trajectory['cells'][step]
            velocities = trajectory['velocity'][step]
            
            # Normalize the velocity for colormap
            norm_velocity_step = torch.sqrt(torch.sum(velocities**2, dim=1))
            colors = cmap(norm(norm_velocity_step.numpy()))
            
            # Plot the triangles with interpolated color
            for tri in triangles:
                verts = [pos[tri[i]].numpy() for i in range(3)]
                face = [[verts[0], verts[1], verts[2]]]
                poly2d = PolyCollection(face, alpha=1, edgecolor='k', facecolors=[colors[tri[i]] for i in range(3)])
                ax.add_collection(poly2d)
            
            ax.set_title('Step %d' % (step))
            return fig,
        ani = animation.FuncAnimation(fig, animate, frames=num_frames, interval=50)
    
        # Add colorbar
        sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
        sm.set_array([])
        fig.colorbar(sm, ax=ax, orientation='vertical', label='Velocity magnitude')
    
    elif color_field == 'pressure':
        # Create a colormap and normalization object
        cmap = cm.viridis
        norm = plt.Normalize(vmin=trajectory['pressure'].min().item(), vmax=trajectory['pressure'].max().item())
        
        ## core function modified
        def animate(num):
            step = num * skip
            print(f"render step: {step}")
            ax.cla()
            ax.set_xlim([bound[0][0], bound[1][0]])
            ax.set_ylim([bound[0][1], bound[1][1]])
            
            pos = trajectory['mesh_pos'][step]
            triangles = trajectory['cells'][step]
            pressure = trajectory['pressure'][step]
            
            # Normalize the velocity for colormap
            colors = cmap(norm(pressure.numpy()))
            
            # Plot the triangles with interpolated color
            for tri in triangles:
                verts = [pos[tri[i]].numpy() for i in range(3)]
                face = [[verts[0], verts[1], verts[2]]]
                poly2d = PolyCollection(face, alpha=1, edgecolor='k', facecolors=[colors[tri[i]] for i in range(3)])
                ax.add_collection(poly2d)
            
            ax.set_title('Step %d' % (step))
            return fig,
        ani = animation.FuncAnimation(fig, animate, frames=num_frames, interval=50)
    
        # Add colorbar
        sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
        sm.set_array([])
        fig.colorbar(sm, ax=ax, orientation='vertical', label='pressure')
    
    else:
        raise ValueError('use velocity or pressure option')


    
    
    # plt.show()
    return ani



# 示例数据
if __name__ == '__main__':
    trajectory = {
        'world_pos': torch.rand((10, 8, 3)),  # 10 steps, 8 points each, 3D coordinates
        'cells': torch.randint(0, 8, (10, 4, 4))  # 10 steps, 4 tetrahedrons each
    }

    render(trajectory, skip=1)
