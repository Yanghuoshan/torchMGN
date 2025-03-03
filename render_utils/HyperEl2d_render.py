import torch
import matplotlib.pyplot as plt
import matplotlib.animation as animation

def render(trajectory, skip=1):
    trajectory['world_pos'] = trajectory['world_pos'].to('cpu')
    trajectory['cells'] = trajectory['cells'].to('cpu')
    
    fig, ax = plt.subplots(figsize=(8, 8))
    num_steps = trajectory['world_pos'].shape[0]  # total steps
    num_frames = num_steps // skip
    
    # Compute bounding box
    bb_min = torch.min(trajectory['world_pos'], dim=0).values
    bb_min = torch.min(bb_min, dim=0).values
    bb_max = torch.max(trajectory['world_pos'], dim=0).values
    bb_max = torch.max(bb_max, dim=0).values
    bound = (bb_min, bb_max)
    
    def animate(num):
        step = num * skip
        print(f'render step: {step}')
        ax.clear()
        ax.set_xlim([bound[0][0], bound[1][0]])
        ax.set_ylim([bound[0][1], bound[1][1]])
        pos = trajectory['world_pos'][step]
        faces = trajectory['cells'][step]
        
        for face in faces:
            vertices = pos[face].numpy()
            polygon = plt.Polygon(vertices, edgecolor='black', facecolor='gray', alpha=0.5)
            ax.add_patch(polygon)
        
        ax.set_title(f'Step {step}')
        return fig,
    
    anim = animation.FuncAnimation(fig, animate, frames=num_frames, interval=100)
    return anim
