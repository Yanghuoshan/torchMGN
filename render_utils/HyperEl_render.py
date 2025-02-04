from os import name
import pickle

from absl import app
from absl import flags

from matplotlib import animation
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
import torch

# import vtk
import numpy as np

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

def render(trajectory, skip=1):
    trajectory['world_pos'] = trajectory['world_pos'].to('cpu')
    trajectory['cells'] = trajectory['cells'].to('cpu')
    fig = plt.figure(figsize=(8, 8))
    ax = fig.add_subplot(111, projection='3d')
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

    # def animate(num):
    #     step = num*skip
    #     ax.cla()
    #     ax.set_xlim([bound[0][0], bound[1][0]])
    #     ax.set_ylim([bound[0][1], bound[1][1]])
    #     ax.set_zlim([bound[0][2], bound[1][2]])
    #     pos = trajectory['world_pos'][step]
    #     faces = trajectory['cells'][step]
    #     ax.plot_trisurf(pos[:, 0], pos[:, 1], faces, pos[:, 2], shade=True)
    #     ax.set_title('Step %d' % (step))
    #     return fig,

    def animate(num):
        step = num * skip
        ax.cla()
        ax.set_xlim([bound[0][0], bound[1][0]])
        ax.set_ylim([bound[0][1], bound[1][1]])
        ax.set_zlim([bound[0][2], bound[1][2]])
        pos = trajectory['world_pos'][step]
        faces = trajectory['cells'][step]
        # for face in faces:
        #     vertices = pos[face]
        #     ax.plot_trisurf(vertices[:, 0], vertices[:, 1], vertices[:, 2], shade=True)
        for tet in faces:
            verts = [pos[tet[i]].numpy() for i in range(4)]
            faces = [[verts[0], verts[1], verts[2]],
            [verts[0], verts[1], verts[3]],
            [verts[0], verts[2], verts[3]],
            [verts[1], verts[2], verts[3]]]
            poly3d = Poly3DCollection(faces, alpha=0.2, edgecolor='k')
            ax.add_collection3d(poly3d)
        ax.set_title('Step %d' % (step))
        return fig,


    anim = animation.FuncAnimation(fig, animate, frames=num_frames, interval=100)
    # plt.show(block=True)
    return anim

def render2(trajectory, skip=1):
    trajectory['world_pos'] = trajectory['world_pos'].to('cpu').numpy()
    trajectory['cells'] = trajectory['cells'].to('cpu').numpy()
    
    num_steps = trajectory['world_pos'].shape[0]
    num_frames = num_steps // skip
    
    # 计算边界
    bb_min = np.min(trajectory['world_pos'], axis=(0, 1))
    bb_max = np.max(trajectory['world_pos'], axis=(0, 1))
    bound = (bb_min, bb_max)

    # 创建渲染器、渲染窗口和交互器
    renderer = vtk.vtkRenderer()
    renderWindow = vtk.vtkRenderWindow()
    renderWindow.AddRenderer(renderer)
    renderWindowInteractor = vtk.vtkRenderWindowInteractor()
    renderWindowInteractor.SetRenderWindow(renderWindow)
    
    # 创建 PolyData 对象
    poly_data = vtk.vtkPolyData()
    points = vtk.vtkPoints()
    cells = vtk.vtkCellArray()
    tetra = vtk.vtkTetra()

    def animate(num):
        step = num * skip
        points.Reset()
        cells.Reset()
        for pos in trajectory['world_pos'][step]:
            points.InsertNextPoint(pos)

        for tet in trajectory['cells'][step]:
            tetra.GetPointIds().SetId(0, tet[0])
            tetra.GetPointIds().SetId(1, tet[1])
            tetra.GetPointIds().SetId(2, tet[2])
            tetra.GetPointIds().SetId(3, tet[3])
            cells.InsertNextCell(tetra)

        poly_data.SetPoints(points)
        poly_data.SetPolys(cells)
        
        # 创建映射器和演员
        mapper = vtk.vtkPolyDataMapper()
        mapper.SetInputData(poly_data)
        actor = vtk.vtkActor()
        actor.SetMapper(mapper)
        
        renderer.AddActor(actor)
        renderer.SetBackground(0.1, 0.2, 0.4)  # 设置背景颜色

        renderWindow.Render()

    # 动画循环
    for num in range(num_frames):
        animate(num)
    
    renderWindowInteractor.Initialize()
    renderWindowInteractor.Start()

    return

# 示例数据
if __name__ == '__main__':
    trajectory = {
        'world_pos': torch.rand((10, 8, 3)),  # 10 steps, 8 points each, 3D coordinates
        'cells': torch.randint(0, 8, (10, 4, 4))  # 10 steps, 4 tetrahedrons each
    }

    render(trajectory, skip=1)
