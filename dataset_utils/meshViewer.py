import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
from dataset_utils import datasets
import os
os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'


dl = datasets.get_dataloader("D:\project_summary\Graduation Project\\tmp\datasets_np\\deforming_plate",model="HyperEl",shuffle=False)
# dl = datasets.get_dataloader("D:\project_summary\Graduation Project\\tmp\datasets_np\\flag_simple\\train",dataset_type="Cloth")
dl = iter(dl)
input = next(dl)[0] 
for k,v in input.items():
    input[k] = input[k].squeeze(0).numpy()

print(max(input["mesh_pos"][:,0]),max(input["mesh_pos"][:,1]),min(input["mesh_pos"][:,0]),min(input["mesh_pos"][:,1]))

# 定义点坐标
points = input["world_pos"]

# 定义面
mesh = input["cells"]

# 创建一个绘图对象
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')

# 绘制点
# ax.scatter(points[:, 0], points[:, 1], points[:, 2])

# 绘制四面体
for tet in mesh:
    verts = [points[tet[i]] for i in range(4)]
    faces = [[verts[0], verts[1], verts[2]],
    [verts[0], verts[1], verts[3]],
    [verts[0], verts[2], verts[3]],
    [verts[1], verts[2], verts[3]]]
    poly3d = Poly3DCollection(faces, alpha=0.2, edgecolor='k')
    ax.add_collection3d(poly3d)

# 设置坐标轴
ax.set_xlabel('X')
ax.set_ylabel('Y')
ax.set_zlabel('Z')
ax.set_xlim([-0.1, 0.7])
ax.set_ylim([-0.1, 0.7])
ax.set_zlim([-0.1, 0.7])

# 显示图形
plt.show()
