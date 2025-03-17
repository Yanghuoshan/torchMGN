import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d.art3d import Poly3DCollection,PolyCollection
from dataset_utils import datasets
import os
from model_utils.common import *
os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'


dl = datasets.get_dataloader_hdf5_batch("D:\project_summary\Graduation Project\\tmp\datasets_hdf5\\airway",model="IncompNS",shuffle=True, batch_size=1)
# dl = datasets.get_dataloader("D:\project_summary\Graduation Project\\tmp\datasets_np\\flag_simple\\train",dataset_type="Cloth")
dl = iter(dl)
input = next(dl)[0] 
for k,v in input.items():
    input[k] = input[k].squeeze(0).numpy()

print(max(input["mesh_pos"][:,0]),max(input["mesh_pos"][:,1]),min(input["mesh_pos"][:,0]),min(input["mesh_pos"][:,1]))

# 定义点坐标 
points = input["mesh_pos"]
velocity = np.sqrt(np.sum(input['velocity']**2,axis=1))
print(velocity.shape)
node_type = input["node_type"]
colormap = plt.cm.get_cmap('jet')  
node_colors = colormap(velocity / max(velocity))
print(points.shape)

# 定义四面体单元
triangles = input["cells"]
print(input["cells"].shape)

# 创建一个三维绘图对象
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')

# 绘制四面体
# for tet in triangles:
#     verts = [points[tet[i]] for i in range(4)]
#     faces = [
#         [verts[0], verts[1], verts[2]],
#         [verts[0], verts[1], verts[3]],
#         [verts[0], verts[2], verts[3]],
#         [verts[1], verts[2], verts[3]]
#     ]
#     poly3d = Poly3DCollection(faces, alpha=0.1, edgecolor='k')
#     ax.add_collection3d(poly3d)

# 绘制散点
scatter = ax.scatter(points[:, 0], points[:, 1], points[:, 2], c=velocity, cmap='jet')
# 添加colorbar
cbar = plt.colorbar(scatter, ax=ax, shrink=0.5, aspect=10)
cbar.set_label("Velocity Magnitude")  # 设置colorbar标签

ax.set_xlabel('X')
ax.set_ylabel('Y')
ax.set_zlabel('Z')




# 显示图形
plt.show()
