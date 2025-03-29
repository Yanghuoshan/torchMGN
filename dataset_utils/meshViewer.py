import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d.art3d import Poly3DCollection,PolyCollection
from dataset_utils import datasets
import os
os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'


dl = datasets.get_dataloader_hdf5_batch("D:\project_summary\Graduation Project\\tmp\datasets_hdf5\\new_airway",model="IncompNS",shuffle=True, batch_size=1)
# dl = datasets.get_dataloader("D:\project_summary\Graduation Project\\tmp\datasets_np\\flag_simple\\train",dataset_type="Cloth")
dl = iter(dl)
input = next(dl)[0] 
for k,v in input.items():
    input[k] = input[k].squeeze(0).numpy()

print(max(input["mesh_pos"][:,0]),max(input["mesh_pos"][:,1]),min(input["mesh_pos"][:,0]),min(input["mesh_pos"][:,1]))

# 定义点坐标 
points = input["mesh_pos"]
print(points.shape)
node_type = input["node_type"]
colormap = plt.cm.get_cmap("tab10", 9)  # 使用一个有九种颜色的colormap
node_colors = colormap(node_type / max(node_type))
print(points.shape)

# 定义四面体单元
triangles = input["cells"]
print(input["cells"].shape)

# 创建一个三维绘图对象
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')

# 绘制四面体
for tet in triangles:
    verts = [points[tet[i]] for i in range(4)]
    faces = [
        [verts[0], verts[1], verts[2]],
        [verts[0], verts[1], verts[3]],
        [verts[0], verts[2], verts[3]],
        [verts[1], verts[2], verts[3]]
    ]
    poly3d = Poly3DCollection(faces, alpha=1, edgecolor='k')
    ax.add_collection3d(poly3d)

# 绘制散点
scatter = ax.scatter(points[:, 0], points[:, 1], points[:, 2], c=node_colors)
cbar = fig.colorbar(scatter, ax=ax, ticks=np.arange(9))
ax.set_xlabel('X')
ax.set_ylabel('Y')
ax.set_zlabel('Z')



# 显示图形
plt.show()
