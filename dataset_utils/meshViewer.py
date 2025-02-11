import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d.art3d import Poly3DCollection,PolyCollection
from dataset_utils import datasets
import os
os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'


dl = datasets.get_dataloader_hdf5_batch("D:\project_summary\Graduation Project\\tmp\datasets_hdf5\\flag_simple",model="Cloth",shuffle=False)
# dl = datasets.get_dataloader("D:\project_summary\Graduation Project\\tmp\datasets_np\\flag_simple\\train",dataset_type="Cloth")
dl = iter(dl)
input = next(dl)[0] 
for k,v in input.items():
    input[k] = input[k].squeeze(0).numpy()

print(max(input["mesh_pos"][:,0]),max(input["mesh_pos"][:,1]),min(input["mesh_pos"][:,0]),min(input["mesh_pos"][:,1]))

# 定义点坐标
points = input["world_pos"]
print(points.shape)
# print(input["world_pos"])
# print(input["mesh_pos"])

# 定义面
# triangles = input["triangles"]
# rectangles = input["rectangles"]
# print(rectangles)
# 创建一个绘图对象
fig = plt.figure()
ax = fig.add_subplot(111)

# 绘制点
# ax.scatter(points[:, 0], points[:, 1])

# 绘制四面体
for tet in triangles:
    verts = [points[tet[i]] for i in range(3)]
    faces = [[verts[0], verts[1], verts[2]]]
    poly2d = PolyCollection(faces, alpha=0.2, edgecolor='k')
    ax.add_collection(poly2d)

# 设置坐标轴
ax.set_xlabel('X')
ax.set_ylabel('Y')

ax.set_xlim([-0.1, 4])
ax.set_ylim([-5, 5])


# 显示图形
plt.show()
