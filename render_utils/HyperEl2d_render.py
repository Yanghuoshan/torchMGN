import torch
import matplotlib.pyplot as plt
import matplotlib.tri as tri
import matplotlib.animation as animation
import numpy as np

def render(trajectory, skip=1):
    trajectory['world_pos'] = trajectory['world_pos'].to('cpu')
    trajectory['cells'] = trajectory['cells'].to('cpu')
    trajectory['stress'] = trajectory['stress'].to('cpu')
    trajectory['node_type'] = trajectory['node_type'].to('cpu')

    fig, ax = plt.subplots(figsize=(8, 8))
    num_steps = trajectory['world_pos'].shape[0]  # total steps
    num_frames = num_steps // skip

    # Compute bounding box
    bb_min = torch.min(trajectory['world_pos'], dim=0).values
    bb_min = torch.min(bb_min, dim=0).values
    bb_max = torch.max(trajectory['world_pos'], dim=0).values
    bb_max = torch.max(bb_max, dim=0).values
    bound = (bb_min, bb_max)

    # 处理全局 stress 和 node_type（假设 node_type 在所有时间步都相同）
    stress_all = trajectory['stress'].numpy()  # shape: [time_step, num_node]
    node_type_all = trajectory['node_type'].numpy().squeeze()  # shape: [num_node]

    # 过滤掉 node_type 为 3 的 stress（保留 node_type==1 用于后续灰色绘制）
    stress_filtered_all = np.where(node_type_all == 3, 0, stress_all)
    # 对 stress 进行对数放缩（使用 log1p 避免 log(0) 的问题）
    stress_filtered_all_log = np.log1p(stress_filtered_all)
    stress_min, stress_max = np.min(stress_filtered_all_log), np.max(stress_filtered_all_log)

    ax.set_aspect('equal', adjustable='datalim')
    
    from matplotlib.collections import PolyCollection

    def animate(num):
        step = num * skip
        print(f'render step: {step}')
        ax.clear()
        ax.set_xlim([bound[0][0], bound[1][0]])
        ax.set_ylim([bound[0][1], bound[1][1]])

        pos = trajectory['world_pos'][step].numpy()      # shape: [num_nodes, 2]
        faces = trajectory['cells'][step].numpy()          # shape: [num_triangles, 3]
        stress = trajectory['stress'][step].numpy().flatten()  # shape: [num_nodes]

        # 使用全局 node_type_all
        node_type = node_type_all

        # 对于 stress，只对 node_type==3 进行置零，保留 node_type==1 后续覆盖为灰色
        stress_for_tripcolor = np.where(node_type == 3, 0, stress)
        # 对数放缩
        stress_for_tripcolor = np.log1p(stress_for_tripcolor)

        # 创建三角形插值对象
        triang = tri.Triangulation(pos[:, 0], pos[:, 1], faces)

        # 绘制插值热力图
        tpc = ax.tripcolor(triang, stress_for_tripcolor, cmap='coolwarm',
                           shading='gouraud', vmin=stress_min, vmax=stress_max)

        # 找出所有顶点均为 node_type==1 的三角形，并覆盖为较浅的灰色
        mask_type1 = np.all(node_type[faces] == 1, axis=1)
        if np.any(mask_type1):
            triangles_coords = pos[faces[mask_type1]]  # shape: [n_gray, 3, 2]
            # 这里依然可以使用 PolyCollection 绘制覆盖区域
            gray_poly = PolyCollection(triangles_coords, facecolors='lightgray', 
                                       edgecolors='k', linewidth=0.1, alpha=1)
            ax.add_collection(gray_poly)

        ax.set_title(f'Step {step}')
        return tpc,

    anim = animation.FuncAnimation(fig, animate, frames=num_frames, interval=100)
    return anim


from mpl_toolkits.mplot3d.art3d import Poly3DCollection


def render3d(trajectory, skip=1, start_angle=90, angle=240, num_slices=30):
    # 将各 tensor 转换到 cpu 上
    trajectory['world_pos'] = trajectory['world_pos'].to('cpu')
    trajectory['cells'] = trajectory['cells'].to('cpu')
    trajectory['stress'] = trajectory['stress'].to('cpu')
    trajectory['node_type'] = trajectory['node_type'].to('cpu')

    # 计算全局应力范围（对 stress 进行对数放缩）
    stress_all = trajectory['stress'].numpy()  # shape: [time_step, num_node]
    node_type_all = trajectory['node_type'].numpy().squeeze()  # shape: [num_node]
    # 对于 node_type 为 1 或 3 的节点，将 stress 设为 0（log1p(0)=0）
    stress_filtered_all = np.where((node_type_all == 1) | (node_type_all == 3), 0, stress_all)
    stress_filtered_all_log = np.log1p(stress_filtered_all)
    stress_min, stress_max = np.min(stress_filtered_all_log), np.max(stress_filtered_all_log)

    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection='3d')
    num_steps = trajectory['world_pos'].shape[0]  # 总步数
    num_frames = num_steps // skip

    pos_all = trajectory['world_pos'].numpy()
    x_min, x_max = np.min(pos_all[:, :, 0]), np.max(pos_all[:, :, 0])
    z_min, z_max = np.min(pos_all[:, :, 1]), np.max(pos_all[:, :, 1])
    max_radius = max(abs(x_min), abs(x_max), abs(z_min), abs(z_max))  # 让Z轴不会过短

    # 预设坐标轴范围，确保三轴比例一致
    ax.set_xlim3d([-max_radius, max_radius])
    ax.set_ylim3d([-max_radius, max_radius])
    ax.set_zlim3d([-max_radius, max_radius])
    ax.set_box_aspect([1, 1, 1])  # 确保三轴比例相同

    # 预计算旋转角度
    start_angle_rad = np.radians(start_angle)
    angle_rad = np.radians(angle)
    theta_vals = np.linspace(start_angle_rad, start_angle_rad + angle_rad, num_slices)
    cos_t = np.cos(theta_vals)
    sin_t = np.sin(theta_vals)

    def animate(num):
        step = num * skip
        ax.clear()
        
        # 设定坐标轴范围
        ax.set_xlim3d([-max_radius, max_radius])
        ax.set_ylim3d([-max_radius, max_radius])
        ax.set_zlim3d([-max_radius, max_radius])
        ax.view_init(elev=25, azim=30)

        # 读取当前时间步的数据
        world_pos = trajectory['world_pos'][step].numpy()
        stress = trajectory['stress'][step].numpy().flatten()  # shape: [num_nodes]
        cells = trajectory['cells'][step].numpy()
        num_nodes = world_pos.shape[0]

        # 提取 x 和 z
        x = world_pos[:, 0]
        z = world_pos[:, 1]

        # 计算旋转后的坐标
        X = np.outer(cos_t, x)
        Y = np.outer(sin_t, x)
        Z = np.tile(z, (num_slices, 1))
        vertices = np.stack([X, Y, Z], axis=2).reshape(-1, 3)

        # 对当前帧的 stress 进行对数放缩
        stress_log = np.log1p(stress)
        # 计算每个顶点的 stress（复制 num_slices 次）
        vertex_stress = np.tile(stress_log, (num_slices, 1)).flatten()
        vertex_colors = plt.cm.coolwarm((vertex_stress - stress_min) / (stress_max - stress_min))

        # 生成面片索引
        faces_list = []
        for i in range(num_slices - 1):
            offset1 = i * num_nodes
            offset2 = (i + 1) * num_nodes
            face1 = cells + offset1
            face2 = np.column_stack((cells[:, 0] + offset1, cells[:, 2] + offset1, cells[:, 0] + offset2))
            face3 = np.column_stack((cells[:, 0] + offset2, cells[:, 2] + offset1, cells[:, 2] + offset2))
            face4 = np.column_stack((cells[:, 0] + offset2, cells[:, 2] + offset2, cells[:, 1] + offset2))
            face5 = np.column_stack((cells[:, 0] + offset2, cells[:, 1] + offset2, cells[:, 1] + offset1))
            face6 = np.column_stack((cells[:, 1] + offset1, cells[:, 1] + offset2, cells[:, 2] + offset1))
            faces_layer = np.vstack((face1, face2, face3, face4, face5, face6))
            faces_list.append(faces_layer)
        faces = np.vstack(faces_list)
        faces = np.unique(faces, axis=0)

        # 判断每个面片是否包含 node_type 为 1 的节点
        face_nodes = faces % num_nodes  # 转换为原始节点索引
        node_types = node_type_all[face_nodes]
        has_type1_mask = np.any(node_types == 1, axis=1)

        # 计算面片颜色
        original_face_colors = vertex_colors[faces].mean(axis=1)
        grey_color = np.array([0.8, 0.8, 0.8, 0.85])    # RGBA 灰色
        face_colors = original_face_colors.copy()
        face_colors[has_type1_mask] = grey_color

        # 创建面片集合
        mesh = Poly3DCollection(
            [vertices[face] for face in faces], 
            facecolors=face_colors,
            edgecolor='k', 
            linewidths=0.1,
            alpha=0.85
        )
        ax.add_collection3d(mesh)

        ax.set_xlabel("X")
        ax.set_ylabel("Y")
        ax.set_zlabel("Z")
        ax.set_title(f'Step {step}')

    anim = animation.FuncAnimation(fig, animate, frames=num_frames, interval=100)
    return anim
