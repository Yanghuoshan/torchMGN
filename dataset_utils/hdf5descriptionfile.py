import h5py
import json
from matplotlib import pyplot as plt

# 打开 HDF5 文件
file_path = 'D:\project_summary\Graduation Project\\tmp\datasets_hdf5\\symmetry_waterballoon\\train\\dataset.h5'
dict1 = {"files":dict()}
with h5py.File(file_path, 'r') as f:
    # 列出文件中的所有数据集
    print("Datasets in the file:")
    for name in f:
        print(name, end=' ')
        print(f[name]["world_pos"].shape[0])
        dict1["files"][name]=f[name]["world_pos"].shape[0]
    string = json.dumps(dict1)
    with open('D:\project_summary\Graduation Project\\tmp\datasets_hdf5\\symmetry_waterballoon\\train\\metadata.json','w',encoding='utf-8') as f:
        f.write(string)