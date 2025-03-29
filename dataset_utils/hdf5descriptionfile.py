import h5py
import json
from matplotlib import pyplot as plt

# 打开 HDF5 文件
file_path = 'D:\project_summary\Graduation Project\\tmp\datasets_hdf5\\new_airway\\train\\'

dict1 = {"files":dict()}
with h5py.File(file_path+'dataset.h5', 'r') as f:
    # 列出文件中的所有数据集
    print("Datasets in the file:")
    flag = 0
    for name in f:
        flag = flag + 1
        print(name, end=' ')
        print(f[name]["mesh_pos"].shape[0])
        dict1["files"][name]=f[name]["mesh_pos"].shape[0]
    string = json.dumps(dict1)
    with open(file_path+'metadata.json','w',encoding='utf-8') as f:
        f.write(string)
    print(flag)