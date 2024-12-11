# Workflow
## datasets_utils
### tfrecord2npz.py
能够把tfrecord格式的文件转换成许多个npz格式的文件，其中每个文件存放了一个trajectory。

---

### tfrecord2numpydict.py
已经废弃，因为要把整个tfrecord转换成单个numpydict太耗资源，时间久，不划算。

---

### datasets.py
能够创建给dataloader使用的dataset。

dataset会先对每个trajectory的npz文件进行整理，检查一共能生成多少个样本。根据idx从dataset中取sample时，会根据每个trajectory包含的样本数，从相应的npz文件中取sample。

例如：

ex0有400个样本，ex1有400个样本，那么当 idx = 0 时，会取ex0的第0个sample，当 idx = 401 时，则会取ex1的第1和sample。以此类推。

### PyGdatasets.py
能够创建给dataloader使用的PyG图dataset。已废弃

### model_utils_abandon
使用PyG的版本，发现PyG难以同时处理mesh_edge和world_edge放弃