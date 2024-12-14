# Guideline
## rollout_test.py

测试模型输入输出

## train.py

模型训练，正在施工

## datasets_utils

|Files     |Description|
|:-------- |:-----|
|tfrecord2npz.py  | 能够把tfrecord格式的文件转换成许多个npz格式的文件，其中每个文件存放了一个trajectory。|
|tfrecord2numpydict.py  | 已经废弃，因为要把整个tfrecord转换成单个numpydict太耗资源，时间久，不划算。|
|datasets.py  | 能够创建给dataloader使用的dataset。dataset会先对每个trajectory的npz文件进行整理，检查一共能生成多少个样本。根据idx从dataset中取sample时，会根据每个trajectory包含的样本数，从相应的npz文件中取sample。例如：ex0有400个样本，ex1有400个样本，那么当 idx = 0 时，会取ex0的第0个sample，当 idx = 401 时，则会取ex1的第1和sample。以此类推。|
|PyGdatasets.py  | 能够创建给dataloader使用的PyG图dataset。已废弃|


## model_utils_abandon
使用PyG的版本，发现PyG难以同时处理mesh_edge和world_edge放弃

## model_utils
主体基于meshgraphnets_cloth_cfd_deform_simulation(meshgraphnets)

|Files     |Description|
|:-------- |:-----|
|common.py|包含对于点类型的规定，和将meshes转换成edges的算法|
|HyperEl_model.py|用于弹性材料的模型，其中规定了如何构造图数据，以及用于此模型的损失函数|
|normalization.py|用于数据归一化的组件|
|encode_process_decode.py|包含了主体的 encoder_processor_decoder 框架，以及graphnet的实现 |
