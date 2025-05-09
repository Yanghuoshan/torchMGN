# Guideline
## rollout_test*.py

测试模型输入输出

## train.py

模型训练，正在施工

由于模型中传递的输入数据为自定义的MultiGraph对象，所以无法支持DataParallel的多卡训练

## eval.py

模型测试，还在施工

## rollout_test_2.py

模型效果渲染

## datasets_utils

|Files     |Description|
|:-------- |:-----|
|tfrecord2npz.py  | 能够把tfrecord格式的文件转换成许多个npz格式的文件，其中每个文件存放了一个trajectory。|
|tfrecord2numpydict.py  | 已经废弃，因为要把整个tfrecord转换成单个numpydict太耗资源，时间久，不划算。|
|tfrecord2hdf5.py  | 能够把tfrecord格式的文件转换成单独的h5格式文件，加快IO速度。|
|datasets.py  | 能够创建给dataloader使用的dataset。dataset会先对每个trajectory的npz文件进行整理，检查一共能生成多少个样本。根据idx从dataset中取sample时，会根据每个trajectory包含的样本数，从相应的npz文件中取sample。例如：ex0有400个样本，ex1有400个样本，那么当 idx = 0 时，会取ex0的第0个sample，当 idx = 401 时，则会取ex1的第1和sample。以此类推。为以图作为输入的训练和渲染提供适配的数据集格式。由于批量数据集会将几个图合并在一起，world_pos会重叠在建立世界域的边时可能会把不同图的相近的点连在一起，因此需要对建图的函数进行修改。|
|PyGdatasets.py  | 能够创建给dataloader使用的PyG图dataset。已废弃。|



## model_utils_abandon
使用PyG的版本，发现PyG难以同时处理mesh_edge和world_edge放弃

## model_utils
主体基于meshgraphnets_cloth_cfd_deform_simulation(meshgraphnets)

|Files     |Description|
|:-------- |:-----|
|common.py|包含对于点类型的定义，基础图和边集dataclass的定义，将meshes转换成edges的算法,建图算法。|
|HyperEl.py|用于弹性材料的模型，其中规定了如何构造图数据，以及用于此模型的损失函数。|
|Cloth.py|用于布料的模型，其中规定了如何构造图数据，以及用于此模型的损失函数，支持用图或者是字典进行训练。|
|IncompNS.py|用于流固耦合的模型，采用了动态网格的欧拉系统，可选新的GraphNetBlockWithU模块|
|normalization.py|用于数据归一化的组件。|
|encode_process_decode.py|包含了主体的 encoder_processor_decoder 框架，以及graphnet的实现，sigmoid激活函数不收敛，relu收敛。 |
|Inflaction.py|基于当前时刻的速度和压力，预测下一时刻的速度和压力，基于当前时刻的world_pos和下一时刻的压力减去当前时刻的压力，即压力的变化率|

## render_utils

|Files     |Description|
|:-------- |:-----|
|Cloth_render.py|渲染布料运动，读取整个轨迹，输出动态图像。|

## run_utils

|Files     |Description|
|:-------- |:-----|
|utils.py|train.py, eval.py等文件用到的工具函数集，包括日志文件的保存读取等功能。|