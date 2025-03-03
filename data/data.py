from datasets import load_dataset

# 加载 LaDe-D 数据集
ds = load_dataset("Cainiao-AI/LaDe-D")

# 选择重庆的 delivery 数据
data_cq = ds['delivery_cq']

# 打印数据的前几行
print(data_cq[0])

#
# # 打印数据集的键
# print(ds.keys())
#
# # 打印数据集的样本结构
# print(ds)
