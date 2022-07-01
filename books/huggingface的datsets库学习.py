from pprint import pprint
from datasets import list_datasets, load_dataset
from datasets import list_metrics, load_metric

datasets = list_datasets()        # 所有的数据集
print(len(datasets))  # 6189

dataset = load_dataset('sst.py', split='train')  # 可以自己取网上把数据处理文件 'sst.py' 下载下来
dataset = load_dataset('sst.py', split='validation')
print(len(dataset))

pprint(dataset[0])

metrics = list_metrics()   # 所有的评价指标
print(', '.join(metrics))
print(len(metrics))

accuracy_metric = load_metric('accuracy.py')
results = accuracy_metric.compute(references=[0, 1, 0], predictions=[1, 1, 0])  # 计算acc
print(results)
