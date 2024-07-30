import pandas as pd
import re

# 从文件中读取日志数据
with open('data/time.log', 'r') as file:
    log_data = file.read()

# print(log_data)

# 正则表达式模式，用于匹配日志中的每条记录
record_pattern = re.compile(
    r"dataset: (\w+).*?"
    r"num_trees:(\d+).*?"
    r"data_size:(\d+).*?"
    r"depth:(\d+).*?"
    r"(epsilon: (\d+).*?"
    r"Monitor: DO.*?"
    r"AddDummy: (\d+\.?\d*s).*?"
    r"PostProcess: (\d+\.?\d*s).*?"
    r"PredictDMatrixDO: (\d+\.?\d*s).*?"
    r"PredictNO: (\d+\.?\d*s).*?"
    r"shuffle: (\d+\.?\d*s).*?)?"
    r"algorithm: (\w+).*?"
    r"PredictBatch: (\d+\.\d+)", re.DOTALL
)

# 提取记录并填入表格
records = []
for match in record_pattern.finditer(log_data):
    groups = match.groups()
    print(groups)
    dataset, num_trees, data_size, depth, _, epsilon, add_dummy, post_process, predict_dmatrix, predict_no, shuffle, algorithm, predict_batch = groups
    record = {
        'dataset': dataset,
        'num_trees': int(num_trees),
        'data_size': int(data_size),
        'depth': int(depth),
        'epsilon': int(epsilon) if epsilon else None,
        'AddDummy': float(add_dummy[:-1]) if add_dummy else None,
        'PostProcess': float(post_process[:-1]) if post_process else None,
        'PredictDMatrixDO': float(predict_dmatrix[:-1]) if predict_dmatrix else None,
        'PredictNO': float(predict_no[:-1]) if predict_no else None,
        'shuffle': float(shuffle[:-1]) if shuffle else None,
        'algorithm': algorithm,
        'PredictBatch': float(predict_batch)
    }
    records.append(record)
print(records)

# 构建 DataFrame
df = pd.DataFrame(records)

# 保留最新记录（参数完全相同时）
df = df.drop_duplicates(subset=['dataset', 'num_trees', 'data_size', 'depth', 'epsilon', 'algorithm'], keep='last')

# 将结果保存到 CSV 文件
df.to_csv('output_log_records.csv', index=False)
