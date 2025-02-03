import pandas as pd
import re

# 从文件中读取日志数据
with open('data/time.log', 'r') as file:
    log_data = file.read()

# 使用分隔符将日志文件分割成若干个字符串
log_entries = log_data.split('+'*50)

# 定义正则表达式模式，用于匹配每条记录
detailed_pattern = re.compile(
    r"dataset: (\w+).*?"
    r"num_trees:(\d+).*?"
    r"data_size:(\d+).*?"
    r"depth:(\d+).*?"
    r"epsilon: (\d+\.?\d*).*?"
    r"shuffleMethod: (\w+).*?Monitor: DO.*?"
    r"AddDummy: (\d+\.?\d*s).*?"
    r"PostProcess: (\d+\.?\d*s).*?"
    r"PredictDMatrixDO: (\d+\.?\d*s).*?"
    r"PredictNO: (\d+\.?\d*s).*?"
    r"shuffle: (\d+\.?\d*s).*?"
    r"algorithm: (\w+).*?"
    r"PredictBatch: (\d+\.?\d*).*?"
    r"time_response:(\d+\.?\d*)", re.DOTALL
)

simple_pattern = re.compile(
    r"dataset: (\w+).*?"
    r"num_trees:(\d+).*?"
    r"data_size:(\d+).*?"
    r"depth:(\d+).*?"
    r"algorithm: \s?(\w+).*?"
    r"PredictBatch: (\d+\.?\d*).*?"
    r"time_response:(\d+\.?\d*)", re.DOTALL
)

# 提取记录并填入表格
records = []
for entry in log_entries:
    entry = entry.strip()
    if not entry:
        continue
    
    match = detailed_pattern.search(entry)
    if match:
        # print('detailed pattern match')
        groups = match.groups()
        dataset, num_trees, data_size, depth, epsilon, shuffleMethod, add_dummy, post_process, predict_dmatrix, predict_no, shuffle, algorithm, predict_batch, time_response = groups
        record = {
            'dataset': dataset,
            'num_trees': int(num_trees),
            'data_size': int(data_size),
            'depth': int(depth),
            'epsilon': float(epsilon) if epsilon else None,
            'shuffleMethod': shuffleMethod,
            'AddDummy': float(add_dummy[:-1]) if add_dummy else None,
            'PostProcess': float(post_process[:-1]) if post_process else None,
            'PredictDMatrixDO': float(predict_dmatrix[:-1]) if predict_dmatrix else None,
            'PredictNO': float(predict_no[:-1]) if predict_no else None,
            'shuffle': float(shuffle[:-1]) if shuffle else None,
            'algorithm': algorithm,
            'PredictBatch': float(predict_batch),
            'time_response': float(time_response)
        }
    else:
        match = simple_pattern.search(entry)
        if match:
            # print('simple pattern match')
            groups = match.groups()
            dataset, num_trees, data_size, depth, algorithm, predict_batch, time_response = groups
            record = {
                'dataset': dataset,
                'num_trees': int(num_trees),
                'data_size': int(data_size),
                'depth': int(depth),
                'epsilon': None,
                'shuffleMethod': None,
                'AddDummy': None,
                'PostProcess': None,
                'PredictDMatrixDO': None,
                'PredictNO': None,
                'shuffle': None,
                'algorithm': algorithm,
                'PredictBatch': float(predict_batch),
                'time_response': float(time_response)
            }
    records.append(record)

# 构建 DataFrame
df = pd.DataFrame(records)

# # 保留最新记录（参数完全相同时）
# df = df.drop_duplicates(subset=['dataset', 'num_trees', 'data_size', 'depth', 'epsilon', 'shuffleMethod', 'algorithm'], keep='first')
# # 将结果保存到 CSV 文件
# df.to_csv('data/output_log_records.csv', index=False)


# 计算每个组的记录数，并将其作为一个新的列 'count'
df['count'] = df.groupby(['dataset', 'num_trees', 'data_size', 'depth', 'epsilon', 'shuffleMethod', 'algorithm'])['time_response'].transform('size')

# 对于参数完全相同的记录，计算其他列的平均值，并增加一列表示有多少条记录
df_grouped  = df.groupby(['dataset', 'num_trees', 'data_size', 'depth', 'epsilon', 'shuffleMethod', 'algorithm']).agg(
    {
        'AddDummy': 'mean',
        'PostProcess': 'mean',
        'PredictDMatrixDO': 'mean',
        'PredictNO': 'mean',
        'shuffle': 'mean',
        'PredictBatch': 'mean',
        'time_response': 'mean',
        'count': 'first'  # 保留每个组的记录数
    }
).reset_index()

# # 增加一列表示有多少条记录
# df_grouped['count'] = df.groupby(['dataset', 'num_trees', 'data_size', 'depth', 'epsilon', 'shuffleMethod', 'algorithm']).size().reset_index(name='count')['count']

# 将结果保存到 CSV 文件
df_grouped.to_csv('data/output_log_records.csv', index=False)

