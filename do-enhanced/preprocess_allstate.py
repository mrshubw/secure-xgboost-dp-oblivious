import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder


# 数据预处理
def preprocess_data(data):
    # 替换缺失值为NaN
    data = data.replace('?', np.nan)
    # 删除不需要的列
    drop_columns = ['Household_ID', 'Vehicle', 'Calendar_Year', 'Claim_Amount']
    data = data.drop(columns=[col for col in drop_columns if col in data.columns])
    # 编码分类变量
    for col in data.columns:
        if data[col].dtype == 'object':
            data[col] = LabelEncoder().fit_transform(data[col].astype(str))
    return data

# 函数：将数据转换为LibSVM格式
def convert_to_libsvm(X, y=None):
    libsvm_str = ""
    for i in range(X.shape[0]):
        print(i)
        if y is not None:
            libsvm_str += f"{y.iloc[i]} "
        for j in range(X.shape[1]):
            if X.iloc[i, j] != 0:  # 只存储非零值，稀疏格式
                libsvm_str += f"{j+1}:{X.iloc[i, j]} "
        libsvm_str = libsvm_str.strip() + "\n"
    return libsvm_str

# 指定列的数据类型，假设第20列（索引19）为对象类型
dtype_spec = {19: str}

# 加载数据
print("reading")
data = pd.read_csv('data/allstate/train_set.csv', dtype=dtype_spec, low_memory=False)
# test_data = pd.read_csv('data/allstate/test_set.csv', dtype=dtype_spec, low_memory=False)
# test_data = test_data.sample(n=1000, random_state=42)

for data_size in [1000, 10000, 100000]:
    data_sampled = data.sample(n=data_size, random_state=42)

    # 预处理训练数据和测试数据
    print("preprocess")
    X_train = preprocess_data(data_sampled)
    y_train = data_sampled['Claim_Amount']
    # X_test = preprocess_data(test_data)

    # 转换训练集和测试集为LibSVM格式
    print("convert")
    libsvm_train = convert_to_libsvm(X_train, y_train)
    # libsvm_test = convert_to_libsvm(X_test)

    print("save")
    # 保存到文件
    with open(f'data/allstate/data{data_size}.txt', 'w') as f:
        f.write(libsvm_train)

# with open('data/allstate/test_libsvm.txt', 'w') as f:
#     f.write(libsvm_test)

print("Data has been converted to LibSVM format and saved to files.")
