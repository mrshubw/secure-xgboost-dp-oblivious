import subprocess

def run_commands_DO():
    # 先构建项目
    subprocess.run(['./build_project.sh', '--DO'])

    # 定义数据集和不同的 t 值
    datasets = ['higgs', 'allstate', 'covtype']
    t_values = [5, 10, 20, 40]
    epsilon_list = [0.1, 1.0, 10.0]
    shuffle_method_list = ["BitonicShuffler", "RecursiveShuffler"]
    # epsilon_list = [1.0]
    # shuffle_method_list = ["BitonicShuffler"]

    # select epsilon and shuffle method
    for shuffle_method in shuffle_method_list:
        for epsilon in epsilon_list:
            # modify config file to set epsilon and shuffle method
            with open('data/config.txt', 'w') as f:
                f.write(f"epsilon={epsilon}\n")
                f.write(f"shuffleMethod={shuffle_method}\n")

            # 迭代数据集和 t 值，执行预测命令
            for dataset in datasets:
                for t in t_values:
                    subprocess.run(['python', 'predict.py', '-d', dataset, '-t', str(t)])

def run_commands_O():
    # 先构建项目
    subprocess.run(['./build_project.sh', '--O'])

    # 定义数据集和不同的 t 值
    datasets = ['higgs', 'allstate', 'covtype']
    t_values = [5, 10, 20, 40]
    epsilon_list = [1.0]
    shuffle_method_list = ["BitonicShuffler"]

    # select epsilon and shuffle method
    for shuffle_method in shuffle_method_list:
        for epsilon in epsilon_list:
            # modify config file to set epsilon and shuffle method
            with open('data/config.txt', 'w') as f:
                f.write(f"epsilon={epsilon}\n")
                f.write(f"shuffleMethod={shuffle_method}\n")

            # 迭代数据集和 t 值，执行预测命令
            for dataset in datasets:
                for t in t_values:
                    subprocess.run(['python', 'predict.py', '-d', dataset, '-t', str(t)])

def run_commands():
    # 先构建项目
    subprocess.run(['./build_project.sh'])

    # 定义数据集和不同的 t 值
    datasets = ['higgs', 'allstate', 'covtype']
    t_values = [5, 10, 20, 40]
    epsilon_list = [1.0]
    shuffle_method_list = ["BitonicShuffler"]

    # select epsilon and shuffle method
    for shuffle_method in shuffle_method_list:
        for epsilon in epsilon_list:
            # modify config file to set epsilon and shuffle method
            with open('data/config.txt', 'w') as f:
                f.write(f"epsilon={epsilon}\n")
                f.write(f"shuffleMethod={shuffle_method}\n")

            # 迭代数据集和 t 值，执行预测命令
            for dataset in datasets:
                for t in t_values:
                    subprocess.run(['python', 'predict.py', '-d', dataset, '-t', str(t)])

def test_throughput():

    # 定义数据集和不同的 t 值
    datasets = ['higgs']
    t_values = [20]
    depth_list = [7]
    epsilon_list = [0.1, 1.0, 10.0]
    shuffle_method_list = ["BitonicShuffler"]
    # epsilon_list = [1.0]
    # shuffle_method_list = ["BitonicShuffler"]

    # 先构建项目
    subprocess.run(['./build_project.sh', '--DO'])
    # select epsilon and shuffle method
    for shuffle_method in shuffle_method_list:
        for epsilon in epsilon_list:
            # modify config file to set epsilon and shuffle method
            with open('data/config.txt', 'w') as f:
                f.write(f"epsilon={epsilon}\n")
                f.write(f"shuffleMethod={shuffle_method}\n")

            # 迭代数据集和 t 值，执行预测命令
            for dataset in datasets:
                for t in t_values:
                    for depth in depth_list:
                        subprocess.run(['python', 'test_throughput.py', '-d', dataset, '-t', str(t), '-D', str(depth)])
    
    subprocess.run(['./build_project.sh', '--O'])
    # 迭代数据集和 t 值，执行预测命令
    for dataset in datasets:
        for t in t_values:
            for depth in depth_list:
                subprocess.run(['python', 'test_throughput.py', '-d', dataset, '-t', str(t), '-D', str(depth)])
    subprocess.run(['./build_project.sh'])
    # 迭代数据集和 t 值，执行预测命令
    for dataset in datasets:
        for t in t_values:
            for depth in depth_list:
                subprocess.run(['python', 'test_throughput.py', '-d', dataset, '-t', str(t), '-D', str(depth)])

if __name__ == '__main__':
    for i in range(9):
        run_commands()
        run_commands_O()
        run_commands_DO()
    # test_throughput()

    # subprocess.run(['python', 'handle_log.py'])