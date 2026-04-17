import os
import subprocess
import sys

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PYTHON = sys.executable
DEFAULT_DELTA = 0.00001


def run_command(command):
    subprocess.run(command, cwd=SCRIPT_DIR, check=True)


def build_project(*build_args):
    run_command(['./build_project.sh', *build_args])


def default_depths(treesnum):
    if treesnum == 5:
        return [2, 3, 4, 5, 6, 7, 8, 9, 10]
    return [2, 3, 4, 5, 6, 7, 8, 9]


def run_prediction(script, dataset, treesnum, depth=None,
                   epsilon=1.0, delta=DEFAULT_DELTA,
                   shuffle_method="BitonicShuffler",
                   data_size_list=None):
    command = [
        PYTHON, script,
        '--dataset', dataset,
        '--treesnum', str(treesnum),
        '--epsilon', str(epsilon),
        '--delta', str(delta),
        '--shuffle-method', shuffle_method,
    ]
    if depth is not None:
        command.extend(['--depth', str(depth)])
    if data_size_list:
        if len(data_size_list) == 1:
            command.extend(['--data-size', str(data_size_list[0])])
        else:
            command.extend([
                '--data-sizes',
                ','.join(str(data_size) for data_size in data_size_list),
            ])
    run_command(command)


def run_predict_grid(script, datasets, t_values, depth_list=None,
                     epsilon_list=None, shuffle_method_list=None,
                     data_size_list=None, delta=DEFAULT_DELTA):
    epsilon_list = epsilon_list or [1.0]
    shuffle_method_list = shuffle_method_list or ["BitonicShuffler"]

    for shuffle_method in shuffle_method_list:
        for epsilon in epsilon_list:
            for dataset in datasets:
                for treesnum in t_values:
                    depths = depth_list or default_depths(treesnum)
                    for depth in depths:
                        run_prediction(
                            script, dataset, treesnum, depth,
                            epsilon=epsilon,
                            delta=delta,
                            shuffle_method=shuffle_method,
                            data_size_list=data_size_list,
                        )


def run_commands_DO():
    build_project('--DO')
    run_predict_grid(
        script='predict.py',
        datasets=['higgs', 'allstate', 'covtype'],
        t_values=[5, 10, 20, 40],
        data_size_list=[1000, 10000, 100000],
        epsilon_list=[0.1, 1.0, 10.0],
        shuffle_method_list=["BitonicShuffler", "RecursiveShuffler"],
    )


def run_commands_O():
    build_project('--O')
    run_predict_grid(
        script='predict.py',
        datasets=['higgs', 'allstate', 'covtype'],
        t_values=[5, 10, 20, 40],
        data_size_list=[1000, 10000, 100000],
    )


def run_commands():
    build_project()
    run_predict_grid(
        script='predict.py',
        datasets=['higgs', 'allstate', 'covtype'],
        t_values=[5, 10, 20, 40],
        data_size_list=[1000, 10000, 100000],
    )


# Test the cost of Oblivious, DOXIE and non-Oblivious with different epsilon,
# shuffle method, dataset, data size, tree depth and tree number.
def experiment_main(iterations=10):
    for _ in range(iterations):
        run_commands()
        run_commands_O()
        run_commands_DO()


def experiment_throughput():
    datasets = ['higgs']
    t_values = [20]
    depth_list = [7]

    build_project('--DO')
    run_predict_grid(
        script='test_throughput.py',
        datasets=datasets,
        t_values=t_values,
        depth_list=depth_list,
        epsilon_list=[0.1, 1.0, 10.0],
        shuffle_method_list=["BitonicShuffler"],
    )

    build_project('--O')
    run_predict_grid(
        script='test_throughput.py',
        datasets=datasets,
        t_values=t_values,
        depth_list=depth_list,
    )

    build_project()
    run_predict_grid(
        script='test_throughput.py',
        datasets=datasets,
        t_values=t_values,
        depth_list=depth_list,
    )


def test_cost(datasets=None, t_values=None, depth_list=None,
              epsilon_list=None, shuffle_method_list=None):
    datasets = datasets or ['higgs']
    t_values = t_values or [20]
    depth_list = depth_list or [7]
    epsilon_list = epsilon_list or [0.1, 1.0, 10.0]
    shuffle_method_list = shuffle_method_list or ["BitonicShuffler"]

    build_project('--DO')
    run_predict_grid(
        script='predict.py',
        datasets=datasets,
        t_values=t_values,
        depth_list=depth_list,
        epsilon_list=epsilon_list,
        shuffle_method_list=shuffle_method_list,
        data_size_list=[1000, 10000, 100000],
    )

    build_project('--O')
    run_predict_grid(
        script='predict.py',
        datasets=datasets,
        t_values=t_values,
        depth_list=depth_list,
        data_size_list=[1000, 10000, 100000],
    )

    build_project()
    run_predict_grid(
        script='predict.py',
        datasets=datasets,
        t_values=t_values,
        depth_list=depth_list,
        data_size_list=[1000, 10000, 100000],
    )


def experiment_NLP():
    test_cost(t_values=[500], depth_list=[8])


def experiment_random_forest():
    test_cost(t_values=[100], depth_list=[8])


def experiment_shufflers():
    test_cost(
        t_values=[100],
        depth_list=[8],
        shuffle_method_list=[
            "BitonicShuffler",
            "RecursiveShuffler",
            "BubbleShuffler",
        ],
    )


if __name__ == '__main__':
    experiment_main()
    experiment_throughput()
    # experiment_NLP()
    # experiment_random_forest()
    # experiment_shufflers()
