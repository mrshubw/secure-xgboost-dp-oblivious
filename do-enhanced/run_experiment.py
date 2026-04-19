import os
import subprocess
import sys
from itertools import product

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PYTHON = sys.executable
DEFAULT_DELTA = 0.00001
DEFAULT_DATASETS = ['higgs', 'allstate', 'covtype']
DEFAULT_T_VALUES = [5, 10, 20, 40]
DEFAULT_DEPTHS = [3, 4, 5, 6, 7, 8, 9]
DEFAULT_DATA_SIZES = [1000, 10000, 100000]
DEFAULT_EPSILONS = [0.1, 1.0, 10.0]
DEFAULT_SHUFFLERS = ["BitonicShuffler"]
DEFAULT_DO_SHUFFLERS = ["BitonicShuffler", "RecursiveShuffler"]
THROUGHPUT_DATA_SIZES = list(range(1000, 30000, 1000))


def run_command(command):
    subprocess.run(command, cwd=SCRIPT_DIR, check=True)


def build_project(*build_args):
    run_command(['./build_project.sh', *build_args])


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

    for shuffle_method, epsilon, dataset, treesnum in product(
        shuffle_method_list, epsilon_list, datasets, t_values
    ):
        for depth in depth_list or DEFAULT_DEPTHS:
            run_prediction(
                script, dataset, treesnum, depth,
                epsilon=epsilon,
                delta=delta,
                shuffle_method=shuffle_method,
                data_size_list=data_size_list,
            )


def run_build_grid(build_args=(), **grid_options):
    build_project(*build_args)
    run_predict_grid(**grid_options)


def run_build_variants(script, datasets, t_values, depth_list=None,
                       data_size_list=None, epsilon_list=None,
                       shuffle_method_list=None, order=("DO", "O", "NO")):
    base_options = {
        "script": script,
        "datasets": datasets,
        "t_values": t_values,
        "depth_list": depth_list,
        "data_size_list": data_size_list,
    }
    variants = {
        "DO": (
            ("--DO",),
            {
                "epsilon_list": epsilon_list or DEFAULT_EPSILONS,
                "shuffle_method_list": shuffle_method_list or DEFAULT_SHUFFLERS,
            },
        ),
        "O": (("--O",), {}),
        "NO": ((), {}),
    }

    for variant in order:
        build_args, extra_options = variants[variant]
        run_build_grid(build_args, **base_options, **extra_options)


def run_commands_DO():
    run_build_grid(
        ('--DO',),
        script='predict.py',
        datasets=DEFAULT_DATASETS,
        t_values=DEFAULT_T_VALUES,
        data_size_list=DEFAULT_DATA_SIZES,
        epsilon_list=DEFAULT_EPSILONS,
        shuffle_method_list=DEFAULT_SHUFFLERS,
    )


def run_commands_O():
    run_build_grid(
        ('--O',),
        script='predict.py',
        datasets=DEFAULT_DATASETS,
        t_values=DEFAULT_T_VALUES,
        data_size_list=DEFAULT_DATA_SIZES,
    )


def run_commands():
    run_build_grid(
        script='predict.py',
        datasets=DEFAULT_DATASETS,
        t_values=DEFAULT_T_VALUES,
        data_size_list=DEFAULT_DATA_SIZES,
    )


# Test the cost of Oblivious, DOXIE and non-Oblivious with different epsilon,
# shuffle method, dataset, data size, tree depth and tree number.
def experiment_main(iterations=1):
    for _ in range(iterations):
        run_commands()
        run_commands_O()
        run_commands_DO()


def experiment_throughput():
    run_build_variants(
        script='predict.py',
        datasets=['higgs'],
        t_values=[20],
        depth_list=[7],
        data_size_list=THROUGHPUT_DATA_SIZES,
    )


def test_cost(datasets=None, t_values=None, depth_list=None,
              epsilon_list=None, shuffle_method_list=None):
    datasets = datasets or ['higgs']
    t_values = t_values or [20]
    depth_list = depth_list or DEFAULT_DEPTHS
    run_build_variants(
        script='predict.py',
        datasets=datasets,
        t_values=t_values,
        depth_list=depth_list,
        epsilon_list=epsilon_list or DEFAULT_EPSILONS,
        shuffle_method_list=shuffle_method_list or DEFAULT_SHUFFLERS,
        data_size_list=DEFAULT_DATA_SIZES,
    )


def experiment_NLP():
    test_cost(t_values=[500], depth_list=[8])


def experiment_higgs_500_trees():
    test_cost(datasets=['higgs'], t_values=[500])


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
    # experiment_main()
    # experiment_throughput()
    # experiment_NLP()
    experiment_higgs_500_trees()
    # experiment_random_forest()
    # experiment_shufflers()
