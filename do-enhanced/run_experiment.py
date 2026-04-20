import os
import subprocess
import sys
from itertools import product

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PYTHON = sys.executable
DEFAULT_DELTA = 0.00001
CANDIDATE_DATASETS = ['higgs', 'allstate', 'covtype']
CANDIDATE_T_VALUES = [5, 10, 20, 40]
CANDIDATE_DEPTHS = [3, 4, 5, 6, 7, 8, 9]
CANDIDATE_DATA_SIZES = [1000, 10000, 100000]
CANDIDATE_EPSILONS = [0.1, 1.0, 10.0]
CANDIDATE_SHUFFLERS = ["BitonicShuffler", "RecursiveShuffler"]

DEFAULT_DATASET = 'higgs'
DEFAULT_T_VALUE = 40
DEFAULT_DEPTH = 9
DEFAULT_DATA_SIZE = 10000
DEFAULT_EPSILON = 1.0
DEFAULT_SHUFFLER = "BitonicShuffler"
THROUGHPUT_DATA_SIZES = list(range(1000, 30000, 1000))


def run_command(command):
    subprocess.run(command, cwd=SCRIPT_DIR, check=True)


def build_project(*build_args):
    run_command(['./build_project.sh', *build_args])


def run_prediction(script, dataset=DEFAULT_DATASET, treesnum=DEFAULT_T_VALUE, depth=DEFAULT_DEPTH,
                   epsilon=DEFAULT_EPSILON, delta=DEFAULT_DELTA,
                   shuffle_method=DEFAULT_SHUFFLER,
                   data_size_list=None):
    if data_size_list is None:
        data_size_list = [DEFAULT_DATA_SIZE]

    command = [
        PYTHON, script,
        '--dataset', dataset,
        '--treesnum', str(treesnum),
        '--depth', str(depth),
        '--epsilon', str(epsilon),
        '--delta', str(delta),
        '--shuffle-method', shuffle_method,
    ]
    if len(data_size_list) == 1:
        command.extend(['--data-size', str(data_size_list[0])])
    else:
        command.extend([
            '--data-sizes',
            ','.join(str(data_size) for data_size in data_size_list),
        ])
    run_command(command)


def run_predict_grid(script='predict.py',
                     datasets=CANDIDATE_DATASETS,
                     t_values=CANDIDATE_T_VALUES,
                     depth_list=CANDIDATE_DEPTHS,
                     epsilon_list=CANDIDATE_EPSILONS,
                     shuffle_method_list=CANDIDATE_SHUFFLERS,
                     data_size_list=CANDIDATE_DATA_SIZES,
                     delta=DEFAULT_DELTA):
    for shuffle_method, epsilon, dataset, treesnum, depth in product(
        shuffle_method_list, epsilon_list, datasets, t_values, depth_list
    ):
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


def run_commands_DO(script='predict.py',
                    datasets=CANDIDATE_DATASETS,
                    t_values=CANDIDATE_T_VALUES,
                    depth_list=CANDIDATE_DEPTHS,
                    data_size_list=CANDIDATE_DATA_SIZES,
                    epsilon_list=CANDIDATE_EPSILONS,
                    shuffle_method_list=CANDIDATE_SHUFFLERS,
                    delta=DEFAULT_DELTA):
    run_build_grid(
        ('--DO',),
        script=script,
        datasets=datasets,
        t_values=t_values,
        depth_list=depth_list,
        data_size_list=data_size_list,
        epsilon_list=epsilon_list,
        shuffle_method_list=shuffle_method_list,
        delta=delta,
    )


def run_commands_O(script='predict.py',
                   datasets=CANDIDATE_DATASETS,
                   t_values=CANDIDATE_T_VALUES,
                   depth_list=CANDIDATE_DEPTHS,
                   data_size_list=CANDIDATE_DATA_SIZES):
    run_build_grid(
        ('--O',),
        script=script,
        datasets=datasets,
        t_values=t_values,
        depth_list=depth_list,
        data_size_list=data_size_list,
        epsilon_list=[DEFAULT_EPSILON],
        shuffle_method_list=[DEFAULT_SHUFFLER],
        delta=DEFAULT_DELTA,
    )


def run_commands_NO(script='predict.py',
                    datasets=CANDIDATE_DATASETS,
                    t_values=CANDIDATE_T_VALUES,
                    depth_list=CANDIDATE_DEPTHS,
                    data_size_list=CANDIDATE_DATA_SIZES):
    run_build_grid(
        script=script,
        datasets=datasets,
        t_values=t_values,
        depth_list=depth_list,
        data_size_list=data_size_list,
        epsilon_list=[DEFAULT_EPSILON],
        shuffle_method_list=[DEFAULT_SHUFFLER],
        delta=DEFAULT_DELTA,
    )


def run_all_build_modes(script='predict.py',
                        datasets=CANDIDATE_DATASETS,
                        t_values=CANDIDATE_T_VALUES,
                        depth_list=CANDIDATE_DEPTHS,
                        data_size_list=CANDIDATE_DATA_SIZES,
                        epsilon_list=CANDIDATE_EPSILONS,
                        shuffle_method_list=CANDIDATE_SHUFFLERS,
                        order=("DO", "O", "NO"),
                        delta=DEFAULT_DELTA):
    base_options = {
        "script": script,
        "datasets": datasets,
        "t_values": t_values,
        "depth_list": depth_list,
        "data_size_list": data_size_list,
    }
    do_options = {
        **base_options,
        "epsilon_list": epsilon_list,
        "shuffle_method_list": shuffle_method_list,
        "delta": delta,
    }
    build_mode_runners = {
        "DO": (run_commands_DO, do_options),
        "O": (run_commands_O, base_options),
        "NO": (run_commands_NO, base_options),
    }

    for build_mode in order:
        try:
            runner, options = build_mode_runners[build_mode]
        except KeyError:
            raise ValueError("unknown build mode: {}".format(build_mode))
        runner(**options)


# Test the cost of Oblivious, DOXIE and non-Oblivious with different epsilon,
# shuffle method, dataset, data size, tree depth and tree number.
def experiment_main():
    run_all_build_modes(shuffle_method_list=[DEFAULT_SHUFFLER])


def experiment_throughput():
    run_all_build_modes(
        script='predict.py',
        datasets=['higgs'],
        t_values=[20],
        depth_list=[7],
        data_size_list=THROUGHPUT_DATA_SIZES,
    )

def experiment_fig4():
    run_all_build_modes(
        datasets=['higgs'],
        t_values=[10,40,100,500],
        depth_list=[3,5,7,9],
        data_size_list=CANDIDATE_DATA_SIZES,
        epsilon_list=CANDIDATE_EPSILONS,
        shuffle_method_list=[DEFAULT_SHUFFLER]
    )

def experiment_fig5():
    run_commands_DO(
        datasets=CANDIDATE_DATASETS,
        t_values=CANDIDATE_T_VALUES,
        depth_list=[3,5,7,9],
        data_size_list=CANDIDATE_DATA_SIZES,
        epsilon_list=CANDIDATE_EPSILONS,
        shuffle_method_list=[DEFAULT_SHUFFLER]
    )

def run_cost_experiment(datasets=None, t_values=None, depth_list=None,
                        epsilon_list=None, shuffle_method_list=None):
    datasets = datasets or [DEFAULT_DATASET]
    t_values = t_values or [DEFAULT_T_VALUE]
    depth_list = depth_list or CANDIDATE_DEPTHS
    run_all_build_modes(
        script='predict.py',
        datasets=datasets,
        t_values=t_values,
        depth_list=depth_list,
        epsilon_list=epsilon_list or CANDIDATE_EPSILONS,
        shuffle_method_list=shuffle_method_list or CANDIDATE_SHUFFLERS,
        data_size_list=CANDIDATE_DATA_SIZES,
    )


def experiment_NLP():
    run_cost_experiment(t_values=[500], depth_list=[8])


def experiment_higgs_500_trees():
    run_cost_experiment(datasets=['higgs'], t_values=[500])


def experiment_random_forest():
    run_cost_experiment(t_values=[100], depth_list=[8])


def experiment_shufflers():
    run_cost_experiment(
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
    iterations = 1
    for _ in range(iterations):
        experiment_fig4()
        experiment_fig5()
    # experiment_higgs_500_trees()
    # experiment_NLP()
    # experiment_random_forest()
    # experiment_shufflers()
