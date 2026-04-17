import argparse
import os

from predict import predict_batches
from utils import DATA_DIR

THROUGHPUT_RESULTS_FILE = os.path.join(DATA_DIR, "throughput_results.csv")


def main():
    parser = argparse.ArgumentParser(description="run throughput prediction experiments")
    parser.add_argument('--dataset', type=str, help="dataset used", default="higgs")
    parser.add_argument('--treesnum', type=int, help="number of trees", default=20)
    parser.add_argument('--depth', type=int, help="maximum depth", default=7)
    parser.add_argument('--epsilon', type=float, default=1.0)
    parser.add_argument('--delta', type=float, default=0.00001)
    parser.add_argument('--shuffle-method', type=str, default="BitonicShuffler")
    parser.add_argument('--results-file', type=str, default=THROUGHPUT_RESULTS_FILE)
    parser.add_argument('--min-data-size', type=int, default=10000)
    parser.add_argument('--max-data-size', type=int, default=30000)
    parser.add_argument('--data-size-step', type=int, default=1000)

    args = parser.parse_args()

    data_size_list = list(
        range(args.min_data_size, args.max_data_size, args.data_size_step)
    )
    predict_batches(
        dataset=args.dataset,
        max_depth=args.depth,
        num_rounds=args.treesnum,
        data_size_list=data_size_list,
        epsilon=args.epsilon,
        delta=args.delta,
        shuffle_method=args.shuffle_method,
        results_file=args.results_file,
    )


if __name__ == "__main__":
    main()
