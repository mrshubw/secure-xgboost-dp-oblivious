import time
import securexgboost as xgb
import os
import argparse
import csv
from utils import *

RESULTS_FILE = os.path.join(DATA_DIR, "prediction_results.csv")
RESULT_FIELDS = [
    "dataset",
    "data_size",
    "num_trees",
    "depth",
    "time_response",
    "PredictBatch",
    "algorithm",
    "epsilon",
    "delta",
    "shuffleMethod",
    "doxieMemoryAlignment",
    "doxieBlockedKernel",
    "doxieAdvancedComposition",
    "PredictDMatrixDO",
    "PredictOnline",
    "PredictNO",
    "AddDummy",
    "shuffle",
    "PostProcess",
]

@timer
def predict(booster, dtest):
    enc_preds, num_preds = booster.predict(dtest, decrypt=False)
    preds = booster.decrypt_predictions(enc_preds, num_preds)

    return preds

def write_result(row, results_file=RESULTS_FILE):
    results_dir = os.path.dirname(results_file)
    if results_dir:
        os.makedirs(results_dir, exist_ok=True)
        
    write_header = not os.path.exists(results_file) or os.path.getsize(results_file) == 0
    with open(results_file, 'a', newline='') as file:
        writer = csv.DictWriter(file, fieldnames=RESULT_FIELDS)
        if write_header:
            writer.writeheader()
        writer.writerow({field: row.get(field, "") for field in RESULT_FIELDS})

def parse_data_sizes(data_sizes):
    if isinstance(data_sizes, str):
        return [int(data_size) for data_size in data_sizes.split(',') if data_size]
    return [int(data_size) for data_size in data_sizes]

def predict_batches(dataset, max_depth, num_rounds, data_size_list,
                    epsilon=1.0, delta=0.00001,
                    shuffle_method="BitonicShuffler",
                    doxie_memory_alignment=True,
                    doxie_blocked_kernel=True,
                    doxie_advanced_composition=True,
                    results_file=RESULTS_FILE):
    initialize_xgboost()
    data_dir = os.path.join(DATA_DIR, dataset)
    model_name = f"modeld{max_depth}n{num_rounds}.model"
    booster = xgb.Booster(model_file=os.path.join(data_dir, model_name))
    booster.set_param({
        "doxie_epsilon": epsilon,
        "doxie_delta": delta,
        "doxie_shuffle_method": shuffle_method,
        "doxie_memory_alignment": "true" if doxie_memory_alignment else "false",
        "doxie_blocked_kernel": "true" if doxie_blocked_kernel else "false",
        "doxie_advanced_composition":
            "true" if doxie_advanced_composition else "false",
    })

    for data_size in data_size_list:
        enc_test_data = os.path.join(data_dir, f"data{data_size}.enc")
        dtest = xgb.DMatrix({username: enc_test_data})
        time_start = time.time()
        preds = predict(booster=booster, dtest=dtest)
        time_end = time.time()
        time_response = time_end - time_start
        metrics = booster.get_last_prediction_metrics()
        row = {
            "dataset": dataset,
            "num_trees": num_rounds,
            "data_size": data_size,
            "depth": max_depth,
            "time_response": time_response,
        }
        row.update(metrics)
        write_result(row, results_file)

def predict_once(dataset, max_depth, num_rounds, data_size,
                 epsilon=1.0, delta=0.00001,
                 shuffle_method="BitonicShuffler",
                 doxie_memory_alignment=True,
                 doxie_blocked_kernel=True,
                 doxie_advanced_composition=True,
                 results_file=RESULTS_FILE):
    predict_batches(
        dataset=dataset,
        max_depth=max_depth,
        num_rounds=num_rounds,
        data_size_list=[data_size],
        epsilon=epsilon,
        delta=delta,
        shuffle_method=shuffle_method,
        doxie_memory_alignment=doxie_memory_alignment,
        doxie_blocked_kernel=doxie_blocked_kernel,
        doxie_advanced_composition=doxie_advanced_composition,
        results_file=results_file,
    )

def predict_all(dataset, max_depth_list, num_rounds_list, data_size_list,
                epsilon=1.0, delta=0.00001,
                shuffle_method="BitonicShuffler",
                doxie_advanced_composition=True,
                results_file=RESULTS_FILE):
    for num_rounds in num_rounds_list:
        for max_depth in max_depth_list:
            predict_batches(
                dataset=dataset,
                max_depth=max_depth,
                num_rounds=num_rounds,
                data_size_list=data_size_list,
                epsilon=epsilon,
                delta=delta,
                shuffle_method=shuffle_method,
                doxie_advanced_composition=doxie_advanced_composition,
                results_file=results_file,
            )

def evals(preds, test_labels_file, log_file=None):
    import pandas as pd
    from sklearn.metrics import accuracy_score, roc_auc_score

    test_data = pd.read_csv(test_labels_file, header=None, sep=" ", usecols=[0], names=["label"])
    y_test = test_data["label"].values
    threshold = 0.5
    ypred_binary = (preds > threshold).astype(int)
    accuracy = accuracy_score(y_test, ypred_binary)
    auc = roc_auc_score(y_test, preds)
    print(f"Model accuracy on test set: {accuracy * 100:.2f}%")
    print(f"Model AUC on test set: {auc * 100:.2f}%")

    if log_file:
        with open(log_file, 'a') as file:
            file.write(f"accuracy:{accuracy}\n")
            file.write(f"AUC:{auc}\n")
    return accuracy, auc

def main():
    parser = argparse.ArgumentParser(description="run prediction for one booster")
    parser.add_argument('--dataset', type=str, help="dataset used", default="higgs")
    parser.add_argument('--treesnum', type=int, help="number of trees", default=500)
    parser.add_argument('--depth', type=int, help="maximum depth", default=8)
    parser.add_argument('--data-size', type=int, default=10000)
    parser.add_argument('--data-sizes', type=str, default=None,
                        help="comma separated batch sizes, e.g. 1000,10000,100000")
    parser.add_argument('--epsilon', type=float, default=1)
    parser.add_argument('--delta', type=float, default=0.00001)
    parser.add_argument('--shuffle-method', type=str, default="BitonicShuffler")
    parser.add_argument('--doxie-memory-alignment',
                        choices=["true", "false"],
                        default="true",
                        help="enable DOXIE page-aligned tree memory")
    parser.add_argument('--doxie-blocked-kernel',
                        choices=["true", "false"],
                        default="false",
                        help="use the block-major DOXIE prediction kernel")
    parser.add_argument('--doxie-advanced-composition',
                        choices=["true", "false"],
                        default="true",
                        help="use advanced composition for per-tree privacy budget")
    parser.add_argument('--results-file', type=str, default=RESULTS_FILE)

    args = parser.parse_args()
    data_size_list = (
        parse_data_sizes(args.data_sizes)
        if args.data_sizes
        else [args.data_size]
    )

    predict_batches(
        dataset=args.dataset,
        max_depth=args.depth,
        num_rounds=args.treesnum,
        data_size_list=data_size_list,
        epsilon=args.epsilon,
        delta=args.delta,
        shuffle_method=args.shuffle_method,
        doxie_memory_alignment=args.doxie_memory_alignment == "true",
        doxie_blocked_kernel=args.doxie_blocked_kernel == "true",
        doxie_advanced_composition=args.doxie_advanced_composition == "true",
        results_file=args.results_file,
    )

if __name__ == "__main__":
    main()
