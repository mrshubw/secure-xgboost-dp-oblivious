import argparse
import os

import securexgboost as xgb

from utils import *


DEFAULT_DATASET = "allstate"
DEFAULT_DEPTHS = [3, 5, 7, 9]
DEFAULT_NUM_ROUNDS = [100, 500]


def parse_int_list(value):
    return [int(item) for item in value.split(",") if item]


def load_data(username, enc_training_data, enc_test_data):
    dtrain = xgb.DMatrix({username: enc_training_data})
    dtest = xgb.DMatrix({username: enc_test_data})
    return dtrain, dtest


@timer
def train(dtrain, dtest, params, num_rounds):
    booster = xgb.train(
        params,
        dtrain,
        num_rounds,
        evals=[(dtrain, "train"), (dtest, "test")],
    )
    return booster


def train_multi_model(data_dir, max_depth_list, num_rounds_list, params):
    enc_training_data = os.path.join(data_dir, "data100000.enc")
    enc_test_data = os.path.join(data_dir, "data1000.enc")
    dtrain, dtest = load_data(username, enc_training_data, enc_test_data)

    for max_depth in max_depth_list:
        for num_rounds in num_rounds_list:
            model_params = dict(params)
            model_params["max_depth"] = str(max_depth)
            booster = train(
                dtrain,
                dtest,
                model_params,
                num_rounds,
            )

            model_name = "modeld{}n{}.model".format(max_depth, num_rounds)
            print("Saving model {}".format(model_name))
            booster.save_model(os.path.join(data_dir, model_name))


def build_params():
    params_covtype = {
        "objective": "multi:softmax",
        "num_class": 7,
        "tree_method": "hist",
        "max_bin": "16",
        "eval_metric": "mlogloss",
        "eta": 0.1,
        "subsample": 0.8,
        "colsample_bytree": 0.8,
        "seed": 42,
    }
    params_higgs = {
        "tree_method": "hist",
        "max_bin": "16",
        "n_gpus": "0",
        "objective": "binary:logistic",
        "min_child_weight": "1",
        "gamma": "0.1",
        "verbosity": "2",
    }
    params_allstate = {
        "tree_method": "hist",
        "max_bin": "16",
        "n_gpus": "0",
        "objective": "reg:squarederror",
        "min_child_weight": "1",
        "gamma": "0.1",
        "verbosity": "2",
    }
    return {
        "higgs": params_higgs,
        "allstate": params_allstate,
        "covtype": params_covtype,
    }


def main():
    parser = argparse.ArgumentParser(description="Train Secure XGBoost models")
    parser.add_argument(
        "--dataset",
        type=str,
        default=DEFAULT_DATASET,
        choices=["allstate", "covtype", "higgs"],
    )
    parser.add_argument(
        "--depths",
        type=str,
        default="3,5,7,9",
        help="Comma-separated max_depth values",
    )
    parser.add_argument(
        "--num-rounds",
        type=str,
        default="100,500",
        help="Comma-separated target tree counts",
    )
    args = parser.parse_args()

    initialize_xgboost()
    params = build_params()
    data_dir = os.path.join(DATA_DIR, args.dataset)
    train_multi_model(
        data_dir,
        max_depth_list=parse_int_list(args.depths),
        num_rounds_list=parse_int_list(args.num_rounds),
        params=params[args.dataset],
    )


if __name__ == "__main__":
    main()
