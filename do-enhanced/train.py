import pickle
import time
import securexgboost as xgb
import os
import pandas as pd
from sklearn.metrics import accuracy_score
from utils import *
import matplotlib.pyplot as plt

def load_data(username, enc_training_data, enc_test_data):
    # Load training data
    dtrain = xgb.DMatrix({username: enc_training_data})
    # Load test data
    dtest = xgb.DMatrix({username: enc_test_data})
    return dtrain, dtest

@timer
def train(dtrain, dtest, params, num_rounds):
    booster = xgb.train(params, dtrain, num_rounds, evals=[(dtrain, "train"), (dtest, "test")])
    return booster

def train_multi_model(data_dir, max_depth_list, num_rounds_list):
    enc_training_data = os.path.join(data_dir, f"data100000.enc")
    enc_test_data = os.path.join(data_dir, f"data1000.enc")
    dtrain, dtest = load_data(username, enc_training_data, enc_test_data)

    for max_depth in max_depth_list:
        for num_rounds in num_rounds_list:
            # reg:squarederror
            # binary:logistic
            params = {
                "tree_method": "hist",
                "max_bin": "16",
                "n_gpus": "0",
                "objective": "reg:squarederror",
                "min_child_weight": "1",
                "gamma": "0.1",
                "max_depth": f"{max_depth}",
                "verbosity": "2" 
            }
            booster = train(dtrain, dtest, params, num_rounds)
            
            # Save model to a file
            model_name = f"modeld{max_depth}n{num_rounds}.model"
            print("Saving model "+model_name)
            booster.save_model(os.path.join(data_dir, model_name))

if __name__ == "__main__":
    initialize_xgboost()
    data_dir = os.path.join(DATA_DIR, "demo")
    train_multi_model(data_dir, max_depth_list=range(8,11), num_rounds_list=[10, 20, 40])
