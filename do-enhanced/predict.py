import time
import securexgboost as xgb
import os
import pandas as pd
from sklearn.metrics import accuracy_score
from utils import *

@timer
def predict(booster, dtest):
    enc_preds, num_preds = booster.predict(dtest, decrypt=False)
    preds = booster.decrypt_predictions(enc_preds, num_preds)

    return preds

def predict_all(dataset, max_depth_list, num_rounds_list, data_size_list):
    initialize_xgboost()
    data_dir = os.path.join(DATA_DIR, dataset)

    dtest = {}
    for data_size in data_size_list:
        enc_test_data = os.path.join(data_dir, f"data{data_size}.enc")
        dtest[f"{data_size}"] = xgb.DMatrix({username: enc_test_data})

    booster = {}
    for num_rounds in num_rounds_list:
        booster[f"{num_rounds}"] = {}
        for max_depth in max_depth_list:
            model_name = f"modeld{max_depth}n{num_rounds}.model"
            booster[f"{num_rounds}"][f"{max_depth}"] = xgb.Booster(model_file=os.path.join(data_dir, model_name))

    with open(LOG_FILE, 'a') as file:
        file.write("=========================\n")
        file.write("dataset: "+dataset+"\n")
    for num_rounds in num_rounds_list:
        for data_size in data_size_list:
            for max_depth in max_depth_list:
                with open(LOG_FILE, 'a') as file:
                    file.write(f"n:{num_rounds} s:{data_size} d:{max_depth} ")
                preds = predict(booster=booster[f"{num_rounds}"][f"{max_depth}"], dtest=dtest[f"{data_size}"])
                print(preds)
    with open(LOG_FILE, 'a') as file:
        file.write("dataset: "+dataset+"\n")
        file.write("=========================\n")

def evals(preds, test_labels_file):
    test_data = pd.read_csv(test_labels_file, header=None, sep=" ", usecols=[0], names=["label"])
    y_test = test_data["label"].values
    threshold = 0.5
    ypred_binary = (preds > threshold).astype(int)
    accuracy = accuracy_score(y_test, ypred_binary)
    print(f"Model accuracy on test set: {accuracy * 100:.2f}%")
    return accuracy

if __name__ == "__main__":
    # predict_all(dataset="allstate", max_depth_list=range(3,11), num_rounds_list=[5], data_size_list=[1000, 10000, 100000])
    predict_all(dataset="demo", max_depth_list=range(3,8), num_rounds_list=[5, 10, 20, 40], data_size_list=[10000])
