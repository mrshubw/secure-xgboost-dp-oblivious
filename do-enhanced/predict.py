import time
import securexgboost as xgb
import os
import pandas as pd
import argparse
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

    for num_rounds in num_rounds_list:
        for data_size in data_size_list:
            for max_depth in max_depth_list:
                with open(LOG_FILE, 'a') as file:
                    file.write(50*'+'+'\n')
                    file.write("dataset: "+dataset+"\n")
                    file.write(f"num_trees:{num_rounds}\ndata_size:{data_size}\ndepth:{max_depth}\n")
                preds = predict(booster=booster[f"{num_rounds}"][f"{max_depth}"], dtest=dtest[f"{data_size}"])
                print(preds)
    # with open(LOG_FILE, 'a') as file:
    #     file.write("dataset: "+dataset+"\n")
    #     file.write("=========================\n")

def evals(preds, test_labels_file):
    test_data = pd.read_csv(test_labels_file, header=None, sep=" ", usecols=[0], names=["label"])
    y_test = test_data["label"].values
    threshold = 0.5
    ypred_binary = (preds > threshold).astype(int)
    accuracy = accuracy_score(y_test, ypred_binary)
    print(f"Model accuracy on test set: {accuracy * 100:.2f}%")
    return accuracy

def main():
    # Create the parser
    parser = argparse.ArgumentParser(description="used for batch processing")

    # Add arguments
    parser.add_argument('-d', '--dataset', type=str, help="datset used", default="higgs")
    parser.add_argument('-t', '--treesnum', type=int, help="number of trees", default=5)
    parser.add_argument('-D', '--depth', type=int, help="maximum depth", default=0)

    # Parse the arguments
    args = parser.parse_args()

    if args.depth == 0:
        depth_list = [2, 3, 4, 5, 6, 7, 8, 9]
        if args.treesnum == 5:
            depth_list = [2, 3, 4, 5, 6, 7, 8, 9, 10]
    else:
        depth_list = [args.depth]


    predict_all(dataset=args.dataset, max_depth_list=depth_list, num_rounds_list=[args.treesnum], data_size_list=[1000, 10000, 100000])

if __name__ == "__main__":
    # main()

    # predict_all(dataset="allstate", max_depth_list=range(3,11), num_rounds_list=[5], data_size_list=[1000, 10000, 100000])
    predict_all(dataset="covtype", max_depth_list=[2, 3, 4, 5, 6, 7, 8, 9, 10], num_rounds_list=[40], data_size_list=[1000])
# [5, 10, 20, 40]
# [2, 3, 4, 5, 6, 7, 8, 9, 10]