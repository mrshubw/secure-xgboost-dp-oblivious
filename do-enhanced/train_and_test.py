import time
import securexgboost as xgb
import os
import pandas as pd
from sklearn.metrics import accuracy_score
from utils import *

def initialize_xgboost(username, key_file, pub_key, cert_file, enclave_image):
    xgb.init_client(user_name=username, sym_key_file=key_file, priv_key_file=pub_key, cert_file=cert_file)
    xgb.init_server(enclave_image=enclave_image, client_list=[username])
    # Pass in `verify=False` if running in simulation mode.
    xgb.attest(verify=True)

def load_data(username, enc_training_data, enc_test_data):
    # Load training data
    dtrain = xgb.DMatrix({username: enc_training_data})
    # Load test data
    dtest = xgb.DMatrix({username: enc_test_data})
    return dtrain, dtest

@timer
def train_model(dtrain, dtest, params, num_rounds):
    booster = xgb.train(params, dtrain, num_rounds, evals=[(dtrain, "train"), (dtest, "test")])
    return booster

@timer
def predict(booster, dtest):
    enc_preds, num_preds = booster.predict(dtest, decrypt=False)
    preds = booster.decrypt_predictions(enc_preds, num_preds)
    print(preds)
    return preds

def evals(preds, test_labels_file):
    test_data = pd.read_csv(test_labels_file, header=None, sep=" ", usecols=[0], names=["label"])
    y_test = test_data["label"].values
    threshold = 0.5
    ypred_binary = (preds > threshold).astype(int)
    accuracy = accuracy_score(y_test, ypred_binary)
    print(f"Model accuracy on test set: {accuracy * 100:.2f}%")
    return accuracy

def train_and_test(data_dir,params, num_rounds, test_labels_file=None):
    initialize_xgboost(username, KEY_FILE, PUB_KEY, CERT_FILE, HOME_DIR + "build/enclave/xgboost_enclave.signed")

    enc_training_data = os.path.join(data_dir, TRAIN_ENC)
    enc_test_data = os.path.join(data_dir, TEST_ENC)
    dtrain, dtest = load_data(username, enc_training_data, enc_test_data)

    booster = train_model(dtrain, dtest, params, num_rounds)
    preds = predict(booster, dtest)
    accuracy = evals(preds, test_labels_file)

    return preds

if __name__ == "__main__":
    data_list = {
        "demo":[DEMO_DIR, DEMO_DIR+'/agaricus.txt.test'],
        "higgs":[HIGGS_DIR, HIGGS_DIR+"/test_sparse.txt"]
    }
    data_dir, test_labels_file = data_list["demo"]
    # # Set parameters
    params = {
        "tree_method": "hist",
        "max_bin": "16",
        "n_gpus": "0",
        "objective": "binary:logistic",
        "min_child_weight": "1",
        "gamma": "0.1",
        "max_depth": "3",
        "verbosity": "3" 
    }
    
    num_rounds = 5
    preds = train_and_test(data_dir, params, num_rounds, test_labels_file)
