import os
import matplotlib.pyplot as plt
import seaborn as sns

HOME_DIR = os.path.abspath('') + "/../"
CURRENT_DIR = os.path.abspath('')
DATA_DIR = os.path.join(CURRENT_DIR, 'data')

username = "user1"
ENCLAVE_FILE = HOME_DIR + "build/enclave/xgboost_enclave.signed"
KEY_FILE = os.path.join(DATA_DIR, "key.txt")
PUB_KEY = HOME_DIR + "config/user1.pem"
CERT_FILE = HOME_DIR + "config/{0}.crt".format(username)
LOG_FILE = os.path.join(DATA_DIR, "time.log")
HIGGS_DIR = os.path.join(DATA_DIR, "higgs")
DEMO_DIR = os.path.join(DATA_DIR, 'demo')
ALLSTATE_DIR = os.path.join(DATA_DIR, "allstate")

TRAIN_SPARSE = "train_sparse.txt"
TEST_SPARSE = "test_sparse.txt"
TRAIN_PROCESSED = "train_processed.csv"
TEST_PROCESSED = "test_processed.csv"
TRAIN_ENC = "train.enc"
TEST_ENC = "test.enc"

import securexgboost as xgb

def initialize_xgboost(username=username, key_file=KEY_FILE, pub_key=PUB_KEY, cert_file=CERT_FILE, enclave_image=ENCLAVE_FILE):
    xgb.init_client(user_name=username, sym_key_file=key_file, priv_key_file=pub_key, cert_file=cert_file)
    xgb.init_server(enclave_image=enclave_image, client_list=[username])
    # Pass in `verify=False` if running in simulation mode.
    xgb.attest(verify=True)

# Convert DataFrame to sparse format
def convert_to_sparse_format(df, labels):
    sparse_data = []
    for i, row in df.iterrows():
        label = labels.iloc[i]
        features = " ".join([f"{j+1}:{v}" for j, v in enumerate(row)])
        sparse_data.append(f"{int(label)} {features}")
    return sparse_data

import time
from functools import wraps

def timer(func):
    @wraps(func)
    def wrapper(*args, **kwargs):
        start_time = time.perf_counter()
        result = func(*args, **kwargs)
        end_time = time.perf_counter()
        print(f"Function '{func.__name__}' elapsed time: {end_time - start_time:.4f} seconds")
        return result
    return wrapper


def data_summary(data):
    plt.hist(data['label'])
    plt.xlabel('label')
    plt.ylabel('Count')
    plt.title('Target Distribution')
    # plt.show()

    corr_matrix = data.corr()
    plt.figure(figsize=(10, 8))
    sns.heatmap(corr_matrix, annot=True, cmap='coolwarm')
    plt.title('Correlation Matrix')
    plt.show()