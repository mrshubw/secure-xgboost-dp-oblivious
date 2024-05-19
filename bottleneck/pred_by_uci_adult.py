import pandas as pd
from sklearn.model_selection import train_test_split
import securexgboost as xgb
import os

username = "user1"
HOME_DIR = os.path.abspath('') + "/../"
CURRENT_DIR = os.path.abspath('') + "/"
PUB_KEY = HOME_DIR + "config/user1.pem"
CERT_FILE = HOME_DIR + "config/{0}.crt".format(username)

KEY_FILE = CURRENT_DIR + "data/key.txt"

# Generate a key you will be using for encryption
xgb.generate_client_key(KEY_FILE)

# Load dataset
# Step 1: Load and preprocess the dataset
column_names = ["age", "workclass", "fnlwgt", "education", "education_num", "marital_status", "occupation", 
                "relationship", "race", "sex", "capital_gain", "capital_loss", "hours_per_week", "native_country", "income"]
train_data = pd.read_csv("./DataSets/UCI/adult/adult.data", names=column_names)
test_data = pd.read_csv("./DataSets/UCI/adult/adult.test", names=column_names, skiprows=1)
train_data['income'] = train_data['income'].apply(lambda x: 1 if x.strip() == ">50K" else 0)
test_data['income'] = test_data['income'].apply(lambda x: 1 if x.strip() == ">50K." else 0)
train_data = pd.get_dummies(train_data, drop_first=True)
test_data = pd.get_dummies(test_data, drop_first=True)
train_data.to_csv("train.csv", index=False)
test_data.to_csv("test.csv", index=False)

xgb.encrypt_file("train.csv", "train.enc", KEY_FILE)
xgb.encrypt_file("test.csv", "test.enc", KEY_FILE)

# Initialize client and connect to enclave
xgb.init_client(user_name=username, sym_key_file=KEY_FILE, priv_key_file=PUB_KEY, cert_file=CERT_FILE)
xgb.init_server(enclave_image=HOME_DIR + "build/enclave/xgboost_enclave.signed", client_list=[username])

# Perform remote attestation
# xgb.attest(verify=True)

# Print configuration to check for nonce
import securexgboost.core as core
print(core._CONF)
# If nonce is not set, debug further or reinitialize
if "nonce" not in core._CONF:
    print("Nonce not found in configuration")

# Load the encrypted data
dtrain = xgb.DMatrix({"user1": "train.enc"})
dtest = xgb.DMatrix({"user1": "test.enc"})

params = {"objective": "binary:logistic", "gamma": "0.1", "max_depth": "3"}
num_rounds = 5
booster = xgb.train(params, dtrain, num_rounds, evals=[(dtrain, "train"), (dtest, "test")])
