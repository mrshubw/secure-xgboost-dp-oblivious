import time
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

training_data = HOME_DIR + "bottleneck/data/agaricus.txt100k.train"
enc_training_data = CURRENT_DIR + "data/train.enc"

# Encrypt training data
xgb.encrypt_file(training_data, enc_training_data, KEY_FILE)

test_data = HOME_DIR + "bottleneck/data/agaricus.txt.test"
enc_test_data = CURRENT_DIR + "data/test.enc"

# Encrypt test data
xgb.encrypt_file(test_data, enc_test_data, KEY_FILE)

xgb.init_client(user_name=username, sym_key_file=KEY_FILE, priv_key_file=PUB_KEY, cert_file=CERT_FILE)
xgb.init_server(enclave_image=HOME_DIR + "build/enclave/xgboost_enclave.signed", client_list=[username])
# Remote Attestation

# Pass in `verify=False` if running in simulation mode.
xgb.attest(verify=False)

# Load training data
dtrain = xgb.DMatrix({username: enc_training_data})
# Load test data
dtest = xgb.DMatrix({username: enc_test_data})

# Set parameters
params = {
        "tree_method": "hist",
        "max_bin": "16",
        "n_gpus": "0",
        "objective": "binary:logistic",
        "min_child_weight": "1",
        "gamma": "0.1",
        "max_depth": "8",
        "verbosity": "3" 
}

# Train
num_rounds = 5
print('xgb.train...')
time_start = time.time()
booster = xgb.train(params, dtrain, num_rounds, evals=[(dtrain, "train"), (dtest, "test")])
time_end = time.time()
time_c= time_end - time_start   #运行所花时间
print('xgb.train cost', time_c, 's')


# Get Encrypted Predictions
print('booster.predict...')
time_start = time.time()
enc_preds, num_preds = booster.predict(dtest, decrypt=False)
time_end = time.time()
time_c= time_end - time_start   #运行所花时间
print('booster.predict cost', time_c, 's')

# Decrypt Predictions
preds = booster.decrypt_predictions(enc_preds, num_preds)
print(preds)

# for test_file in ['agaricus10.txt.test', 'agaricus100.txt.test']:
# # for test_file in ['agaricus10.txt.test']:
#     print("="*50)
#     test_data = HOME_DIR + "bottleneck/data/" + test_file
#     enc_test_data = CURRENT_DIR + "test.enc"
    
#     # Encrypt test data
#     xgb.encrypt_file(test_data, enc_test_data, KEY_FILE)
#     # Load test data
#     dtest = xgb.DMatrix({username: enc_test_data})

#     # Get Encrypted Predictions
#     print('booster.predict...')
#     time_start = time.time()
#     enc_preds, num_preds = booster.predict(dtest, decrypt=False)
#     time_end = time.time()
#     time_c= time_end - time_start   #运行所花时间
#     print('booster.predict cost', time_c, 's')

#     # Decrypt Predictions
#     preds = booster.decrypt_predictions(enc_preds, num_preds)
#     print(preds)