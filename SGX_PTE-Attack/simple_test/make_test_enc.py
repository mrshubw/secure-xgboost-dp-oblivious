import os
import securexgboost as xgb

HOME_DIR = "/home/qsp/secure-xgboost/" # os.path.abspath('') + "/../../"
CURRENT_DIR = os.path.abspath('')
DATA_DIR = os.path.join(CURRENT_DIR, 'data')

username = "user1"
KEY_FILE = "key.txt"
ENCLAVE_FILE = HOME_DIR + "build/enclave/xgboost_enclave.signed"
PUB_KEY = HOME_DIR + "config/user1.pem"
CERT_FILE = HOME_DIR + "config/{0}.crt".format(username)

DATA_SIZE = 2

xgb.generate_client_key(KEY_FILE)
xgb.encrypt_file(f"data/higgs/datauser1.txt", f"data/higgs/datauser1.enc", KEY_FILE)
xgb.encrypt_file(f"data/higgs/datauser2.txt", f"data/higgs/datauser2.enc", KEY_FILE)