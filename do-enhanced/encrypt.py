from utils import *
import securexgboost as xgb
import os



def encrypt_file(data_dir, file_list, key_file=KEY_FILE, verbose=True):
    """
    Encrypt the processed data files.
    
    Parameters:
    - data_dir: Path to save the processed data
    - train_file: Filename of the training data
    - test_file: Filename of the testing data
    - key_file: Encryption key file
    - verbose: Whether to print detailed logs
    """
    for file in file_list:
        file_enc = os.path.splitext(file)[0] + ".enc"
        xgb.encrypt_file(os.path.join(data_dir, file), os.path.join(data_dir, file_enc), key_file)
    if verbose:
        print("Data encrypted.")

encrypt_file(COVTYPE_DIR, ["data1000.txt", "data10000.txt", "data100000.txt"])