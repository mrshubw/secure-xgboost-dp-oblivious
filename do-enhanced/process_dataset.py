import os
import argparse
import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder, MinMaxScaler, StandardScaler
import securexgboost as xgb
from utils import *

random_state = 42
CURRENT_DIR = os.path.dirname(__file__)
DEFAULT_DATASET = 'allstate'
DEFAULT_SIZE_LIST = list(range(1000, 30000, 1000))

# Define column names for Higgs dataset
higgs_column_names = ["label", "lepton_pT", "lepton_eta", "lepton_phi", "missing_energy_magnitude", "missing_energy_phi",
                      "jet1_pt", "jet1_eta", "jet1_phi", "jet1_b-tag",
                      "jet2_pt", "jet2_eta", "jet2_phi", "jet2_b-tag",
                      "jet3_pt", "jet3_eta", "jet3_phi", "jet3_b-tag",
                      "jet4_pt", "jet4_eta", "jet4_phi", "jet4_b-tag",
                      "m_jj", "m_jjj", "m_lv", "m_jlv", "m_bb", "m_wbb", "m_wwbb"]

# Function to preprocess Allstate dataset
def preprocess_allstate(data, size, encrypt=True):
    # Sample the data
    data = data.sample(n=size, random_state=42)

    # Replace missing values with NaN
    data = data.replace('?', np.nan)
    # Drop unnecessary columns
    drop_columns = ['Household_ID', 'Vehicle', 'Calendar_Year']
    data = data.drop(columns=[col for col in drop_columns if col in data.columns])
    # Encode categorical variables
    for col in data.columns:
        if data[col].dtype == 'object':
            data[col] = LabelEncoder().fit_transform(data[col].astype(str))
    
    X_train = data.drop(columns=['Claim_Amount'])
    y_train = data['Claim_Amount']
    
    # Convert to LibSVM format
    libsvm_train = convert_to_libsvm(X_train, y_train)

    data_dir = os.path.join(CURRENT_DIR, 'data/allstate')
    os.makedirs(data_dir, exist_ok=True)
    data_path = os.path.join(data_dir, f"data{size}.txt")

    # Save to file
    with open(data_path, 'w') as f:
        f.write(libsvm_train)

    if encrypt:
        xgb.encrypt_file(data_path, os.path.join(data_dir, f"data{size}.enc"), KEY_FILE)

# Function to preprocess Covtype dataset
def preprocess_covtype(data, size, encrypt=True):
    # Convert labels to be in the range [0, num_class)
    data.iloc[:, -1] = data.iloc[:, -1] - 1
    
    # Sample the data
    data_sampled = data.sample(n=size, random_state=42)
    
    # Convert to LibSVM format
    libsvm_data = convert_to_libsvm(data_sampled.iloc[:, :-1], data_sampled.iloc[:, -1])

    data_dir = os.path.join(CURRENT_DIR, 'data/covtype')
    os.makedirs(data_dir, exist_ok=True)
    data_path = os.path.join(data_dir, f"data{size}.txt")

    # Save to file
    with open(data_path, 'w') as f:
        f.write(libsvm_data)

    if encrypt:
        xgb.encrypt_file(data_path, os.path.join(data_dir, f"data{size}.enc"), KEY_FILE)

# Function to preprocess Higgs dataset
def preprocess_higgs(data, size, filename=None, encrypt=True, scaler_type='minmax'):
    dataset = "higgs"
    # Sample the data
    global random_state
    data_sampled = data.sample(n=size, random_state=random_state)
    random_state += 1  # Increment random state for reproducibility in subsequent calls
    
    # Split data into features and labels
    X = data_sampled.drop("label", axis=1)
    y = data_sampled["label"]
    
    # Scale features
    if scaler_type == 'minmax':
        scaler = MinMaxScaler()
    elif scaler_type == 'standard':
        scaler = StandardScaler()
    else:
        raise ValueError("Unsupported scaler_type. Choose 'minmax' or 'standard'.")
    
    X_scaled = scaler.fit_transform(X)
    
    # Convert to sparse format and save to text files
    sparse_data = convert_to_sparse_format(pd.DataFrame(X_scaled), y.reset_index(drop=True))
    
    if filename is None:
        filename = f"data{size}"
    data_dir = os.path.join(CURRENT_DIR, 'data/higgs')
    os.makedirs(data_dir, exist_ok=True)
    data_path = os.path.join(data_dir, filename + ".txt")
    with open(data_path, "w") as train_file:
        train_file.write("\n".join(sparse_data))

    if encrypt:
        # Encrypt the file
        xgb.encrypt_file(data_path, os.path.join(data_dir, filename + ".enc"), KEY_FILE)

# Function to convert data to LibSVM format
def convert_to_libsvm(X, y=None):
    libsvm_str = ""
    for i in range(X.shape[0]):
        if y is not None:
            libsvm_str += f"{y.iloc[i]} "
        for j in range(X.shape[1]):
            if X.iloc[i, j] != 0:  # Only store non-zero values
                libsvm_str += f"{j+1}:{X.iloc[i, j]} "
        libsvm_str = libsvm_str.strip() + "\n"
    return libsvm_str

# Main function to handle dataset processing
def process_dataset(dataset, size, iterations=None, encrypt=True):
    if dataset == 'allstate':
        data = pd.read_csv('data/allstate/train_set.csv', dtype={19: str}, low_memory=False)
        preprocess_allstate(data, size, encrypt=encrypt)
    elif dataset == 'covtype':
        data = pd.read_csv('data/covtype/covtype.data', header=None)
        preprocess_covtype(data, size, encrypt=encrypt)
    elif dataset == 'higgs':
        data_path = os.path.join(CURRENT_DIR, 'data/higgs/HIGGS.csv')
        data = pd.read_csv(data_path, header=None, names=higgs_column_names)
        # preprocess_higgs(data, size)
        if iterations is not None:
            for i in range(iterations):
                preprocess_higgs(data, size, filename=f"data{size}_iter{i}", encrypt=encrypt)
        else:
            preprocess_higgs(data, size, encrypt=encrypt)
    else:
        raise ValueError("Unsupported dataset. Choose 'allstate', 'covtype', or 'higgs'.")


def parse_sizes(size=None, sizes=None):
    if sizes:
        return [int(item.strip()) for item in sizes.split(',') if item.strip()]
    if size is not None:
        return [size]
    raise ValueError("Either --size or --sizes must be provided.")

def test():
    print("Testing dataset processing...")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Process datasets.')
    parser.add_argument('--dataset', type=str, default=DEFAULT_DATASET, choices=['allstate', 'covtype', 'higgs'],
                        help='Dataset to process: allstate, covtype, or higgs')
    parser.add_argument('--size', type=int, default=None, help='One dataset size to process')
    parser.add_argument('--sizes', type=str, default=None,
                        help='Comma-separated dataset sizes, e.g. 1000,2000,3000')
    parser.add_argument('--iterations', type=int, default=None, help='Number of iterations for higgs dataset')
    parser.add_argument('--no-encrypt', action='store_true',
                        help='Only generate .txt files and skip .enc encryption')
    args = parser.parse_args()

    if args.size is None and args.sizes is None and args.dataset == DEFAULT_DATASET:
        size_list = DEFAULT_SIZE_LIST
    else:
        size_list = parse_sizes(size=args.size, sizes=args.sizes)

    for size in size_list:
        process_dataset(
            args.dataset,
            size,
            iterations=args.iterations,
            encrypt=not args.no_encrypt,
        )
        print(f"Dataset {args.dataset} with size {size} has been processed.")
