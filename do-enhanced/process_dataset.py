import os
import argparse
import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder, MinMaxScaler, StandardScaler
import securexgboost as xgb
from utils import *

# Define column names for Higgs dataset
higgs_column_names = ["label", "lepton_pT", "lepton_eta", "lepton_phi", "missing_energy_magnitude", "missing_energy_phi",
                      "jet1_pt", "jet1_eta", "jet1_phi", "jet1_b-tag",
                      "jet2_pt", "jet2_eta", "jet2_phi", "jet2_b-tag",
                      "jet3_pt", "jet3_eta", "jet3_phi", "jet3_b-tag",
                      "jet4_pt", "jet4_eta", "jet4_phi", "jet4_b-tag",
                      "m_jj", "m_jjj", "m_lv", "m_jlv", "m_bb", "m_wbb", "m_wwbb"]

# Function to preprocess Allstate dataset
def preprocess_allstate(data, size):
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
    
    # Save to file
    with open(f'data/allstate/data{size}.txt', 'w') as f:
        f.write(libsvm_train)

# Function to preprocess Covtype dataset
def preprocess_covtype(data, size):
    # Convert labels to be in the range [0, num_class)
    data.iloc[:, -1] = data.iloc[:, -1] - 1
    
    # Sample the data
    data_sampled = data.sample(n=size, random_state=42)
    
    # Convert to LibSVM format
    libsvm_data = convert_to_libsvm(data_sampled.iloc[:, :-1], data_sampled.iloc[:, -1])
    
    # Save to file
    with open(f'data/covtype/data{size}.txt', 'w') as f:
        f.write(libsvm_data)

# Function to preprocess Higgs dataset
def preprocess_higgs(data, size, scaler_type='minmax'):
    # Sample the data
    data_sampled = data.sample(n=size, random_state=42)
    
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
    
    data_path = os.path.join('data/higgs', f"data{size}.txt")
    with open(data_path, "w") as train_file:
        train_file.write("\n".join(sparse_data))

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
def process_dataset(dataset, size):
    if dataset == 'allstate':
        data = pd.read_csv('data/allstate/train_set.csv', dtype={19: str}, low_memory=False)
        preprocess_allstate(data, size)
    elif dataset == 'covtype':
        data = pd.read_csv('data/covtype/covtype.data', header=None)
        preprocess_covtype(data, size)
    elif dataset == 'higgs':
        data = pd.read_csv('data/higgs/HIGGS.csv', header=None, names=higgs_column_names)
        preprocess_higgs(data, size)
    else:
        raise ValueError("Unsupported dataset. Choose 'allstate', 'covtype', or 'higgs'.")
    
    
    data_path = os.path.join('data/' + dataset, f"data{size}.txt")
    # Encrypt the file
    xgb.encrypt_file(data_path, os.path.join('data/' + dataset, f"data{size}.enc"), KEY_FILE)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Process datasets.')
    parser.add_argument('--dataset', type=str, required=True, choices=['allstate', 'covtype', 'higgs'],
                        help='Dataset to process: allstate, covtype, or higgs')
    parser.add_argument('--size', type=int, required=True, help='Size of the dataset to process')
    args = parser.parse_args()
    
    process_dataset(args.dataset, args.size)
    print(f"Dataset {args.dataset} with size {args.size} has been processed.")