import os
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler, StandardScaler
import securexgboost as xgb
from utils import *

# Define column names based on the dataset description
higggs_column_names = ["label", "lepton_pT", "lepton_eta", "lepton_phi", "missing_energy_magnitude", "missing_energy_phi",
                "jet1_pt", "jet1_eta", "jet1_phi", "jet1_b-tag",
                "jet2_pt", "jet2_eta", "jet2_phi", "jet2_b-tag",
                "jet3_pt", "jet3_eta", "jet3_phi", "jet3_b-tag",
                "jet4_pt", "jet4_eta", "jet4_phi", "jet4_b-tag",
                "m_jj", "m_jjj", "m_lv", "m_jlv", "m_bb", "m_wbb", "m_wwbb"]

def preprocess_higgs_data(data_dir, raw_data, out_data_size=None, column_names=None, scaler_type='minmax', random_state=42, verbose=True):
    """
    Preprocess the Higgs dataset.
    
    Parameters:
    - raw_data: Path to the raw data file
    - processed_data_path: Path to save the processed data
    - sparse_format: Whether to save data in sparse format
    - sample_frac: Fraction of data to sample
    - test_size: Proportion of the dataset to include in the test split
    - scaler_type: Type of feature scaling ('minmax' or 'standard')
    - random_state: Random seed
    - verbose: Whether to print detailed logs
    """
    # Ensure the output directory exists
    os.makedirs(data_dir, exist_ok=True)

    if verbose:
        print('Reading data...')
    data = pd.read_csv(os.path.join(data_dir, raw_data), header=None, names=column_names)

    
    if scaler_type == 'minmax':
        scaler = MinMaxScaler()
    elif scaler_type == 'standard':
        scaler = StandardScaler()
    else:
        raise ValueError("Unsupported scaler_type. Choose 'minmax' or 'standard'.")
    
    for n in out_data_size:
        # Sample a fraction of the data
        data_sampled = data.sample(n=n, random_state=random_state)

        # Split data into features and labels
        X = data_sampled.drop("label", axis=1)
        y = data_sampled["label"]

        # Scale features
        if verbose:
            print('Scaling data...')
        
        X_scaled = scaler.fit_transform(X)
            
        # Convert to sparse format and save to text files
        sparse_data = convert_to_sparse_format(pd.DataFrame(X_scaled), y.reset_index(drop=True))

        data_path = os.path.join(data_dir, f"data{n}.txt")

        with open(data_path, "w") as train_file:
            train_file.write("\n".join(sparse_data))

        if verbose:
            print("Sparse data saved.")

        xgb.encrypt_file(data_path, os.path.join(data_dir, f"data{n}.enc"), KEY_FILE)

    if verbose:
        print("Data preprocessing complete.")

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

if __name__ == "__main__":
    # preprocess_higgs_data(
    #     data_dir=HIGGS_DIR, 
    #     raw_data="HIGGS.csv", 
    #     out_data_size=[1000, 10000, 100000],
    #     column_names=higggs_column_names,
    #     scaler_type='minmax',
    #     random_state=42,
    #     verbose=True
    # )
    for n in [1000, 10000, 100000]:
        xgb.encrypt_file(os.path.join(DEMO_DIR, f"data{n}.txt"), os.path.join(DEMO_DIR, f"data{n}.enc"), KEY_FILE)
