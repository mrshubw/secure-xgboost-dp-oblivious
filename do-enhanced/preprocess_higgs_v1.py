import os
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler, StandardScaler
import securexgboost as xgb
from utils import *

def preprocess_higgs_data(raw_data, processed_data_dir, sparse_format=False, sample_frac=0.001, test_size=0.2, scaler_type='minmax', random_state=42, verbose=True):
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
    os.makedirs(processed_data_dir, exist_ok=True)

    # Define column names based on the dataset description
    column_names = ["label", "lepton_pT", "lepton_eta", "lepton_phi", "missing_energy_magnitude", "missing_energy_phi",
                    "jet1_pt", "jet1_eta", "jet1_phi", "jet1_b-tag",
                    "jet2_pt", "jet2_eta", "jet2_phi", "jet2_b-tag",
                    "jet3_pt", "jet3_eta", "jet3_phi", "jet3_b-tag",
                    "jet4_pt", "jet4_eta", "jet4_phi", "jet4_b-tag",
                    "m_jj", "m_jjj", "m_lv", "m_jlv", "m_bb", "m_wbb", "m_wwbb"]

    if verbose:
        print('Reading data...')
    data = pd.read_csv(raw_data, header=None, names=column_names)

    # Sample a fraction of the data
    data_sampled = data.sample(frac=sample_frac, random_state=random_state)

    # Split data into features and labels
    X = data_sampled.drop("label", axis=1)
    y = data_sampled["label"]

    # Split the data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=random_state)

    # Scale features
    if verbose:
        print('Scaling data...')
    
    if scaler_type == 'minmax':
        scaler = MinMaxScaler()
    elif scaler_type == 'standard':
        scaler = StandardScaler()
    else:
        raise ValueError("Unsupported scaler_type. Choose 'minmax' or 'standard'.")
    
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
        
    if sparse_format:
        # Convert to sparse format and save to text files
        train_sparse = convert_to_sparse_format(pd.DataFrame(X_train_scaled), y_train.reset_index(drop=True))
        test_sparse = convert_to_sparse_format(pd.DataFrame(X_test_scaled), y_test.reset_index(drop=True))

        train_sparse_path = os.path.join(processed_data_dir, TRAIN_SPARSE)
        test_sparse_path = os.path.join(processed_data_dir, TEST_SPARSE)

        with open(train_sparse_path, "w") as train_file:
            train_file.write("\n".join(train_sparse))

        with open(test_sparse_path, "w") as test_file:
            test_file.write("\n".join(test_sparse))

        if verbose:
            print("Sparse data saved.")
    else:
        # Save the processed data to CSV without the header
        train_data = pd.DataFrame(X_train_scaled, columns=column_names[1:])
        train_data["label"] = y_train.reset_index(drop=True)

        test_data = pd.DataFrame(X_test_scaled, columns=column_names[1:])
        test_data["label"] = y_test.reset_index(drop=True)

        train_csv_path = os.path.join(processed_data_dir, TRAIN_PROCESSED)
        test_csv_path = os.path.join(processed_data_dir, TEST_PROCESSED)

        train_data.to_csv(train_csv_path, index=False, header=False)
        test_data.to_csv(test_csv_path, index=False, header=False)

        if verbose:
            print("CSV data saved.")

    encrypt_file(processed_data_dir, TRAIN_SPARSE if sparse_format else TRAIN_PROCESSED, TEST_SPARSE if sparse_format else TEST_PROCESSED, key_file=KEY_FILE, verbose=verbose)

    if verbose:
        print("Data preprocessing complete.")

def encrypt_file(data_dir, train_file, test_file, key_file, verbose=True):
    """
    Encrypt the processed data files.
    
    Parameters:
    - data_dir: Path to save the processed data
    - train_file: Filename of the training data
    - test_file: Filename of the testing data
    - key_file: Encryption key file
    - verbose: Whether to print detailed logs
    """
    train_path = os.path.join(data_dir, train_file)
    test_path = os.path.join(data_dir, test_file)
    xgb.encrypt_file(train_path, os.path.join(data_dir, TRAIN_ENC), key_file)
    xgb.encrypt_file(test_path, os.path.join(data_dir, TEST_ENC), key_file)
    if verbose:
        print("Data encrypted.")

if __name__ == "__main__":
    preprocess_higgs_data(
        raw_data=RAW_HIGGS_FILE, 
        processed_data_dir=HIGGS_DIR, 
        sparse_format=True,
        sample_frac=0.001,
        test_size=0.2,
        scaler_type='minmax',
        random_state=42,
        verbose=True
    )
