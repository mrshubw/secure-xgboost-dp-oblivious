### Directory Structure:

- data: Contains model files and datasets

- data/higgs/modeld8n5.model: Model file for the Higgs dataset, with depth 8 and 5 trees. It has been encrypted using key.txt.

- data/higgs/data1000.txt: 1000 Higgs data samples in plaintext, using the libsvm format.

- data/higgs/data1000.enc: Encrypted version of data1000.txt, using the key in key.txt.

- key.txt: The encryption key file.

- predict.py: Script for running the prediction test.

### Testing Steps:

1. After extracting the files, place them in the secure xgboost project directory.
Then modify the HOME_DIR variable in predict.py according to its relative path inside the project.

1. Open the enclave/include/enclave_context.h file in the secure xgboost project, and modify the generate_symm_key function as follows:

    ```cpp
    bool generate_symm_key() {
        // generate_random(m_symm_key, CIPHER_KEY_SIZE);
        
        // In the DP Oblivious experiment, in order to separate training from inference, the key is forced to be 0. 
        // Notice, this is not secure.
        memset(m_symm_key, 0, CIPHER_KEY_SIZE);
    }
    ```
    This change is necessary because the original generate_symm_key function generates a different key each time it runs, making it impossible to decrypt the model during inference.
    For convenience, the key is fixed to zero here to enable prediction using pre-trained models.

1. Run predict.py.