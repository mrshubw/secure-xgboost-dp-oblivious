/*!
 * Copyright 2019 XGBoost contributors
 * Modifications copyright 2020 Secure XGBoost contributors
 *
 * \file c-api-demo.c
 * \brief A simple example of using xgboost C API.
 *
 * Note: This demo requires the library to be built with the build parameter CONSENSUS=OFF.
 */

#include <iostream>
#include <fstream>
#include <stdio.h>
#include <stdlib.h>
#include <unistd.h>
#include <xgboost/c_api.h>

#include <enclave/crypto.h>
#include <openenclave/host.h>

static char test_key[CIPHER_KEY_SIZE] = {
    0, 0, 0, 0, 0, 0, 0, 0,
    0, 0, 0, 0, 0, 0, 0, 0,
    0, 0, 0, 0, 0, 0, 0, 0,
    0, 0, 0, 0, 0, 0, 0, 0
};

#define safe_xgboost(call) {                                            \
int err = (call);                                                       \
if (err != 0) {                                                         \
  fprintf(stderr, "%s:%d: error in %s: %s\n", __FILE__, __LINE__, #call, XGBGetLastError()); \
  exit(1);                                                              \
}                                                                       \
}

int main(int argc, char** argv) {

  for (int i = 0; i < argc; i++) {
    if (strcmp(argv[i], "--encrypt") == 0) {
      encrypt_file_with_keybuf("../data/agaricus.txt.train", "train.encrypted", test_key);
      encrypt_file_with_keybuf("../data/agaricus.txt.test", "test.encrypted", test_key);
      return 0;
    }
  }

  std::cout << "Creating enclave\n";
  int log_verbosity = 1;
  char* usernames[1] = {"user1"};
  safe_xgboost(XGBCreateEnclave(argv[1], usernames, 1, log_verbosity));
  
  oe_result_t result;
  int ret = 1;
  oe_enclave_t* enclave = NULL;

  uint8_t* pem_key = NULL;
  size_t key_size = 0;
  uint8_t* nonce = NULL;
  size_t nonce_size = 0;
  uint8_t* remote_report = NULL;
  size_t remote_report_size = 0;

  size_t size;
	char *path=NULL;
	path = getcwd(path,size);
	std::string cwd(path);
  std::string fname1(cwd + "/../data/agaricus.txt.train.enc");
  std::string fname2(cwd + "/../data/agaricus.txt.test.enc");

  safe_xgboost(get_remote_report_with_pubkey_and_nonce(&pem_key, &key_size, &nonce, &nonce_size, &remote_report, &remote_report_size));
  // NOTE: Verification will fail in simulation mode
  // Comment out this line for testing the code in simulation mode
  safe_xgboost(verify_remote_report_and_set_pubkey_and_nonce(pem_key, key_size, nonce, nonce_size, remote_report, remote_report_size));

  uint8_t* encrypted_data = (uint8_t*) malloc(1024*sizeof(uint8_t));
  size_t encrypted_data_size = 1024;
  uint8_t* signature = (uint8_t*) malloc(1024*sizeof(uint8_t));
  size_t sig_len;

  safe_xgboost(encrypt_data_with_pk(test_key, CIPHER_KEY_SIZE, pem_key, key_size, encrypted_data, &encrypted_data_size));

  std::string pubkey(cwd + "/../../config/user1.pem");
  std::string crt_file(cwd + "/../../config/user1.crt");
  std::ifstream ifs(crt_file);
  std::string crt((std::istreambuf_iterator<char>(ifs)),
                       (std::istreambuf_iterator<char>()));

  safe_xgboost(sign_data_with_keyfile((char*)pubkey.c_str(), encrypted_data, encrypted_data_size, signature, &sig_len));

  std::cout << "Adding client key\n";
  safe_xgboost(add_client_key_with_certificate((char*)crt.c_str(), crt.length() + 1, encrypted_data, encrypted_data_size, signature, sig_len));

  int silent = 1;
  
  // load the data
  const char* fnames1[2] = {fname1.c_str(), fname1.c_str()};
  const char* fnames2[1] = {fname2.c_str()};
  char* unames[2] = {"user1", "user1"};
  DMatrixHandle dtrain, dtest;
  std::cout << "Loading train data\n";
  safe_xgboost(XGDMatrixCreateFromEncryptedFile(fnames1, unames, 2, silent, &dtrain));
  //safe_xgboost(XGDMatrixCreateFromFile(fname1.c_str(), "user1", silent, &dtrain));
  std::cout << "Loading test data\n";
  safe_xgboost(XGDMatrixCreateFromEncryptedFile(fnames2, unames, 1, silent, &dtest));
  //safe_xgboost(XGDMatrixCreateFromFile(fname2.c_str(), "user1", silent, &dtest));
  std::cout << "Data loaded" << std::endl;

  // create the booster
  BoosterHandle booster;
  DMatrixHandle eval_dmats[2] = {dtrain, dtest};
  safe_xgboost(XGBoosterCreate(eval_dmats, 2, &booster));
  std::cout << "Booster created" << std::endl;

  // configure the training
  // available parameters are described here:
  // https://xgboost.readthedocs.io/en/latest/parameter.html
  safe_xgboost(XGBoosterSetParam(booster, "tree_method", "hist"));
  safe_xgboost(XGBoosterSetParam(booster, "objective", "binary:logistic"));
  safe_xgboost(XGBoosterSetParam(booster, "min_child_weight", "1"));
  safe_xgboost(XGBoosterSetParam(booster, "gamma", "0.1"));
  safe_xgboost(XGBoosterSetParam(booster, "max_depth", "3"));
  safe_xgboost(XGBoosterSetParam(booster, "verbosity", silent ? "0" : "3"));
  std::cout << "All parameters set" << std::endl;
  
  // train and evaluate for 10 iterations
  int n_trees = 10;
  const char* eval_names[2] = {"train", "test"};
  const char* eval_result = NULL;
  for (int i = 0; i < n_trees; ++i) {
    safe_xgboost(XGBoosterUpdateOneIter(booster, i, dtrain));
    //safe_xgboost(XGBoosterEvalOneIter(booster, i, eval_dmats, eval_names, 2, &eval_result));
    //printf("%s\n", eval_result);
  }
  std::cout << "Training done" << std::endl;
  
  // save model
  std::string fname(cwd + "/demo_model.model");
  safe_xgboost(XGBoosterSaveModel(booster, fname.c_str()));
  std::cout << "Saved model to demo_model.model" << std::endl;

  // load model
  booster = NULL;
  safe_xgboost(XGBoosterCreate(eval_dmats, 2, &booster));
  safe_xgboost(XGBoosterLoadModel(booster, fname.c_str()));
  std::cout << "Loaded model from demo_model.model" << std::endl;

  // predict
  bst_ulong out_len = 0;
  uint8_t* enc_result = NULL;
  float* out_result = NULL;
  int n_print = 10;
  
  std::cout << "Running predictions" << std::endl;
  safe_xgboost(XGBoosterPredict(booster, dtest, 0, 0, 0, &out_len, &enc_result));
  safe_xgboost(decrypt_predictions(test_key, enc_result, out_len, &out_result));
  printf("n_pred: %d %x\n", out_len, out_result);
  printf("y_pred: ");
  for (int i = 0; i < n_print; ++i) {
    printf("%1.4f ", out_result[i]);
  }
  printf("\n");
  
  safe_xgboost(XGDMatrixGetFloatInfo(dtest, "label", &out_len, (const float**)&out_result));
  printf("y_test: ");
  for (int i = 0; i < n_print; ++i) {
    printf("%1.4f ", out_result[i]);
  }
  printf("\n");
  
  safe_xgboost(XGBoosterFree(booster));
  safe_xgboost(XGDMatrixFree(dtrain));
  safe_xgboost(XGDMatrixFree(dtest));
  return 0;
}
