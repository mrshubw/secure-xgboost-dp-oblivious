/*!
 * Copyright (c) 2015 by Contributors
 * Modifications Copyright (c) 2020-22 by Secure XGBoost Contributors
 * \file c_api.h
 * \author Tianqi Chen
 * \brief C API of XGBoost, used for interfacing to other languages.
 */
#ifndef XGBOOST_C_API_H_
#define XGBOOST_C_API_H_

#ifdef __cplusplus
#define XGB_EXTERN_C extern "C"
#include <cstdio>
#include <cstdint>
#include <string>
#else
#define XGB_EXTERN_C
#include <stdio.h>
#include <stdint.h>
#endif  // __cplusplus

#if defined(_MSC_VER) || defined(_WIN32)
#define XGB_DLL XGB_EXTERN_C __declspec(dllexport)
#else
#define XGB_DLL XGB_EXTERN_C
#endif  // defined(_MSC_VER) || defined(_WIN32)

#ifdef __ENCLAVE__ // macros for errors / safety checks
#define safe_ocall(call) {                                \
oe_result_t result = (call);                              \
if (result != OE_OK) {                                    \
  fprintf(stderr,                                         \
      "%s:%d: Ocall failed; error in %s: %s\n",           \
      __FILE__, __LINE__, #call, oe_result_str(result));  \
  exit(1);                                                \
}                                                         \
}

#define check_enclave_buffer(ptr, size) {                 \
if (!oe_is_within_enclave((ptr), size)) {                 \
    fprintf(stderr,                                       \
            "%s:%d: Buffer bounds check failed\n",        \
            __FILE__, __LINE__);                          \
    exit(1);                                              \
}                                                         \
}

#define check_host_buffer(ptr, size) {                    \
if (!oe_is_outside_enclave((ptr), size)) {                \
    fprintf(stderr,                                       \
            "%s:%d: Buffer bounds check failed\n",        \
            __FILE__, __LINE__);                          \
    exit(1);                                              \
}                                                         \
}
#endif

// manually define unsigned long
typedef uint64_t bst_ulong;  // NOLINT(*)

// FIXME added this here, but perhaps not necessary
typedef float bst_float;  // NOLINT(*)


/*! \brief handle to DMatrix */
typedef char* DMatrixHandle;  // NOLINT(*)
/*! \brief handle to Booster */
typedef char* BoosterHandle;  // NOLINT(*)

/*!
 * \brief Return the version of the XGBoost library being currently used.
 *
 *  The output variable is only written if it's not NULL.
 *
 * \param major Store the major version number
 * \param minor Store the minor version number
 * \param patch Store the patch (revision) number
 */

XGB_DLL void XGBoostVersion(int* major, int* minor, int* patch);
/*!
 * \brief get string message of the last error
 *
 *  all function in this file will return 0 when success
 *  and -1 when an error occurred,
 *  XGBGetLastError can be called to retrieve the error
 *
 *  this function is thread safe and can be called by different thread
 * \return const char* error information
 */
XGB_DLL const char *XGBGetLastError(void);

/*!
 * \brief register callback function for LOG(INFO) messages -- helpful messages
 *        that are not errors.
 * Note: this function can be called by multiple threads. The callback function
 *       will run on the thread that registered it
 * \return 0 for success, -1 for failure
 */
XGB_DLL int XGBRegisterLogCallback(void (*callback)(const char*));

#if defined(__HOST__)
XGB_DLL int XGBCreateEnclave(const char *enclave_image, char** usernames, size_t num_clients, int log_verbosity);
#endif

/*!
 * \brief load a data matrix
 * \param fname the name of the file
 * \param silent whether print messages during loading
 * \param out a loaded data matrix
 * \return 0 when success, -1 when failure happens
 */
XGB_DLL int XGDMatrixCreateFromFile(const char *fname,
                                    int silent,
                                    DMatrixHandle *out);

/*!
 * \brief load a data matrix from an encrypted file
 * \param fname the name of the encrypted file
 * \param silent whether print messages during loading
 * \param nonce nonce received from the enclave during initialization
 * \param nonce_size size in bytes of nonce
 * \param nonce_ctr incrementing counter used to indicate sequence number of API call
 * \param out a loaded data matrix
 * \param out_sig signature over the output and nonce
 * \param out_sig_length length of output signature
 * \param signers list of usernames of signing clients
 * \param signatures list of client signatures
 * \param sig_lengths list of signature lengths
 * \return 0 when success, -1 when failure happens
 */
XGB_DLL int XGDMatrixCreateFromEncryptedFile(const char *fnames[],
                                             char* usernames[],
                                             bst_ulong num_files,
                                             int silent,
                                             uint8_t* nonce,
                                             size_t nonce_size,
                                             uint32_t nonce_ctr,
                                             DMatrixHandle *out,
                                             uint8_t** out_sig,
                                             size_t *out_sig_length,
                                             char **signers,
                                             uint8_t* signatures[],
                                             size_t* sig_lengths);

/*!
 * \brief create a matrix content from CSR format
 * \param indptr pointer to row headers
 * \param indices findex
 * \param data fvalue
 * \param nindptr number of rows in the matrix + 1
 * \param nelem number of nonzero elements in the matrix
 * \param num_col number of columns; when it's set to 0, then guess from data
 * \param out created dmatrix
 * \return 0 when success, -1 when failure happens
 */
XGB_DLL int XGDMatrixCreateFromCSREx(const size_t* indptr,
                                     const unsigned* indices,
                                     const float* data,
                                     size_t nindptr,
                                     size_t nelem,
                                     size_t num_col,
                                     DMatrixHandle* out);
/*!
 * \brief create a matrix content from CSC format
 * \param col_ptr pointer to col headers
 * \param indices findex
 * \param data fvalue
 * \param nindptr number of rows in the matrix + 1
 * \param nelem number of nonzero elements in the matrix
 * \param num_row number of rows; when it's set to 0, then guess from data
 * \param out created dmatrix
 * \return 0 when success, -1 when failure happens
 */
XGB_DLL int XGDMatrixCreateFromCSCEx(const size_t* col_ptr,
                                     const unsigned* indices,
                                     const float* data,
                                     size_t nindptr,
                                     size_t nelem,
                                     size_t num_row,
                                     DMatrixHandle* out);

/*!
 * \brief create matrix content from dense matrix
 * \param data pointer to the data space
 * \param nrow number of rows
 * \param ncol number columns
 * \param missing which value to represent missing value
 * \param out created dmatrix
 * \return 0 when success, -1 when failure happens
 */
XGB_DLL int XGDMatrixCreateFromMat(const float *data,
                                   bst_ulong nrow,
                                   bst_ulong ncol,
                                   float missing,
                                   DMatrixHandle *out);
/*!
 * \brief create matrix content from dense matrix
 * \param data pointer to the data space
 * \param nrow number of rows
 * \param ncol number columns
 * \param missing which value to represent missing value
 * \param out created dmatrix
 * \param nthread number of threads (up to maximum cores available, if <=0 use all cores)
 * \return 0 when success, -1 when failure happens
 */
XGB_DLL int XGDMatrixCreateFromMat_omp(const float *data,  // NOLINT
                                       bst_ulong nrow, bst_ulong ncol,
                                       float missing, DMatrixHandle *out,
                                       int nthread);
/*!
 * \brief create matrix content from python data table
 * \param data pointer to pointer to column data
 * \param feature_stypes pointer to strings
 * \param nrow number of rows
 * \param ncol number columns
 * \param out created dmatrix
 * \param nthread number of threads (up to maximum cores available, if <=0 use all cores)
 * \return 0 when success, -1 when failure happens
 */
XGB_DLL int XGDMatrixCreateFromDT(void** data,
                                  const char ** feature_stypes,
                                  bst_ulong nrow,
                                  bst_ulong ncol,
                                  DMatrixHandle* out,
                                  int nthread);
/*
 * ========================== Begin data callback APIs =========================
 *
 * Short notes for data callback
 *
 * There are 2 sets of data callbacks for DMatrix.  The first one is currently exclusively
 * used by JVM packages.  It uses `XGBoostBatchCSR` to accept batches for CSR formated
 * input, and concatenate them into 1 final big CSR.  The related functions are:
 *
 * - XGBCallbackSetData
 * - XGBCallbackDataIterNext
 * - XGDMatrixCreateFromDataIter
 *
 * Another set is used by Quantile based DMatrix (used by hist algorithm) for reducing
 * memory usage.  Currently only GPU implementation is available.  It accept foreign data
 * iterators as callbacks and works similar to external memory.  For GPU Hist, the data is
 * first compressed by quantile sketching then merged.  This is particular useful for
 * distributed setting as it eliminates 2 copies of data.  1 by a `concat` from external
 * library to make the data into a blob for normal DMatrix initialization, another by the
 * internal CSR copy of DMatrix.  Related functions are:
 *
 * - XGProxyDMatrixCreate
 * - XGDMatrixCallbackNext
 * - DataIterResetCallback
 * - XGDeviceQuantileDMatrixSetDataCudaArrayInterface
 * - XGDeviceQuantileDMatrixSetDataCudaColumnar
 * - ... (data setters)
 */

/*  ==== First set of callback functions, used exclusively by JVM packages. ==== */

/*! \brief handle to a external data iterator */
typedef void *DataIterHandle;  // NOLINT(*)
/*! \brief handle to a internal data holder. */
typedef void *DataHolderHandle;  // NOLINT(*)


/*! \brief Mini batch used in XGBoost Data Iteration */
typedef struct {  // NOLINT(*)
  /*! \brief number of rows in the minibatch */
  size_t size;
  /* \brief number of columns in the minibatch. */
  size_t columns;
  /*! \brief row pointer to the rows in the data */
#ifdef __APPLE__
  /* Necessary as Java on MacOS defines jlong as long int
   * and gcc defines int64_t as long long int. */
  long* offset; // NOLINT(*)
#else
  int64_t* offset;  // NOLINT(*)
#endif  // __APPLE__
  /*! \brief labels of each instance */
  float* label;
  /*! \brief weight of each instance, can be NULL */
  float* weight;
  /*! \brief feature index */
  int* index;
  /*! \brief feature values */
  float* value;
} XGBoostBatchCSR;

/*!
 * \brief Callback to set the data to handle,
 * \param handle The handle to the callback.
 * \param batch The data content to be set.
 */
XGB_EXTERN_C typedef int XGBCallbackSetData(  // NOLINT(*)
    DataHolderHandle handle, XGBoostBatchCSR batch);

/*!
 * \brief The data reading callback function.
 *  The iterator will be able to give subset of batch in the data.
 *
 *  If there is data, the function will call set_function to set the data.
 *
 * \param data_handle The handle to the callback.
 * \param set_function The batch returned by the iterator
 * \param set_function_handle The handle to be passed to set function.
 * \return 0 if we are reaching the end and batch is not returned.
 */
XGB_EXTERN_C typedef int XGBCallbackDataIterNext(  // NOLINT(*)
    DataIterHandle data_handle, XGBCallbackSetData *set_function,
    DataHolderHandle set_function_handle);

/*!
 * \brief Create a DMatrix from a data iterator.
 * \param data_handle The handle to the data.
 * \param callback The callback to get the data.
 * \param cache_info Additional information about cache file, can be null.
 * \param out The created DMatrix
 * \return 0 when success, -1 when failure happens.
 */
XGB_DLL int XGDMatrixCreateFromDataIter(
    DataIterHandle data_handle,
    XGBCallbackDataIterNext* callback,
    const char* cache_info,
    DMatrixHandle *out);

/*  == Second set of callback functions, used by constructing Quantile based DMatrix. ===
 *
 * Short note for how to use the second set of callback for GPU Hist tree method.
 *
 * Step 0: Define a data iterator with 2 methods `reset`, and `next`.
 * Step 1: Create a DMatrix proxy by `XGProxyDMatrixCreate` and hold the handle.
 * Step 2: Pass the iterator handle, proxy handle and 2 methods into
 *         `XGDeviceQuantileDMatrixCreateFromCallback`.
 * Step 3: Call appropriate data setters in `next` functions.
 *
 * See test_iterative_device_dmatrix.cu or Python interface for examples.
 */

/*!
 * \brief Create a DMatrix proxy for setting data, can be free by XGDMatrixFree.
 *
 * \param out      The created Device Quantile DMatrix
 *
 * \return 0 when success, -1 when failure happens
 */
XGB_DLL int XGProxyDMatrixCreate(DMatrixHandle* out);

/*!
 * \brief Callback function prototype for getting next batch of data.
 *
 * \param iter  A handler to the user defined iterator.
 *
 * \return 0 when success, -1 when failure happens
 */
XGB_EXTERN_C typedef int XGDMatrixCallbackNext(DataIterHandle iter);  // NOLINT(*)

/*!
 * \brief Callback function prototype for reseting external iterator
 */
XGB_EXTERN_C typedef void DataIterResetCallback(DataIterHandle handle); // NOLINT(*)

/*!
 * \brief Create a device DMatrix with data iterator.
 *
 * \param iter     A handle to external data iterator.
 * \param proxy    A DMatrix proxy handle created by `XGProxyDMatrixCreate`.
 * \param reset    Callback function reseting the iterator state.
 * \param next     Callback function yieling the next batch of data.
 * \param missing  Which value to represent missing value
 * \param nthread  Number of threads to use, 0 for default.
 * \param max_bin  Maximum number of bins for building histogram.
 * \param out      The created Device Quantile DMatrix
 *
 * \return 0 when success, -1 when failure happens
 */
XGB_DLL int XGDeviceQuantileDMatrixCreateFromCallback(
    DataIterHandle iter, DMatrixHandle proxy, DataIterResetCallback *reset,
    XGDMatrixCallbackNext *next, float missing, int nthread, int max_bin,
    DMatrixHandle *out);
/*!
 * \brief Set data on a DMatrix proxy.
 *
 * \param handle          A DMatrix proxy created by XGProxyDMatrixCreate
 * \param c_interface_str Null terminated JSON document string representation of CUDA
 *                        array interface.
 *
 * \return 0 when success, -1 when failure happens
 */
XGB_DLL int XGDeviceQuantileDMatrixSetDataCudaArrayInterface(
    DMatrixHandle handle,
    const char* c_interface_str);
/*!
 * \brief Set data on a DMatrix proxy.
 *
 * \param handle          A DMatrix proxy created by XGProxyDMatrixCreate
 * \param c_interface_str Null terminated JSON document string representation of CUDA
 *                        array interface, with an array of columns.
 *
 * \return 0 when success, -1 when failure happens
 */
XGB_DLL int XGDeviceQuantileDMatrixSetDataCudaColumnar(
    DMatrixHandle handle,
    const char* c_interface_str);
/*
 * ==========================- End data callback APIs ==========================
 */



/*!
 * \brief create a new dmatrix from sliced content of existing matrix
 * \param handle instance of data matrix to be sliced
 * \param idxset index set
 * \param len length of index set
 * \param out a sliced new matrix
 * \return 0 when success, -1 when failure happens
 */
XGB_DLL int XGDMatrixSliceDMatrix(DMatrixHandle handle,
    const int *idxset,
    bst_ulong len,
    DMatrixHandle *out);
/*!
 * \brief create a new dmatrix from sliced content of existing matrix
 * \param handle instance of data matrix to be sliced
 * \param idxset index set
 * \param len length of index set
 * \param out a sliced new matrix
 * \param allow_groups allow slicing of an array with groups
 * \return 0 when success, -1 when failure happens
 */
XGB_DLL int XGDMatrixSliceDMatrixEx(DMatrixHandle handle,
    const int *idxset,
    bst_ulong len,
    DMatrixHandle *out,
    int allow_groups);

/*!
 * \brief free space in data matrix
 * \return 0 when success, -1 when failure happens
 */
XGB_DLL int XGDMatrixFree(DMatrixHandle handle);
/*!
 * \brief load a data matrix into binary file
 * \param handle a instance of data matrix
 * \param fname file name
 * \param silent print statistics when saving
 * \return 0 when success, -1 when failure happens
 */
XGB_DLL int XGDMatrixSaveBinary(DMatrixHandle handle,
                                const char *fname, int silent);
/*!
 * \brief Set content in array interface to a content in info.
 * \param handle a instance of data matrix
 * \param field field name.
 * \param c_interface_str JSON string representation of array interface.
 * \return 0 when success, -1 when failure happens
 */
XGB_DLL int XGDMatrixSetInfoFromInterface(DMatrixHandle handle,
    char const* field,
    char const* c_interface_str);

/*!
 * \brief set float vector to a content in info
 * \param handle a instance of data matrix
 * \param field field name, can be label, weight
 * \param array pointer to float vector
 * \param len length of array
 * \return 0 when success, -1 when failure happens
 */
XGB_DLL int XGDMatrixSetFloatInfo(DMatrixHandle handle,
                                  const char *field,
                                  const float *array,
                                  bst_ulong len);
/*!
 * \brief set uint32 vector to a content in info
 * \param handle a instance of data matrix
 * \param field field name
 * \param array pointer to unsigned int vector
 * \param len length of array
 * \return 0 when success, -1 when failure happens
 */
XGB_DLL int XGDMatrixSetUIntInfo(DMatrixHandle handle,
                                 const char *field,
                                 const unsigned *array,
                                 bst_ulong len);
/*!
 * \brief set label of the training matrix
 * \param handle a instance of data matrix
 * \param group pointer to group size
 * \param len length of array
 * \return 0 when success, -1 when failure happens
 */
XGB_DLL int XGDMatrixSetGroup(DMatrixHandle handle,
                              const unsigned *group,
                              bst_ulong len);
/*!
 * \brief get float info vector from matrix
 * \param handle a instance of data matrix
 * \param field field name
 * \param out_len used to set result length
 * \param out_dptr pointer to the result
 * \return 0 when success, -1 when failure happens
 */
XGB_DLL int XGDMatrixGetFloatInfo(const DMatrixHandle handle,
                                  const char *field,
                                  bst_ulong* out_len,
                                  const float **out_dptr);
/*!
 * \brief get uint32 info vector from matrix
 * \param handle a instance of data matrix
 * \param field field name
 * \param out_len The length of the field.
 * \param out_dptr pointer to the result
 * \return 0 when success, -1 when failure happens
 */
XGB_DLL int XGDMatrixGetUIntInfo(const DMatrixHandle handle,
                                 const char *field,
                                 bst_ulong* out_len,
                                 const unsigned **out_dptr);
/*!
 * \brief get number of rows.
 * \param handle the handle to the DMatrix
 * \param nonce nonce received from the enclave during initialization
 * \param nonce_size size in bytes of nonce
 * \param nonce_ctr incrementing counter used to indicate sequence number of API call
 * \param out The address to hold number of rows.
 * \param out_sig signature over the output and nonce
 * \param out_sig_length length of output signature
 * \param signers list of usernames of signing clients
 * \param signatures list of client signatures
 * \param sig_lengths list of signature lengths
 * \return 0 when success, -1 when failure happens
 */
XGB_DLL int XGDMatrixNumRow(DMatrixHandle handle,
                            uint8_t *nonce,
                            size_t nonce_size,
                            uint32_t nonce_ctr,
                            bst_ulong *out,
                            uint8_t** out_sig,
                            size_t *out_sig_length,
                            char **signers,
                            uint8_t* signatures[],
                            size_t* sig_lengths);

/*!
 * \brief get number of columns
 * \param handle the handle to the DMatrix
 * \param nonce nonce received from the enclave during initialization
 * \param nonce_size size in bytes of nonce
 * \param nonce_ctr incrementing counter used to indicate sequence number of API call
 * \param out The output of number of columns
 * \param out_sig signature over the output and nonce
 * \param out_sig_length length of output signature
 * \param signers list of usernames of signing clients
 * \param signatures list of client signatures
 * \param sig_lengths list of signature lengths
 * \return 0 when success, -1 when failure happens
 */
XGB_DLL int XGDMatrixNumCol(DMatrixHandle handle,
                            uint8_t *nonce,
                            size_t nonce_size,
                            uint32_t nonce_ctr,
                            bst_ulong *out,
                            uint8_t** out_sig,
                            size_t *out_sig_length,
                            char **signers,
                            uint8_t* signatures[],
                            size_t* sig_lengths);
// --- start XGBoost class

/*!
 * \brief create xgboost learner
 * \param dmats matrices that are set to be cached
 * \param len length of dmats
 * \param nonce nonce received from the enclave during initialization
 * \param nonce_size size in bytes of nonce
 * \param nonce_ctr incrementing counter used to indicate sequence number of API call
 * \param out handle to the result booster
 * \param out_sig signature over the output and nonce
 * \param out_sig_length length of output signature
 * \param signers list of usernames of signing clients
 * \param signatures list of client signatures
 * \param sig_lengths list of signature lengths
 * \return 0 when success, -1 when failure happens
 */
XGB_DLL int XGBoosterCreate(const DMatrixHandle dmats[],
                            bst_ulong len,
                            uint8_t *nonce,
                            size_t nonce_size,
                            uint32_t nonce_ctr,
                            BoosterHandle *out,
                            uint8_t** out_sig,
                            size_t *out_sig_length,
                            char **signers,
                            uint8_t* signatures[],
                            size_t* sig_lengths);

/*!
 * \brief free obj in handle
 * \param handle handle to be freed
 * \return 0 when success, -1 when failure happens
 */
XGB_DLL int XGBoosterFree(BoosterHandle handle);

/*!
 * \brief set parameters
 * \param handle handle
 * \param name  parameter name
 * \param value value of parameter
 * \param nonce nonce received from the enclave during initialization
 * \param nonce_size size in bytes of nonce
 * \param nonce_ctr incrementing counter used to indicate sequence number of API call
 * \param out_sig signature over the output and nonce
 * \param out_sig_length length of output signature
 * \param signers list of usernames of signing clients
 * \param signatures list of client signatures
 * \param sig_lengths list of signature lengths
 * \return 0 when success, -1 when failure happens
 */
XGB_DLL int XGBoosterSetParam(BoosterHandle handle,
                              const char *name,
                              const char *value,
                              uint8_t *nonce,
                              size_t nonce_size,
                              uint32_t nonce_ctr,
                              uint8_t** out_sig,
                              size_t *out_sig_length,
                              char **signers,
                              uint8_t* signatures[],
                              size_t* sig_lengths);

/*!
 * \brief update the model in one round using dtrain
 * \param handle handle
 * \param iter current iteration rounds
 * \param dtrain training data
 * \param nonce nonce received from the enclave during initialization
 * \param nonce_size size in bytes of nonce
 * \param nonce_ctr incrementing counter used to indicate sequence number of API call
 * \param out_sig signature over the output and nonce
 * \param out_sig_length length of output signature
 * \param signers list of usernames of signing clients
 * \param signatures list of client signatures
 * \param sig_lengths list of signature lengths
 * \return 0 when success, -1 when failure happens
 */
XGB_DLL int XGBoosterUpdateOneIter(BoosterHandle handle,
                                   int iter,
                                   DMatrixHandle dtrain,
                                   uint8_t *nonce,
                                   size_t nonce_size,
                                   uint32_t nonce_ctr,
                                   uint8_t** out_sig,
                                   size_t *out_sig_length,
                                   char **signers,
                                   uint8_t* signatures[],
                                   size_t* sig_lengths);

/*!
 * \brief update the model, by directly specify gradient and second order gradient,
 *        this can be used to replace UpdateOneIter, to support customized loss function
 * \param handle handle
 * \param dtrain training data
 * \param grad gradient statistics
 * \param hess second order gradient statistics
 * \param len length of grad/hess array
 * \return 0 when success, -1 when failure happens
 */
XGB_DLL int XGBoosterBoostOneIter(BoosterHandle handle,
                                  DMatrixHandle dtrain,
                                  float *grad,
                                  float *hess,
                                  bst_ulong len);
/*!
 * \brief get evaluation statistics for xgboost
 * \param handle handle
 * \param iter current iteration rounds
 * \param dmats pointers to data to be evaluated
 * \param evnames pointers to names of each data
 * \param len length of dmats
 * \param out_result the string containing evaluation statistics
 * \return 0 when success, -1 when failure happens
 */
XGB_DLL int XGBoosterEvalOneIter(BoosterHandle handle,
                                 int iter,
                                 DMatrixHandle dmats[],
                                 const char *evnames[],
                                 bst_ulong len,
                                 const char **out_result);

/*!
 * \brief make prediction based on dmat
 * \param handle handle
 * \param dmat data matrix
 * \param option_mask bit-mask of options taken in prediction, possible values
 *          0:normal prediction
 *          1:output margin instead of transformed value
 *          2:output leaf index of trees instead of leaf value, note leaf index is unique per tree
 *          4:output feature contributions to individual predictions
 * \param ntree_limit limit number of trees used for prediction, this is only valid for boosted trees
 *    when the parameter is set to 0, we will use all the trees
 * \param nonce nonce received from the enclave during initialization
 * \param nonce_size size in bytes of nonce
 * \param nonce_ctr incrementing counter used to indicate sequence number of API call
 * \param out_len used to store length of returning result
 * \param out_result used to set a pointer to array
 * \param out_sig signature over the output and nonce
 * \param out_sig_length length of output signature
 * \param signers list of usernames of signing clients
 * \param signatures list of client signatures
 * \param sig_lengths list of signature lengths
 * \return 0 when success, -1 when failure happens
 */
XGB_DLL int XGBoosterPredict(BoosterHandle handle,
                             DMatrixHandle dmat,
                             int option_mask,
                             unsigned ntree_limit,
                             int training,
                             uint8_t *nonce,
                             size_t nonce_size,
                             uint32_t nonce_ctr,
                             bst_ulong *out_len,
                             uint8_t **out_result,
                             uint8_t** out_sig,
                             size_t *out_sig_length,
                             char **signers,
                             uint8_t* signatures[],
                             size_t* sig_lengths);

/*!
 * \brief load model from existing file
 * \param handle handle
 * \param fname file name
 * \param nonce nonce received from the enclave during initialization
 * \param nonce_size size in bytes of nonce
 * \param nonce_ctr incrementing counter used to indicate sequence number of API call
 * \param out_sig signature over the output and nonce
 * \param out_sig_length length of output signature
 * \param signers list of usernames of signing clients
 * \param signatures list of client signatures
 * \param sig_lengths list of signature lengths
 * \return 0 when success, -1 when failure happens
 */
XGB_DLL int XGBoosterLoadModel(BoosterHandle handle,
                               const char *fname,
                               uint8_t *nonce,
                               size_t nonce_size,
                               uint32_t nonce_ctr,
                               uint8_t** out_sig,
                               size_t *out_sig_length,
                               char **signers,
                               uint8_t* signatures[],
                               size_t* sig_lengths);

/*!
 * \brief save model into existing file
 * \param handle handle
 * \param fname file name
 * \param nonce nonce received from the enclave during initialization
 * \param nonce_size size in bytes of nonce
 * \param nonce_ctr incrementing counter used to indicate sequence number of API call
 * \param out_sig signature over the output and nonce
 * \param out_sig_length length of output signature
 * \param signers list of usernames of signing clients
 * \param signatures list of client signatures
 * \param sig_lengths list of signature lengths
 * \return 0 when success, -1 when failure happens
 */
XGB_DLL int XGBoosterSaveModel(BoosterHandle handle,
                               const char *fname,
                               uint8_t *nonce,
                               size_t nonce_size,
                               uint32_t nonce_ctr,
                               uint8_t** out_sig,
                               size_t *out_sig_length,
                               char** signers,
                               uint8_t* signatures[],
                               size_t* sig_lengths);

/*!
 * \brief load model from in memory buffer
 * \param handle handle
 * \param buf pointer to the buffer
 * \param len the length of the buffer
 * \param out_sig signature over the output and nonce
 * \param out_sig_length length of output signature
 * \param signers list of usernames of signing clients
 * \param signatures list of client signatures
 * \param sig_lengths list of signature lengths
 * \return 0 when success, -1 when failure happens
 */
XGB_DLL int XGBoosterLoadModelFromBuffer(BoosterHandle handle,
                                         const void *buf,
                                         bst_ulong len,
                                         uint8_t** out_sig,
                                         size_t *out_sig_length,
                                         char** signers,
                                         uint8_t* signatures[],
                                         size_t* sig_lengths);

/*!
 * \brief save model into binary raw bytes, return header of the array
 * user must copy the result out, before next xgboost call
 * \param handle handle
 * \param nonce nonce received from the enclave during initialization
 * \param nonce_size size in bytes of nonce
 * \param nonce_ctr incrementing counter used to indicate sequence number of API call
 * \param out_len the argument to hold the output length
 * \param out_dptr the argument to hold the output data pointer
 * \param out_sig signature over the output and nonce
 * \param out_sig_length length of output signature
 * \param signers list of usernames of signing clients
 * \param signatures list of client signatures
 * \param sig_lengths list of signature lengths
 * \return 0 when success, -1 when failure happens
 */
XGB_DLL int XGBoosterGetModelRaw(BoosterHandle handle,
                                 uint8_t *nonce,
                                 size_t nonce_size,
                                 uint32_t nonce_ctr,
                                 bst_ulong *out_len,
                                 const char **out_dptr,
                                 uint8_t** out_sig,
                                 size_t *out_sig_length,
                                 char **signers,
                                 uint8_t* signatures[],
                                 size_t* sig_lengths);

/*!
 * \brief dump model, return array of strings representing model dump
 * \param handle handle
 * \param fmap  name to fmap can be empty string
 * \param with_stats whether to dump with statistics
 * \param nonce nonce received from the enclave during initialization
 * \param nonce_size size in bytes of nonce
 * \param nonce_ctr incrementing counter used to indicate sequence number of API call
 * \param out_len length of output array
 * \param out_dump_array pointer to hold representing dump of each model
 * \return 0 when success, -1 when failure happens
 */
XGB_DLL int XGBoosterDumpModel(BoosterHandle handle,
                               const char *fmap,
                               int with_stats,
                               uint8_t *nonce,
                               size_t nonce_size,
                               uint32_t nonce_ctr,
                               bst_ulong *out_len,
                               const char ***out_dump_array);

/*!
 * \brief dump model, return array of strings representing model dump
 * \param handle handle
 * \param fmap  name to fmap can be empty string
 * \param with_stats whether to dump with statistics
 * \param format the format to dump the model in
 * \param nonce nonce received from the enclave during initialization
 * \param nonce_size size in bytes of nonce
 * \param nonce_ctr incrementing counter used to indicate sequence number of API call
 * \param out_len length of output array
 * \param out_dump_array pointer to hold representing dump of each model
 * \param out_sig signature over the output and nonce
 * \param out_sig_length length of output signature
 * \param signers list of usernames of signing clients
 * \param signatures list of client signatures
 * \param sig_lengths list of signature lengths
 * \return 0 when success, -1 when failure happens
 */
XGB_DLL int XGBoosterDumpModelEx(BoosterHandle handle,
                                 const char *fmap,
                                 int with_stats,
                                 const char *format,
                                 uint8_t *nonce,
                                 size_t nonce_size,
                                 uint32_t nonce_ctr,
                                 bst_ulong *out_len,
                                 const char ***out_dump_array,
                                 uint8_t** out_sig,
                                 size_t *out_sig_length,
                                 char **signers,
                                 uint8_t* signatures[],
                                 size_t* sig_lengths);

/*!
 * \brief dump model, return array of strings representing model dump
 * \param handle handle
 * \param fnum number of features
 * \param fname names of features
 * \param ftype types of features
 * \param with_stats whether to dump with statistics
 * \param nonce nonce received from the enclave during initialization
 * \param nonce_size size in bytes of nonce
 * \param nonce_ctr incrementing counter used to indicate sequence number of API call
 * \param out_len length of output array
 * \param out_models pointer to hold representing dump of each model
 * \param out_sig signature over the output and nonce
 * \param out_sig_length length of output signature
 * \param signers list of usernames of signing clients
 * \param signatures list of client signatures
 * \param sig_lengths list of signature lengths
 * \return 0 when success, -1 when failure happens
 */
XGB_DLL int XGBoosterDumpModelWithFeatures(BoosterHandle handle,
                                           int fnum,
                                           const char **fname,
                                           const char **ftype,
                                           int with_stats,
                                           uint8_t *nonce,
                                           size_t nonce_size,
                                           uint32_t nonce_ctr,
                                           bst_ulong *out_len,
                                           const char ***out_models,
                                           uint8_t** out_sig,
                                           size_t *out_sig_length,
                                           char **signers,
                                           size_t signer_lengths[],
                                           uint8_t* signatures[],
                                           size_t* sig_lengths);

/*!
 * \brief dump model, return array of strings representing model dump
 * \param handle handle
 * \param fnum number of features
 * \param fname names of features
 * \param ftype types of features
 * \param with_stats whether to dump with statistics
 * \param format the format to dump the model in
 * \param nonce nonce received from the enclave during initialization
 * \param nonce_size size in bytes of nonce
 * \param nonce_ctr incrementing counter used to indicate sequence number of API call
 * \param out_len length of output array
 * \param out_models pointer to hold representing dump of each model
 * \param out_sig signature over the output and nonce
 * \param out_sig_length length of output signature
 * \param signers list of usernames of signing clients
 * \param signatures list of client signatures
 * \param sig_lengths list of signature lengths
 * \return 0 when success, -1 when failure happens
 */
XGB_DLL int XGBoosterDumpModelExWithFeatures(BoosterHandle handle,
                                             int fnum,
                                             const char **fname,
                                             const char **ftype,
                                             int with_stats,
                                             const char *format,
                                             uint8_t *nonce,
                                             size_t nonce_size,
                                             uint32_t nonce_ctr,
                                             bst_ulong *out_len,
                                             const char ***out_models,
                                             uint8_t** out_sig,
                                             size_t *out_sig_length,
                                             char **signers,
                                             uint8_t* signatures[],
                                             size_t* sig_lengths);

/*!
 * \brief Get string attribute from Booster.
 * \param handle handle
 * \param key The key of the attribute.
 * \param out The result attribute, can be NULL if the attribute do not exist.
 * \param success Whether the result is contained in out.
 * \return 0 when success, -1 when failure happens
 */
XGB_DLL int XGBoosterGetAttr(BoosterHandle handle,
                             const char* key,
                             const char** out,
                             int *success);
/*!
 * \brief Set or delete string attribute.
 *
 * \param handle handle
 * \param key The key of the attribute.
 * \param value The value to be saved.
 *              If nullptr, the attribute would be deleted.
 * \return 0 when success, -1 when failure happens
 */
XGB_DLL int XGBoosterSetAttr(BoosterHandle handle,
                             const char* key,
                             const char* value);
/*!
 * \brief Get the names of all attribute from Booster.
 * \param handle handle
 * \param out_len the argument to hold the output length
 * \param out pointer to hold the output attribute stings
 * \return 0 when success, -1 when failure happens
 */
XGB_DLL int XGBoosterGetAttrNames(BoosterHandle handle,
                                  bst_ulong* out_len,
                                  const char*** out);

// --- Distributed training API----
// NOTE: functions in rabit/c_api.h will be also available in libxgboost.so
/*!
 * \brief Initialize the booster from rabit checkpoint.
 *  This is used in distributed training API.
 * \param handle handle
 * \param version The output version of the model.
 * \return 0 when success, -1 when failure happens
 */
XGB_DLL int XGBoosterLoadRabitCheckpoint(
    BoosterHandle handle,
    int* version);

/*!
 * \brief Save the current checkpoint to rabit.
 * \param handle handle
 * \return 0 when success, -1 when failure happens
 */
XGB_DLL int XGBoosterSaveRabitCheckpoint(BoosterHandle handle);

XGB_DLL int get_remote_report_with_pubkey_and_nonce(
    uint8_t** pem_key,
    size_t* key_size,
    uint8_t** nonce,
    size_t* nonce_size,
    char*** client_list,
    size_t* client_list_size,
    uint8_t** remote_report,
    size_t* remote_report_size);

XGB_DLL int verify_remote_report_and_set_pubkey(
    uint8_t* pem_key,
    size_t pem_key_size,
    uint8_t* remote_report,
    size_t remote_report_size);

XGB_DLL int verify_remote_report_and_set_pubkey_and_nonce(
    uint8_t* pem_key,
    size_t pem_key_size,
    uint8_t* nonce,
    size_t nonce_size,
    char** usernames,
    size_t num_users,
    uint8_t* remote_report,
    size_t remote_report_size);

XGB_DLL int add_client_key(
    //char* fname,
    uint8_t* data,
    size_t data_len,
    uint8_t* signature,
    size_t sig_len);

XGB_DLL int add_client_key_with_certificate(
    char* cert,
    int cert_len,
    uint8_t* data,
    size_t data_len,
    uint8_t* signature,
    size_t sig_len);

XGB_DLL int get_enclave_symm_key(
    char* username,
    uint8_t** out,
    size_t* out_size);

XGB_DLL int encrypt_data_with_pk(
    char* data,
    size_t len,
    uint8_t* pem_key,
    size_t key_size,
    uint8_t* encrypted_data,
    size_t* encrypted_data_size);

XGB_DLL int verify_signature(
    uint8_t* pem_key,
    size_t key_size,
    uint8_t* data,
    size_t data_size,
    uint8_t* signature,
    size_t sig_len);

XGB_DLL int sign_data_with_keyfile(
    char* keyfile,
    uint8_t* encrypted_data,
    size_t encrypted_data_size,
    uint8_t* signature,
    size_t* sig_len);

XGB_DLL int decrypt_predictions(
    char* key,
    uint8_t* encrypted_preds,
    size_t preds_len,
    bst_float** preds);

XGB_DLL int decrypt_enclave_key(
    char* key,
    uint8_t* encrypted_key,
    size_t len,
    uint8_t** out_key);

XGB_DLL int encrypt_file(
    char* fname,
    char* e_fname,
    char* k_fname);

XGB_DLL int encrypt_file_with_keybuf(
    char* fname,
    char* e_fname,
    char* key);

XGB_DLL int decrypt_file_with_keybuf(
    char* fname,
    char* e_fname,
    char* key);

#if defined(__HOST__)
// Ocalls
int ocall_rabit__GetRank();

int ocall_rabit__GetWorldSize();

int ocall_rabit__IsDistributed();
#endif  // __HOST__

#endif  // XGBOOST_C_API_H_
