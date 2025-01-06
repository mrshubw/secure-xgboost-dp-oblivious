#include <array>
// #include <sgx_tcrypto.h>
#include "psrr/wakson/SortingNetwork/SortingNetwork.hpp"
/* 
static unsigned char aesdeckey[SGX_AESGCM_KEY_SIZE];
static unsigned char aesenckey[SGX_AESGCM_KEY_SIZE];

int num_oddevenmerge_comps(uint64_t N) {
    int logn = calculatelog2(N);
    return (N/4) * logn * logn - (N/4) * logn + N - 1;
}

// The input-output buffer must contain N decrypted blocks of length block_size bytes each.
// Each block consists of a SN_KEY_SIZE-byte key followed by a (block_size-SN_KEY_SIZE)-byte data
// block. The data will be sorted in-place ascending by key. block_size must be a multiple of 16.
void OddEvenMergeSort(unsigned char *buf, uint64_t N, size_t block_size) {
  if(block_size==4){
    OddEvenMergeSort<OSWAP_4>(buf, N, block_size);
  } else if(block_size==8){
    OddEvenMergeSort<OSWAP_8>(buf, N, block_size);
  } else if(block_size==12){
    OddEvenMergeSort<OSWAP_12>(buf, N, block_size);
  } else if(block_size%16==0){
    OddEvenMergeSort<OSWAP_16X>(buf, N, block_size);
  } else{
    OddEvenMergeSort<OSWAP_8_16X>(buf, N, block_size);
  }  
}

double DecryptAndOddEvenMergeSort(unsigned char *encrypted_buffer, uint64_t N, 
        size_t encrypted_block_size, unsigned char *result_buffer) {

  long t1, t2;
  //ocall_clock(&t0);

  // Decrypt buffer to decrypted_buffer
  unsigned char *decrypted_buffer = NULL;
  size_t decrypted_block_size = decryptBuffer(encrypted_buffer, N, encrypted_block_size,
    &decrypted_buffer);
  ocall_clock(&t1);

  // Apply odd-even mergesort network
  PRB_pool_init(1);
  OddEvenMergeSort(decrypted_buffer, N, decrypted_block_size);
  ocall_clock(&t2);

  // Encrypt buffer to result_buffer
  encryptBuffer(decrypted_buffer, N, decrypted_block_size, result_buffer);
  PRB_pool_shutdown();
  //ocall_clock(&t3);

  // CLOCKS_PER_SEC == 1000000, so CLOCKS_PER_MS == 1000
  //double decryption_ms = ((double)(t1-t0))/1000.0;
  double compare_ms = ((double)(t2-t1))/1000.0;
  //double encryption_ms = ((double)(t3-t2))/1000.0;

  int num_comparisons = num_oddevenmerge_comps(N);
  //printf("Estimated comparisons for %d items: %d\n", N, num_comparisons);
  //printf("Counted comparisons for %d items: %d\n", N, OSWAP_COUNTER);

  //printf("Buffer decryption time: %lf ms\n", decryption_ms);
  //printf("Compare-and-swaps time: %lf ms\n", compare_ms);
  //printf("Buffer encryption time: %lf ms\n", encryption_ms);

  free(decrypted_buffer);
  return(compare_ms);
}


// NOTE: We don't need a _timed and non-timed one. If we dont keep SN Application. We can
// just make _timed the only one and remove _timed from the name
double DecryptAndOddEvenMergeSort_timed(unsigned char *encrypted_buffer, uint64_t N, 
        size_t encrypted_block_size, unsigned char *result_buffer, enc_ret *ret) {

  long t1, t2;
  //ocall_clock(&t0);

  // Decrypt buffer to decrypted_buffer
  unsigned char *decrypted_buffer = NULL;
  size_t decrypted_block_size = decryptBuffer(encrypted_buffer, N, encrypted_block_size,
    &decrypted_buffer);
  ocall_clock(&t1);

  // Apply odd-even mergesort network
  PRB_pool_init(1);
  OddEvenMergeSort(decrypted_buffer, N, decrypted_block_size);
  ocall_clock(&t2);

  // Encrypt buffer to result_buffer
  encryptBuffer(decrypted_buffer, N, decrypted_block_size, result_buffer);
  PRB_pool_shutdown();
  //ocall_clock(&t3);

  // CLOCKS_PER_SEC == 1000000, so CLOCKS_PER_MS == 1000
  //double decryption_ms = ((double)(t1-t0))/1000.0;
  double compare_ms = ((double)(t2-t1))/1000.0;
  //double encryption_ms = ((double)(t3-t2))/1000.0;

  int num_comparisons = num_oddevenmerge_comps(N);
  //printf("Estimated comparisons for %d items: %d\n", N, num_comparisons);
  //printf("Counted comparisons for %d items: %d\n", N, OSWAP_COUNTER);

  //printf("Buffer decryption time: %lf ms\n", decryption_ms);
  //printf("Compare-and-swaps time: %lf ms\n", compare_ms);
  //printf("Buffer encryption time: %lf ms\n", encryption_ms);

  free(decrypted_buffer);
  ret->OSWAP_count = OSWAP_COUNTER;
  ret->ptime = compare_ms;
  return(compare_ms);
}
 */
/*
ascend: 1 = ascending
        0 = descending
*/

//Same as BitonicSort but along with key swaps, swap associated_data based on the same flag.
// NOTE: 1) We assume keys are limited to 8 byte values!
//       2) We assume associated_data1 and associated_data2 have the same data_size! This helps set
//          the Oswap_Style cleanly!
/* 
void BitonicSort(unsigned char *keys, size_t N, unsigned char *associated_data1, unsigned char *associated_data2, size_t data_size, bool ascend) {
  if(data_size==4){
    BitonicSort<OSWAP_4>(keys, N, associated_data1, associated_data2, data_size, ascend);
  } else if(data_size==8){
    BitonicSort<OSWAP_8>(keys, N, associated_data1, associated_data2, data_size, ascend);
  } else if (data_size%16==0){
    BitonicSort<OSWAP_16X>(keys, N, associated_data1, associated_data2, data_size, ascend);
  } else{
    BitonicSort<OSWAP_8_16X>(keys, N, associated_data1, associated_data2, data_size, ascend);
  }
}
 */
// void BitonicSort(unsigned char *buffer, size_t N, size_t block_size, bool ascend) {
//   if(block_size==4){
//     BitonicSort<OSWAP_4>(buffer, N, block_size, ascend);
//   } else if(block_size==8){
//     BitonicSort<OSWAP_8>(buffer, N, block_size, ascend);
//   } else if(block_size==12){
//     BitonicSort<OSWAP_12>(buffer, N, block_size, ascend);
//   } else if (block_size%16==0){
//     BitonicSort<OSWAP_16X>(buffer, N, block_size, ascend);
//   }
//   else{
//     BitonicSort<OSWAP_8_16X>(buffer, N, block_size, ascend);
//   }
// }
/* 
//TODO: Take this off, if we no longer plan to support SN_App!
double DecryptAndBitonicSort(unsigned char *encrypted_buffer, uint64_t N, size_t encrypted_block_size,
  unsigned char *result_buffer) {

  long t1, t2;

  // Decrypt buffer to decrypted_buffer
  unsigned char *decrypted_buffer = NULL;
  
  size_t decrypted_block_size = decryptBuffer(encrypted_buffer, N, encrypted_block_size,
    &decrypted_buffer);

  ocall_clock(&t1);

  // Apply odd-even mergesort network
  //PRB_pool_init(1);
  BitonicSort(decrypted_buffer, N, decrypted_block_size, true);
  ocall_clock(&t2);

  // Encrypt buffer to result_buffer
  encryptBuffer(decrypted_buffer, N, decrypted_block_size, result_buffer);
  //PRB_pool_shutdown();

  // CLOCKS_PER_SEC == 1000000, so CLOCKS_PER_MS == 1000
  double compare_ms = ((double)(t2-t1))/1000.0;

  free(decrypted_buffer);
  return(compare_ms);
}

double DecryptAndBitonicSort(unsigned char *encrypted_buffer, uint64_t N, size_t encrypted_block_size,
  unsigned char *result_buffer, enc_ret *ret) {

  long t1, t2;

  // Decrypt buffer to decrypted_buffer
  unsigned char *decrypted_buffer = NULL;
  
  size_t decrypted_block_size = decryptBuffer(encrypted_buffer, N, encrypted_block_size,
    &decrypted_buffer);

  ocall_clock(&t1);

  memcpy(result_buffer, decrypted_buffer, N*decrypted_block_size);
  // Apply odd-even mergesort network
  PRB_pool_init(1);
  BitonicSort(decrypted_buffer, N, decrypted_block_size, true);
  //BitonicSort(decrypted_buffer, N, result_buffer, NULL, decrypted_block_size, true);
  ocall_clock(&t2);

  // Encrypt buffer to result_buffer
  encryptBuffer(decrypted_buffer, N, decrypted_block_size, result_buffer);
  PRB_pool_shutdown();

  // CLOCKS_PER_SEC == 1000000, so CLOCKS_PER_MS == 1000
  double compare_ms = ((double)(t2-t1))/1000.0;

  free(decrypted_buffer);
  ret->OSWAP_count = OSWAP_COUNTER;
  ret->ptime = compare_ms;

  return(compare_ms);
}

double DecryptAndOddEvenMergeSortShuffle(unsigned char *encrypted_buffer, uint64_t N, size_t encrypted_block_size,
  unsigned char *result_buffer, enc_ret *ret) {

  long t1, t2;
  //ocall_clock(&t0);

  // Decrypt buffer to decrypted_buffer
  PRB_pool_init(1);
  unsigned char *decrypted_buffer = NULL;
  unsigned char *random_bytes = new unsigned char[8*N];
  getBulkRandomBytes(random_bytes, 8*N);

  size_t decrypted_block_size = decryptBuffer_attachRTags(encrypted_buffer, N, encrypted_block_size,
    random_bytes, &decrypted_buffer);
  ocall_clock(&t1);

  // Apply odd-even mergesort network
  OddEvenMergeSort(decrypted_buffer, N, decrypted_block_size);
  ocall_clock(&t2);

  // Encrypt buffer to result_buffer
  encryptBuffer_removeRTags(decrypted_buffer, N, decrypted_block_size, result_buffer);
  PRB_pool_shutdown();

  // CLOCKS_PER_SEC == 1000000, so CLOCKS_PER_MS == 1000
  double compare_ms = ((double)(t2-t1))/1000.0;

  free(decrypted_buffer);
  delete []random_bytes;
  ret->OSWAP_count = OSWAP_COUNTER;
  ret->ptime = compare_ms;
  
  return(compare_ms);
}



double DecryptAndBitonicSortShuffle(unsigned char *encrypted_buffer, uint64_t N, 
        size_t encrypted_block_size, unsigned char *result_buffer, enc_ret *ret) {
  long t1, t2;

  // Decrypt buffer to decrypted_buffer
  PRB_pool_init(1);
  unsigned char *decrypted_buffer = NULL;
  size_t rsize = 8 * (size_t) N;
  unsigned char *random_bytes = (unsigned char*) malloc(rsize);
  if(random_bytes == NULL)
    printf("Failed to allocate random_bytes in D&BSS\n");
  getBulkRandomBytes(random_bytes, rsize);
  
  size_t decrypted_block_size = decryptBuffer_attachRTags(encrypted_buffer, N, 
          encrypted_block_size, random_bytes, &decrypted_buffer);

  ocall_clock(&t1);

  // Apply odd-even mergesort network
  // NOTE: We will never have decrypted_block_size==8, since attaching rTag will add 8 bytes.
  // So minimum block_size here is 16
  BitonicSort(decrypted_buffer, N, decrypted_block_size, true);

  ocall_clock(&t2);

  // Encrypt buffer to result_buffer
  encryptBuffer_removeRTags(decrypted_buffer, N, decrypted_block_size, result_buffer);
  PRB_pool_shutdown();

  // CLOCKS_PER_SEC == 1000000, so CLOCKS_PER_MS == 1000
  double compare_ms = ((double)(t2-t1))/1000.0;

  free(decrypted_buffer);
  free(random_bytes);

  ret->OSWAP_count = OSWAP_COUNTER;
  ret->ptime = compare_ms;
  return(compare_ms);
}
 */
/* 
void testBitonicSort(){
  size_t N = 10;
  // Test the normal version of bitonic sort; each data item is a 16-byte key and two 8-byte integers
  unsigned char *data = new unsigned char[N*(16+8+8)];
  PRB_pool_init(1);
  for(size_t i=0; i<N; i++){
    unsigned char *item = data+(i*(16+8+8));
    getRandomBytes((unsigned char*)item, 16);
    *(uint64_t*)(item+16) = i;
    *(uint64_t*)(item+24) = N-i;
  }
  PRB_pool_shutdown();

  printf("Before BitonicSort\n");
  for(size_t i=0; i<N; i++){
    unsigned char *item = data+(i*(16+8+8));
    printf("(");
    for (size_t j=0; j<16; ++j) { printf("%02x", item[15-j]); }
    printf(", %d, %d)\n", *(uint64_t*)(item+16), *(uint64_t*)(item+24));
  }
  printf("\n");

  BitonicSort<OSWAP_16X, __uint128_t>((unsigned char*) data, N, 16+8+8, true);
  
  printf("After BitonicSort\n");
  for(size_t i=0; i<N; i++){
    unsigned char *item = data+(i*(16+8+8));
    printf("(");
    for (size_t j=0; j<16; ++j) { printf("%02x", item[15-j]); }
    printf(", %d, %d)\n", *(uint64_t*)(item+16), *(uint64_t*)(item+24));
  }
  printf("\n");

  printf("\n\n\n");

  delete []data;

  // Test the associated data version of bitonic sort
  __uint128_t *key_array = new __uint128_t[N];
  uint64_t *ass_data1 = new uint64_t[N];  
  uint64_t *ass_data2 = new uint64_t[N];

  PRB_pool_init(1);
  for(size_t i=0; i<N; i++){
    size_t random_coin;
    getRandomBytes((unsigned char*) (key_array+i), 16);
    ass_data1[i] = i;
    ass_data2[i] = N-i;
  }
  PRB_pool_shutdown();


  printf("Before BitonicSort (with associated data)\n");
  for(size_t i=0; i<N; i++){
    printf("(");
    for (size_t j=0; j<16; ++j) { printf("%02x", ((unsigned char*)(key_array+i))[15-j]); }
    printf(", %d, %d)\n", ass_data1[i], ass_data2[i]);
  }
  printf("\n");

  BitonicSort<OSWAP_8, __uint128_t>((unsigned char*) key_array, N, (unsigned char*)ass_data1, 
        (unsigned char*) ass_data2, 8, true);
  
  printf("After BitonicSort (with associated data)\n");
  for(size_t i=0; i<N; i++){
    printf("(");
    for (size_t j=0; j<16; ++j) { printf("%02x", ((unsigned char*)(key_array+i))[15-j]); }
    printf(", %d, %d)\n", ass_data1[i], ass_data2[i]);
  }

  printf("\n\n\n");

  delete []key_array;
  delete []ass_data1;
  delete []ass_data2;
}
 */