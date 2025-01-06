#ifndef __SORTINGNETWORK_HPP__
#define __SORTINGNETWORK_HPP__

// #include "../Enclave_globals.h"
#include "../oasm_lib.h"
#include "../utils.hpp"

// enum OSort_Style{BUFFERS, KEY_DATA, KEY_DATAX2};
/* 
void OddEvenMergeSort(unsigned char *buf, uint64_t N, size_t block_size);

double DecryptAndOddEvenMergeSort(unsigned char *encrypted_buffer, uint64_t N, size_t block_size,
  unsigned char *result_buffer);
 */
// void BitonicSort(unsigned char *buffer, size_t N, size_t block_size, bool ascend);

// void BitonicSort(unsigned char *keys, size_t N, unsigned char *associated_data1,
//       unsigned char *associated_data2, size_t data_size, bool ascend);

template<OSwap_Style oswap_style, typename KeyType = uint64_t>
void BitonicSort(unsigned char *buffer, size_t N, size_t block_size, bool ascend);

// template<OSwap_Style oswap_style, typename KeyType = uint64_t>
// inline void BitonicSort(unsigned char *keys, size_t N, unsigned char *associated_data1,
//       unsigned char *associated_data2, size_t data_size, bool ascend);

// template<OSwap_Style oswap_style, typename KeyType = uint64_t>
// void BitonicMerge(unsigned char *keys, size_t N, unsigned char *associated_data1,
//       unsigned char *associated_data2, size_t data_size, bool ascend);

// void testBitonicSort();

#include "SortingNetwork.tcc"
#endif
