/*
*    ZeroTrace: Oblivious Memory Primitives from Intel SGX
*    Copyright (C) 2018  Sajin (sshsshy)
*
*    This program is free software: you can redistribute it and/or modify
*    it under the terms of the GNU General Public License as published by
*    the Free Software Foundation, version 3 of the License.
*
*    This program is distributed in the hope that it will be useful,
*    but WITHOUT ANY WARRANTY; without even the implied warranty of
*    MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
*    GNU General Public License for more details.
*
*    You should have received a copy of the GNU General Public License
*    along with this program.  If not, see <https://www.gnu.org/licenses/>.
*/


#ifndef __OASM_LIB__
	#define __OASM_LIB__

  // #ifndef BEFTS_MODE
  //   #include "../CONFIG.h"
  //   #include "Enclave_globals.h"
  // #endif

#include <stdint.h>  // 提供 uint8_t, uint32_t 等类型
#include <stddef.h>  // 提供 size_t 类型
/* 
  // Oblivious Buffer move/swap functions:
	extern "C" void oswap_buffer_16x(unsigned char *dest, unsigned char *source, uint32_t buffersize, uint8_t flag);
	extern "C" void oswap_buffer_byte(unsigned char *dest, unsigned char *source, uint32_t buffersize, uint8_t flag);

	extern "C" void oswap_buffer_byte_16x(unsigned char *dest, unsigned char *source, uint32_t buffersize, uint8_t flag);
	extern "C" void oswap_buffer_byte_v2(unsigned char *dest, unsigned char *source, uint8_t flag);

	extern "C" void ogt_comp_swap(uint64_t *key1, uint64_t *key2, unsigned char *buff1, unsigned char *buff2, uint32_t buffersize);
 */

  // 占位，用于确定oswap_buffer的正确性，目前不是oblivious
  inline void oswap(uint8_t *block1, uint8_t *block2, size_t block_size, bool swap_flag){
    if(swap_flag){
        for(size_t i=0; i<block_size; i++){
            uint8_t temp = block1[i];
            block1[i] = block2[i];
            block2[i] = temp;
        }
    }
  }

  enum OSwap_Style {OSWAP_1, OSWAP_2, OSWAP_4, OSWAP_8, OSWAP_12, OSWAP_16X, OSWAP_8_16X, OSWAP_ANY };
  template<OSwap_Style oswap_style> inline void oswap_buffer(unsigned char *dest, unsigned char *source, uint32_t buffersize, uint8_t flag);
  template<typename KeyType> inline void oswap_key(unsigned char *dest, unsigned char *source, uint8_t flag);

  template<OSwap_Style oswap_style> inline void omove_buffer(unsigned char *dest, unsigned char *source, uint32_t buffersize, uint8_t flag);

  inline uint8_t ogt_set_flag(uint64_t key1, uint64_t key2)
  {
    uint8_t flag;
    __asm__ (
          "# inline ogt_set_flag\n"
          "cmp %[key2], %[key1]\n"
          //"# FOAV ogt_set_flag key1 (%[key1]):\n"
          "seta %[flag]\n"
          : [flag] "=r" (flag)
          : [key1] "r" (key1), [key2] "r" (key2)
          : "cc"
      );
    return flag;
  }

  // A size-templated version of oblivious greater than for wide-key Bitonic Sort
  // Returns 1 if (*key1p) > (*key2p), and 0 otherwise, in a fully oblivious manner.
  template<typename keytype>
  inline uint8_t ogt(const keytype *key1p, const keytype *key2p){
      return (*key1p) > (*key2p) ? 1 : 0;
  };

  template<>
  inline uint8_t ogt<uint32_t>(const uint32_t *key1p, const uint32_t *key2p)
  {
    uint8_t flag;
    __asm__ (
          "# inline ogt_uint32\n"
          "movl (%[key1p]), %%eax\n"
          "cmpl (%[key2p]), %%eax\n"
          "seta %[flag]\n"
          : [flag] "=r" (flag)
          : [key1p] "r" (key1p), [key2p] "r" (key2p)
          : "eax", "cc"
      );
    return flag;
  }

  template<>
  inline uint8_t ogt<uint64_t>(const uint64_t *key1p, const uint64_t *key2p) {
    __asm__ ("# inline ogt_uint64\n");
    return ogt_set_flag(*key1p, *key2p);
  }

  template<>
  inline uint8_t ogt<__uint128_t>(const __uint128_t *key1p, const __uint128_t *key2p) {
    uint8_t flag;
    __asm__ (
          "# inline ogt_uint128\n"
	  "movq    8(%[key2p]), %%rcx\n"
          "movq    (%[key1p]), %%rax\n"
          "cmpq    %%rax, (%[key2p])\n"
          "sbbq    8(%[key1p]), %%rcx\n"
          "setc    %[flag]\n"
          : [flag] "=r" (flag)
          : [key1p] "r" (key1p), [key2p] "r" (key2p)
          : "rax", "rcx", "cc"
      );
    return flag;
  }

  inline uint8_t oge_set_flag(uint64_t key1, uint64_t key2)
  {
    uint8_t flag;
    __asm__ (
          "# inline oge_set_flag\n"
          "cmp %[key2], %[key1]\n"
          //"# FOAV oge_set_flag key1 (%[key1]):\n"
          "setae %[flag]\n"
          : [flag] "=r" (flag)
          : [key1] "r" (key1), [key2] "r" (key2)
          : "cc"
      );
    return flag;
  }

  inline uint8_t oe_set_flag(uint32_t key1, uint32_t key2)
  {
    uint8_t flag;
    __asm__ (
          "# inline oe_set_flag\n"
          "cmp %[key2], %[key1]\n"
          //"# FOAV oe_set_flag key1 (%[key1]):\n"
          "sete %[flag]\n"
          : [flag] "=r" (flag)
          : [key1] "r" (key1), [key2] "r" (key2)
          : "cc"
      );
    return flag;
  }

  inline void oset_value(uint64_t *dest, uint64_t value, uint32_t flag)
  {
    __asm__ (
        "# inline oset_value\n"
        "mov (%[dest]), %%r10\n"
        "test %[flag], %[flag]\n"
        "cmovnz %[value], %%r10\n"
        "mov %%r10, (%[dest])\n"
        :
        : [dest] "r" (dest), [value] "r" (value), [flag] "r" (flag)
        : "cc", "memory", "r10"
    );
  }

  inline void oset_value_uint32_t(uint32_t *dest, uint32_t value, uint8_t flag)
  {
    __asm__ (
        "# inline oset_value_uint32_t\n"
        "mov (%[dest]), %%r10d\n"
        "test %[flag], %[flag]\n"
        "cmovnz %[value], %%r10d\n"
        "mov %%r10d, (%[dest])\n"
        :
        : [dest] "r" (dest), [value] "r" (value), [flag] "r" (flag)
        : "cc", "memory", "r10"
    );
  }

  #include "oasm_lib.tcc"

#endif
