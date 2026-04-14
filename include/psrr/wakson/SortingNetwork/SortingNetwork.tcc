#ifndef __SORTINGNETWORK_TCC__
#define __SORTINGNETWORK_TCC__

/* 
// Merge operation for odd-even mergesort. Takes number of spaces to "skip" between items in buffer.
// Left and right parts must be sorted, with left size a power of two and right size smaller.
// Merges them to return a sorted result.
template<OSwap_Style oswap_style>
void OddEvenMerge(unsigned char *buf, uint64_t skip, uint64_t N, size_t block_size) {
  unsigned char *block1;
  unsigned char *block2;

  FOAV_SAFE_CNTXT(OddEvenMerge, N)
  if (N < 2) {
    return;
  }
  FOAV_SAFE_CNTXT(OddEvenMerge, N)
  if (N == 2) {
    block1 = buf;
    block2 = buf + block_size + (block_size*skip);
    oswap_buffer<oswap_style>(block1, block2, block_size, 
          ogt_set_flag(*((uint64_t *) block1), *((uint64_t *) block2)));
    return;
  }

  // Merge odd items
  OddEvenMerge<oswap_style>(buf, 2*skip+1, (N/2)+(N%2), block_size);
  // Merge even items
  OddEvenMerge<oswap_style>(buf+block_size+(block_size*skip), 2*skip+1, N/2, block_size);

  // Compare-and-swap subsequent item pairs, skipping first item
  block2 = buf;

  FOAV_SAFE_CNTXT(OddEvenMerge, N)
  for (int i=0; i<(N-1)/2; i++) {
    FOAV_SAFE_CNTXT(OddEvenMerge, i)
    block1 = block2 + block_size + (block_size*skip);
    block2 = block1 + block_size + (block_size*skip);
    oswap_buffer<oswap_style>(block1, block2, block_size,
        ogt_set_flag(*((uint64_t *) block1), *((uint64_t *) block2)));    
  }
}


template<OSwap_Style oswap_style>
void OddEvenMergeSort(unsigned char *buf, uint64_t N, size_t block_size) {
  // Perform compare-and-swaps
  unsigned char *block1 = buf;
  unsigned char *block2 = buf + block_size;

  FOAV_SAFE_CNTXT(OddEvenMerge, N)
  if (N < 2) {
    return;
  }
  FOAV_SAFE_CNTXT(OddEvenMerge, N)
  if (N == 2) {
    bool swap_flag = ogt_set_flag(*((uint64_t *) block1), *((uint64_t *) block2));
    oswap_buffer<oswap_style>(block1, block2, block_size, swap_flag);    
    return;
  }

  // Divide into maximum power of two and remainder
  uint64_t N1 = pow2_lt(N);
  uint64_t N2 = N - N1;

  // Recursively sort left and right parts
  OddEvenMergeSort<oswap_style>(buf, N1, block_size);
  OddEvenMergeSort<oswap_style>(buf + block_size*N1, N2, block_size);

  // Apply merge operation to complete sort
  OddEvenMerge<oswap_style>(buf, 0, N, block_size);
}
 */
template<OSwap_Style oswap_style, typename KeyType>
void BitonicMerge(unsigned char *buffer, size_t N, size_t block_size, bool ascend) {
  FOAV_SAFE2_CNTXT(BitonicMerge, N, block_size)
  if(N<2){
    return;
  }
  else if((N & (N * -1))!=N) {
    size_t M = pow2_lt(N);
    unsigned char *block1 = buffer;
    unsigned char *block2 = buffer + (M * block_size); 
    size_t feasible_swaps = N - M;

    FOAV_SAFE2_CNTXT(BitonicMerge, feasible_swaps, M)
    for(size_t i=0; i<feasible_swaps; i++) {
      FOAV_SAFE2_CNTXT(BitonicMerge, feasible_swaps, i)
      uint8_t swap_flag = ogt<KeyType>((KeyType*)block1, (KeyType*)block2);
      FOAV_SAFE_CNTXT(BitonicMerge, ascend)
      if(ascend){
        oswap_buffer<oswap_style>(block1, block2, block_size, swap_flag);
      } else {
        oswap_buffer<oswap_style>(block1, block2, block_size, !swap_flag);
      }
      block1+=block_size;
      block2+=block_size; 
      FOAV_SAFE2_CNTXT(BitonicMerge, feasible_swaps, i)
    }
 
    BitonicMerge<oswap_style, KeyType>(buffer, M, block_size, ascend);
    BitonicMerge<oswap_style, KeyType>(buffer + (M * block_size), N-M, block_size, ascend);
  } 
  else{ //Power of 2 case
    size_t split = N/2;
    unsigned char *block1 = buffer;
    unsigned char *block2 = buffer + (split * block_size); 
    
    FOAV_SAFE_CNTXT(BitonicSort, split)
    for(size_t i=0; i<split; i++) {
    FOAV_SAFE_CNTXT(BitonicSort, i)
    FOAV_SAFE_CNTXT(BitonicSort, split)
      uint8_t swap_flag = ogt<KeyType>((KeyType*)block1, (KeyType*)block2);
      FOAV_SAFE_CNTXT(BitonicSort, ascend)
      if(ascend){
        oswap_buffer<oswap_style>(block1, block2, block_size, swap_flag);
        //ogt_comp_swap((uint64_t *) block1, (uint64_t *) block2, block1, block2, block_size);
      } else {
        oswap_buffer<oswap_style>(block1, block2, block_size, !swap_flag);
        //ogt_comp_swap((uint64_t *) block2, (uint64_t *) block1, block2, block1, block_size);
      }
      block1+=block_size;
      block2+=block_size; 
    } 

    BitonicMerge<oswap_style, KeyType>(buffer, split, block_size, ascend);
    BitonicMerge<oswap_style, KeyType>(buffer + (split * block_size), split, block_size, ascend);
  }
}



template<OSwap_Style oswap_style, typename KeyType>
void BitonicSort(unsigned char *buffer, size_t N, size_t block_size, bool ascend) {
  FOAV_SAFE_CNTXT(BitonicSort, N)
  if(N < 2){
    return;
  }
  else {  // Handle non-power of 2 case:
    size_t N1 = N/2; 
    BitonicSort<oswap_style, KeyType>(buffer, N1, block_size, !ascend);
    BitonicSort<oswap_style, KeyType>(buffer + (block_size * N1), N-N1, block_size, ascend);
    BitonicMerge<oswap_style, KeyType>(buffer, N, block_size, ascend);
  }
}


template<typename KeyType>
void BitonicSort(unsigned char *buffer, size_t N, size_t block_size, bool ascend){
  // std::cout << "BitonicSort: block_size=" << block_size << std::endl;
  if (block_size==1){
    BitonicSort<OSWAP_1, KeyType>(buffer, N, block_size, ascend);
  } else if(block_size==2){
    BitonicSort<OSWAP_2, KeyType>(buffer, N, block_size, ascend);
  } else if(block_size==4){
    BitonicSort<OSWAP_4, KeyType>(buffer, N, block_size, ascend);
  } else if(block_size==8){
    BitonicSort<OSWAP_8, KeyType>(buffer, N, block_size, ascend);
  } else if(block_size==12){
    BitonicSort<OSWAP_12, KeyType>(buffer, N, block_size, ascend);
  } else if (block_size%16==0){
    BitonicSort<OSWAP_16X, KeyType>(buffer, N, block_size, ascend);
  }
  else if (block_size%8==0){
    BitonicSort<OSWAP_8_16X, KeyType>(buffer, N, block_size, ascend);
  } else {
    BitonicSort<OSWAP_ANY, KeyType>(buffer, N, block_size, ascend);
  }
}

template<typename KeyType>
void BitonicSort(KeyType *buffer, size_t N, bool ascend){
  BitonicSort<KeyType>(reinterpret_cast<unsigned char*>(buffer), N, sizeof(KeyType), ascend);
}

/* 
template<OSwap_Style oswap_style, typename KeyType>
void BitonicMerge(unsigned char *keys, size_t N, unsigned char *associated_data1, 
      unsigned char *associated_data2, size_t data_size, bool ascend) {
  if(associated_data1==NULL) {
    if(N<2){
      return;
    }
    else if((N & (N * -1))!=N) {
      size_t M = pow2_lt(N);
      unsigned char *block1 = keys;
      unsigned char *block2 = keys + (M * sizeof(KeyType));
      size_t feasible_swaps = N - M;

      for(size_t i=0; i<feasible_swaps; i++) {
        uint8_t swap_flag = ogt<KeyType>((KeyType*)block1, (KeyType*)block2);
        if(ascend){
          oswap_buffer<oswap_style>(block1, block2, data_size, swap_flag);
        } else {
          oswap_buffer<oswap_style>(block1, block2, data_size, !swap_flag);
        }
        block1+=data_size;
        block2+=data_size; 
      }
   
      BitonicMerge<oswap_style, KeyType>(keys, M, associated_data1, associated_data2, data_size, ascend);
      BitonicMerge<oswap_style, KeyType>(keys + (M * sizeof(KeyType)), N-M, associated_data1, associated_data2, data_size, ascend);
    } 
    else{ //Power of 2 case
      size_t split = N/2;
      unsigned char *block1 = keys;
      unsigned char *block2 = keys + (split * sizeof(KeyType)); 
      
      for(size_t i=0; i<split; i++) {
        uint8_t swap_flag = ogt<KeyType>((KeyType*)block1, (KeyType*)block2);
        if(ascend){
          oswap_buffer<oswap_style>(block1, block2, data_size, swap_flag);
          //ogt_comp_swap((uint64_t *) block1, (uint64_t *) block2, block1, block2, block_size);
        } else {
          oswap_buffer<oswap_style>(block1, block2, data_size, !swap_flag);
          //ogt_comp_swap((uint64_t *) block2, (uint64_t *) block1, block2, block1, block_size);
        }
        block1+=data_size;
        block2+=data_size; 
      } 

      BitonicMerge<oswap_style, KeyType>(keys, split, data_size, ascend);
      BitonicMerge<oswap_style, KeyType>(keys + (split * sizeof(KeyType)), split, data_size, ascend);
    }
  } else{
    if(N<2){
      return;
    }
    else if((N & (N * -1))!=N) {
      size_t M = pow2_lt(N);
      unsigned char *block1 = keys;
      unsigned char *block2 = keys + (M * sizeof(KeyType));
      size_t feasible_swaps = N - M;
      unsigned char *adata1_l = associated_data1;
      unsigned char *adata1_r = associated_data1 + (M * data_size);
      unsigned char *adata2_l = associated_data2;
      unsigned char *adata2_r = associated_data2;
      
      if(associated_data2!=NULL) {
        adata2_r = associated_data2 + (M * data_size);
      }

      for(size_t i=0; i<feasible_swaps; i++) {
        uint8_t swap_flag = ogt<KeyType>((KeyType*)block1, (KeyType*)block2);
        if(ascend){
          oswap_key<KeyType>(block1, block2, swap_flag);
          oswap_buffer<oswap_style>(adata1_l, adata1_r, data_size, swap_flag);
          if(associated_data2!=NULL){
            oswap_buffer<oswap_style>(adata2_l, adata2_r, data_size, swap_flag);
          }
        } else {
          oswap_key<KeyType>(block1, block2, !swap_flag);
          oswap_buffer<oswap_style>(adata1_l, adata1_r, data_size, !swap_flag);
          if(associated_data2!=NULL){
            oswap_buffer<oswap_style>(adata2_l, adata2_r, data_size, !swap_flag);
          }
        }
        block1+=sizeof(KeyType);
        block2+=sizeof(KeyType);
        adata1_l+=data_size;
        adata1_r+=data_size;
        if(associated_data2!=NULL){
          adata2_l+=data_size;
          adata2_r+=data_size;
        }
      }
   
      BitonicMerge<oswap_style, KeyType>(keys, M, associated_data1, associated_data2, data_size, ascend);
      if(associated_data2==NULL)
        BitonicMerge<oswap_style, KeyType>(keys + (M * sizeof(KeyType)), N-M, associated_data1 + (M*data_size), associated_data2, data_size, ascend);
      else
        BitonicMerge<oswap_style, KeyType>(keys + (M * sizeof(KeyType)), N-M, associated_data1 + (M*data_size), associated_data2 + (M*data_size), data_size, ascend);
    } 
    else{ //Power of 2 case
      size_t split = N/2;
      unsigned char *block1 = keys;
      unsigned char *block2 = keys + (split * sizeof(KeyType)); 
      unsigned char *adata1_l = associated_data1;
      unsigned char *adata1_r = associated_data1 + (split * data_size);
      unsigned char *adata2_l = associated_data2;
      unsigned char *adata2_r = associated_data2;
      
      if(associated_data2!=NULL) {
        adata2_r = associated_data2 + (split * data_size);
      }
      
      for(size_t i=0; i<split; i++) {
        uint8_t swap_flag = ogt<KeyType>((KeyType*)block1, (KeyType*)block2);
        if(ascend){
          oswap_key<KeyType>(block1, block2, swap_flag);
          oswap_buffer<oswap_style>(adata1_l, adata1_r, data_size, swap_flag);
          if(associated_data2!=NULL){
            oswap_buffer<oswap_style>(adata2_l, adata2_r, data_size, swap_flag);
          }
        } else {
          oswap_key<KeyType>(block1, block2, !swap_flag);
          oswap_buffer<oswap_style>(adata1_l, adata1_r, data_size, !swap_flag);
          if(associated_data2!=NULL){
            oswap_buffer<oswap_style>(adata2_l, adata2_r, data_size, !swap_flag);
          }
        }
        block1+=sizeof(KeyType);
        block2+=sizeof(KeyType); 
        adata1_l+=data_size;
        adata1_r+=data_size;
        if(associated_data2!=NULL){
          adata2_l+=data_size;
          adata2_r+=data_size;
        }
      } 
      BitonicMerge<oswap_style, KeyType>(keys, split, associated_data1, associated_data2, data_size, ascend);
      if(associated_data2==NULL)
        BitonicMerge<oswap_style, KeyType>(keys + (split * sizeof(KeyType)), N-split, associated_data1 + (split*data_size), associated_data2, data_size, ascend);
      else
        BitonicMerge<oswap_style, KeyType>(keys + (split * sizeof(KeyType)), N-split, associated_data1 + (split*data_size), associated_data2 + (split*data_size), data_size, ascend);
    }
  }
}


template<OSwap_Style oswap_style, typename KeyType = uint64_t>
void BitonicSort(unsigned char *keys, size_t N, unsigned char *associated_data1, 
      unsigned char *associated_data2, size_t data_size, bool ascend) {
  FOAV_SAFE_CNTXT(BitonicSort, N)
  if(N < 2){
    return;
  }
  else {
    size_t N1 = N/2;
    FOAV_SAFE_CNTXT(BitonicSort, associated_data1)
    FOAV_SAFE_CNTXT(BitonicSort, associated_data2)
    if(associated_data1==NULL){
      BitonicSort<oswap_style, KeyType>(keys, N1, associated_data1, associated_data2,
                    data_size, !ascend);
      // Increment keys by N1 data_size blocks, since keys holds the entire buffer to sort.
      BitonicSort<oswap_style, KeyType>(keys + (N1*data_size), N-N1, associated_data1,
                    associated_data2, data_size, ascend);
      BitonicMerge<oswap_style, KeyType>(keys, N, associated_data1, associated_data2,
                    data_size, ascend);
    } else if(associated_data2==NULL){
      //There is only one associated_data list.
      BitonicSort<oswap_style, KeyType>(keys, N1, associated_data1, associated_data2,
                    data_size, !ascend);
      BitonicSort<oswap_style, KeyType>(keys + (N1*sizeof(KeyType)), N-N1, associated_data1 + (N1*data_size),
                    associated_data2, data_size, ascend);
      BitonicMerge<oswap_style, KeyType>(keys, N, associated_data1, associated_data2,
                    data_size, ascend);
      FOAV_SAFE_CNTXT(BitonicSort, associated_data1)
      FOAV_SAFE_CNTXT(BitonicSort, associated_data2)
    } else { 
      //Both associated_data lists.
      BitonicSort<oswap_style, KeyType>(keys, N1, associated_data1, associated_data2,
                    data_size, !ascend);
      BitonicSort<oswap_style, KeyType>(keys + (N1*sizeof(KeyType)), N-N1, associated_data1 + (N1*data_size),
                    associated_data2 + (N1*data_size), data_size, ascend);
      BitonicMerge<oswap_style, KeyType>(keys, N, associated_data1, associated_data2,
                    data_size, ascend);
    }

  }
  
}
 */



#endif
