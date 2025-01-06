#ifndef __WAKSMANNETWORK_HPP__
#define __WAKSMANNETWORK_HPP__

#ifdef ENABLE_WAKSMAN_SHUFFLE 

#include <unordered_map>
#include <vector>
#include "../oasm_lib.h"
// #include "../ObliviousPrimitives.hpp"
#include "../SortingNetwork/SortingNetwork.hpp"
// #include <sgx_tcrypto.h>
// #include "../Enclave_globals.h"
#include "../utils.hpp"
#include "../RecursiveShuffle/RecursiveShuffle.hpp"
#include "../aes.hpp"

typedef __uint128_t randkey_t;
#define FPERM_OSWAP_STYLE OSWAP_8_16X // OSwap_Style for forward perm (consistent w/ randkey_t)

/*
  WaksmanNetwork Class: Contains a Waksman permutation network and can apply it to an input array.
    setPermutation(uint32_t *permutation, unsigned char *forward_perm, [optional preallocated
      memory regions]): Takes permutation as an array of N index values (i.e. values in [N]) and
      optional pointers to allocated memory. It sets the Waksman network switches to that
      permutation.
    applyPermutation(unsigned char *buf, size_t block_size): Takes buffer of N items of block_size
      bytes each and applies stored permutation in-place (i.e. modifying input buffer).
    applyInversePermutation(unsigned char *buf, size_t block_size): Takes buffer of N items of
      block_size bytes each and applies inverse of stored permutation in-place.
*/

class WaksmanNetwork {
  uint32_t Ntotal; // number of items to permute
  std::vector<uint32_t> inSwitchVec; // input layer of (numbered) switches
  std::vector<uint8_t> outSwitchVec; // output layer of switches

  // A struct to keep track of the current subnet number, and input and
  // output switches, for each subnet as we traverse the network.
  struct WNTraversal {
    uint64_t subnetNumber;
    uint32_t *inSwitches;
    uint8_t *outSwitches;

    WNTraversal(WaksmanNetwork &wn) : subnetNumber(0),
        inSwitches(wn.inSwitchVec.data()),
        outSwitches(wn.outSwitchVec.data()) {}
  };

  // A struct to hold pre-allocated memory (and AES keys) so that we
  // only allocate memory once, before the recursive setPermutation is
  // called.
  struct WNMem {
    unsigned char *forward_perm;
    uint32_t *unselected_cnt;
    std::unordered_map<randkey_t, std::pair<uint32_t, uint32_t>> *reverse_perm;
    AESkey forward_key, reverse_key;

    WNMem(WaksmanNetwork &wn) {
        // Round Ntotal up to an even number
        uint32_t Neven = wn.Ntotal + (wn.Ntotal&1);
        forward_perm = new unsigned char[Neven * (sizeof(randkey_t) + 8)];
        unselected_cnt = new uint32_t[wn.Ntotal];
        reverse_perm = new std::unordered_map<randkey_t, std::pair<uint32_t, uint32_t>>;
        __m128i forward_rawkey, reverse_rawkey;
        PRB_buffer::getInstance().getRandomBytes((unsigned char *) &forward_rawkey, sizeof(forward_rawkey));
        PRB_buffer::getInstance().getRandomBytes((unsigned char *) &reverse_rawkey, sizeof(reverse_rawkey));
        AES_128_Key_Expansion(forward_key, forward_rawkey);
        AES_128_Key_Expansion(reverse_key, reverse_rawkey);
    }

    ~WNMem() {
        delete[] forward_perm;
        delete[] unselected_cnt;
        delete reverse_perm;
    }
  };

  void setPermutation(uint32_t *permutation, uint32_t N,
    uint32_t depth, WNTraversal &traversal, const WNMem &mem);

  template <OSwap_Style oswap_style>
  void applyPermutation(unsigned char *buf, uint32_t N,
    size_t block_size, WNTraversal &traversal);

  template <OSwap_Style oswap_style>
  void applyInversePermutation(unsigned char *buf, uint32_t N,
    size_t block_size, WNTraversal &traversal);

public:

  // Set up the WaksmanNetwork for N items.  N need not be a power of 2.
  // N <= 2^31
  WaksmanNetwork(uint32_t N);

  size_t numItems() const { return Ntotal; }

  void setPermutation(uint32_t *permutation);

  template <OSwap_Style oswap_style>
  void applyPermutation(unsigned char *buf, size_t block_size);

  template <OSwap_Style oswap_style>
  void applyInversePermutation(unsigned char *buf, size_t block_size);
};

// Define this to show the intermediate states of applyPermutation
// #define SHOW_APPLYPERM

// Apply permutation encoded by control bits to data elements in buffer. Permutes in place.
template <OSwap_Style oswap_style>
void WaksmanNetwork::applyPermutation(unsigned char *buf, size_t block_size) {
    FOAV_SAFE_CNTXT(AP, Ntotal)
    if (Ntotal > 1) {
        WNTraversal traversal(*this);
        applyPermutation<oswap_style>(buf, Ntotal, block_size, traversal);
    }
}

// Apply permutation encoded by control bits to data elements in buffer. Permutes in place.
template <OSwap_Style oswap_style>
void WaksmanNetwork::applyPermutation(unsigned char *buf, uint32_t N,
    size_t block_size, WNTraversal &traversal) {

  FOAV_SAFE_CNTXT(AP, Ntotal)
  FOAV_SAFE_CNTXT(AP, N)
  if (N < 2) return;

  const uint32_t Nleft = (N+1)/2;
  const uint32_t Nright = N/2;
  const uint32_t numInSwitches = (N-1)/2;
  const uint32_t numOutSwitches = N/2;
  const uint32_t *inSwitch = traversal.inSwitches;
  const uint8_t *outSwitch = traversal.outSwitches;

  traversal.subnetNumber += 1;
  traversal.inSwitches += numInSwitches;
  traversal.outSwitches += numOutSwitches;

#ifdef SHOW_APPLYPERM
  printf("s");
  for(uint32_t i=0;i<N;++i) {
    printf(" %2d", *(uint32_t*)(buf+block_size*i));
  }
  printf("\n");
#endif

  if (N == 2) {
#ifdef SHOW_APPLYPERM
    printf("o");
    for(uint32_t i=0;i<numOutSwitches;++i) {
      printf(" %s", outSwitch[i] ? " X" : "||");
    }
    printf("\n");
#endif
    oswap_buffer<oswap_style>(buf, buf + block_size, (uint32_t) block_size, outSwitch[0]);
#ifdef SHOW_APPLYPERM
    printf("e");
    for(uint32_t i=0;i<N;++i) {
      printf(" %2d", *(uint32_t*)(buf+block_size*i));
    }
    printf("\n");
#endif
  } else {
#ifdef SHOW_APPLYPERM
    printf("i");
    for(uint32_t i=0;i<numInSwitches;++i) {
      printf(" %s", (inSwitch[i]&1) ? " X" : "||");
    }
    printf("\n");
#endif
    // Apply input switches to permutation
    const uint32_t *curInSwitchVal = inSwitch;
    for (uint32_t i=0; i<numInSwitches; i++) {
      oswap_buffer<oswap_style>(buf + block_size*(i), buf + block_size*(Nleft+i), block_size,
        (*curInSwitchVal)&1);
      curInSwitchVal += 1;
    }
#ifdef SHOW_APPLYPERM
    printf(" ");
    for(uint32_t i=0;i<N;++i) {
      printf(" %2d", *(uint32_t*)(buf+block_size*i));
    }
    printf("\n");
#endif

    // Apply subnetwork switches
    applyPermutation<oswap_style>(buf, Nleft, block_size, traversal);
    applyPermutation<oswap_style>(buf + block_size*Nleft, Nright,
        block_size, traversal);

#ifdef SHOW_APPLYPERM
    printf("r");
    for(uint32_t i=0;i<N;++i) {
      printf(" %2d", *(uint32_t*)(buf+block_size*i));
    }
    printf("\n");
    printf("o");
    for(uint32_t i=0;i<numOutSwitches;++i) {
      printf(" %s", outSwitch[i] ? " X" : "||");
    }
    printf("\n");
#endif
    // Apply output switches to permutation
    for (uint32_t i=0; i<numOutSwitches; i++) {
      oswap_buffer<oswap_style>(buf + block_size*i, buf + block_size*(Nleft+i), block_size,
        *outSwitch);
      ++outSwitch;
    }
#ifdef SHOW_APPLYPERM
    printf("e");
    for(uint32_t i=0;i<N;++i) {
      printf(" %2d", *(uint32_t*)(buf+block_size*i));
    }
    printf("\n");
#endif
  }
}


// Apply inverse of permutation in control bits to data elements in buffer. Permutes in place.
template <OSwap_Style oswap_style>
void WaksmanNetwork::applyInversePermutation(unsigned char *buf, size_t block_size) {
    FOAV_SAFE_CNTXT(AIP, Ntotal)
    if (Ntotal > 1) {
        WNTraversal traversal(*this);
        applyInversePermutation<oswap_style>(buf, Ntotal, block_size,
            traversal);
    }
}

// Apply inverse of permutation in control bits to data elements in buffer. Permutes in place.
template <OSwap_Style oswap_style>
void WaksmanNetwork::applyInversePermutation(unsigned char *buf,
    uint32_t N, size_t block_size, WNTraversal &traversal) {
  FOAV_SAFE_CNTXT(AIP, N)
  if (N < 2) return;

  const uint32_t Nleft = (N+1)/2;
  const uint32_t Nright = N/2;
  const uint32_t numInSwitches = (N-1)/2;
  const uint32_t numOutSwitches = N/2;
  const uint32_t *inSwitch = traversal.inSwitches;
  const uint8_t *outSwitch = traversal.outSwitches;

  traversal.subnetNumber += 1;
  traversal.inSwitches += numInSwitches;
  traversal.outSwitches += numOutSwitches;

#ifdef SHOW_APPLYPERM
  printf("s");
  for(uint32_t i=0;i<N;++i) {
    printf(" %2d", *(uint32_t*)(buf+block_size*i));
  }
  printf("\n");
#endif

  FOAV_SAFE_CNTXT(AIP, N)
  if (N == 2) {
#ifdef SHOW_APPLYPERM
    printf("o");
    for(uint32_t i=0;i<numOutSwitches;++i) {
      printf(" %s", outSwitch[i] ? " X" : "||");
    }
    printf("\n");
#endif
    oswap_buffer<oswap_style>(buf, buf + block_size, (uint32_t) block_size,
        outSwitch[0]);
#ifdef SHOW_APPLYPERM
    printf("e");
    for(uint32_t i=0;i<N;++i) {
      printf(" %2d", *(uint32_t*)(buf+block_size*i));
    }
    printf("\n");
#endif
  } else {
    // Apply output switches to permutation
#ifdef SHOW_APPLYPERM
    printf("o");
    for(uint32_t i=0;i<numOutSwitches;++i) {
      printf(" %s", outSwitch[i] ? " X" : "||");
    }
    printf("\n");
#endif
    FOAV_SAFE_CNTXT(AIP, numOutSwitches)
    for (uint32_t i=0; i<numOutSwitches; i++) {
    FOAV_SAFE2_CNTXT(AIP, i, numOutSwitches)
      oswap_buffer<oswap_style>(buf + block_size*i, buf + block_size*(Nleft+i), block_size,
        *outSwitch);
      ++outSwitch;
    }
#ifdef SHOW_APPLYPERM
    printf(" ");
    for(uint32_t i=0;i<N;++i) {
      printf(" %2d", *(uint32_t*)(buf+block_size*i));
    }
    printf("\n");
#endif

    // Apply subnetwork switches
    applyInversePermutation<oswap_style>(buf, Nleft,
        block_size, traversal);
    applyInversePermutation<oswap_style>(buf + block_size*Nleft, Nright,
        block_size, traversal);

    // Apply input switches to permutation
#ifdef SHOW_APPLYPERM
    printf("r");
    for(uint32_t i=0;i<N;++i) {
      printf(" %2d", *(uint32_t*)(buf+block_size*i));
    }
    printf("\n");
    printf("i");
    for(uint32_t i=0;i<numInSwitches;++i) {
      printf(" %s", (inSwitch[i]&1) ? " X" : "||");
    }
    printf("\n");
#endif
    const uint32_t *curInSwitchVal = inSwitch;
    FOAV_SAFE_CNTXT(AIP, numInSwitches)
    for (uint32_t i=0; i<numInSwitches; i++) {
    FOAV_SAFE2_CNTXT(AIP, i, numInSwitches)
      oswap_buffer<oswap_style>(buf + block_size*(i), buf + block_size*(Nleft+i), block_size,
        (*curInSwitchVal&1));
      curInSwitchVal += 1;
    }
#ifdef SHOW_APPLYPERM
    printf("e");
    for(uint32_t i=0;i<N;++i) {
      printf(" %2d", *(uint32_t*)(buf+block_size*i));
    }
    printf("\n");
#endif
  }
}


// void OblivWaksmanShuffle(unsigned char *buffer, uint32_t N, size_t block_size, enc_ret *ret);

// void DecryptAndOblivWaksmanShuffle(unsigned char *encrypted_buffer, uint32_t N,
//   size_t encrypted_block_size, unsigned char *result_buffer, enc_ret *ret);

// void DecryptAndOWSS(unsigned char *encrypted_buffer, uint32_t N,
//   size_t encrypted_block_size, unsigned char *result_buffer, enc_ret *ret);


template<typename T>
void generateRandomPermutation(size_t N, T *random_permutation){
  //Initialize random permutation as 1,...,N
  for(size_t i=0; i<N; i++) {
    random_permutation[i]=i;
  }

  //Convert it to a random permutation of [1,N] 
  RecursiveShuffle_M1((unsigned char*) random_permutation, N, sizeof(T));

  /*
  printf("\nPermutation output\n");
  for(T i=0; i<N; i++)
    printf("%ld, ", random_permutation[i]);
  printf("\n");
  */
}

#endif

#endif
