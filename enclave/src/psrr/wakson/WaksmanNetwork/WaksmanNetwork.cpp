
#include "psrr/wakson/WaksmanNetwork/WaksmanNetwork.hpp"
#ifdef ENABLE_WAKSMAN_SHUFFLE 

// Count the number of input and output switches, and the number of
// WaksmanSubnetworks, used to handle N items.  Add the numbers to the
// numInSwitches, numOutSwitches, and numSubnetworks parameters.
static void countSwitches(uint32_t N, size_t &numInSwitches,
    size_t &numOutSwitches, size_t &numSubnetworks)
{
    ++numSubnetworks;
    // Base cases
    FOAV_SAFE_CNTXT(countswitches, N)
    if (N < 2) {
        return;
    } else if (N == 2) {
        ++numOutSwitches;
        return;
    }

    // How many switches do we use ourselves?

    // If N is even, we use (N/2)-1 input and N/2 output switches
    // If N is odd, we use (N-1)/2 input and (N-1)/2 output switches
    // Note that with integer division, both cases can be handled by
    // computing (N-1)/2 input and N/2 output switches.
    numInSwitches += (N-1)/2;
    numOutSwitches += N/2;

    // Then recurse into the two children. If N is even, we divide in
    // half.  If N is odd, the left child will have the extra entry.
    countSwitches((N+1)/2, numInSwitches, numOutSwitches, numSubnetworks);
    countSwitches(N/2, numInSwitches, numOutSwitches, numSubnetworks);
}

WaksmanNetwork::WaksmanNetwork(uint32_t N) : Ntotal(N) {
    size_t numInSwitches = 0, numOutSwitches = 0, numSubnetworks = 0;
    countSwitches(N, numInSwitches, numOutSwitches, numSubnetworks);
    inSwitchVec.resize(numInSwitches);
    outSwitchVec.resize(numOutSwitches);
}

/* Intialize data structure counting unselected permutation mappings for fast random selection.
  Call initially with empty=true - argument only needed for recursive calls.
*/
static inline void initUnselectedCnt(uint32_t *unselected_cnt, uint32_t num_vals, bool empty = true) {
  FOAV_SAFE2_CNTXT(countswitches, num_vals, empty)
  if (num_vals == 0) { // Check just in case - this should never be called with num_vals == 0.
    return;
  }
  if (empty == true) {
    unselected_cnt[num_vals-1] = num_vals;
  }
  if (num_vals == 1) {
    return;
  }
  uint32_t num_left = (num_vals+1)/2;
  initUnselectedCnt(unselected_cnt, num_left, true);
  initUnselectedCnt(unselected_cnt+num_left, num_vals - num_left, false);
}


/* Modifies unselected_cnt to indicate item at index has been selected.
  Call initially with unadjusted=true - argument only needed for recursive calls.
*/
static inline void updateUnselectedCnt(uint32_t *unselected_cnt, uint32_t num_vals, uint32_t index,
  bool unadjusted = true) {
  FOAV_SAFE2_CNTXT(updateUnselectedCnt, num_vals, unadjusted)
  if (num_vals == 0) { // Check just in case - this should never be called with num_vals == 0.
    return;
  }
  FOAV_SAFE2_CNTXT(updateUnselectedCnt, num_vals, unadjusted)
  if (unadjusted == true) {
    unselected_cnt[num_vals-1]--;
  }
  FOAV_SAFE2_CNTXT(updateUnselectedCnt, num_vals, unadjusted)
  if (num_vals == 1) {
    return;
  }
  uint32_t num_left = (num_vals+1)/2;
  FOAV_SAFE2_CNTXT(updateUnselectedCnt, index, num_left) 
  if (index < num_left) {
    updateUnselectedCnt(unselected_cnt, num_left, index, true);
  } else {
    updateUnselectedCnt(unselected_cnt+num_left, num_vals - num_left, index-num_left, false);
  }
}


/* Computes pseudo-random permutation (PRP), __uint128_t -> __uint128_t.
   The input is the 128-bit integer with in_high in the top 64 bits and
   in_low in the lower 64 bits.
*/
static inline __uint128_t prp128(const AESkey &aeskey,
    uint64_t in_high, uint64_t in_low) {
  __m128i ciphertext;
  AES_ECB_encrypt(ciphertext, _mm_set_epi64x(in_high,in_low), aeskey);

  return reinterpret_cast<__uint128_t>(ciphertext);
}


void print_u128(__uint128_t x) {
  unsigned char *c = ((unsigned char *) &x) + sizeof(__uint128_t) - 1;
  for (int i=0; i<sizeof(__uint128_t); i++) {
    printf("%.2hhx", *c);
    c--;
  }
}


/* Look up either (1) permutation mapping corresponding to hash, or (2) random unselected mapping.
  Returns index into forward_perm pointing to the mapping looked up.
*/
static inline uint32_t permOrRand(uint32_t N, unsigned char *forward_perm, randkey_t hashval, uint32_t *unselected_cnt,
  uint8_t rand_flag) {
  uint32_t rand_bytes;
  uint32_t start = 0;
  uint32_t end = N-1;
  uint32_t mid;
  uint8_t hash_dir;
  uint8_t rand_dir;
  uint32_t tot_unselected_cnt = unselected_cnt[end];
  uint32_t left_unselected_cnt;
  uint64_t rand_val;

  PRB_buffer::getInstance().getRandomBytes((unsigned char *) &rand_bytes, sizeof(uint32_t));
  rand_val = tot_unselected_cnt * rand_bytes;
  rand_val >>= 32;

  while (true) {
    FOAV_SAFE_CNTXT(permOrRand, start)
    FOAV_SAFE_CNTXT(permOrRand, end)
    if (start == end) {
      return start;
    }
    mid = (start+end)/2;
    // Compare desired hash value to hash value just after the current midpoint
    hash_dir = ogt<randkey_t>((randkey_t *) (forward_perm + ((mid+1)*(sizeof(randkey_t) + 8))),
      &hashval);
    // Compare random unselected value to number unselected in left half
    left_unselected_cnt = unselected_cnt[mid];
    rand_dir = ogt_set_flag(left_unselected_cnt, rand_val);
    // Pick between hash_dir and rand_dir based on rand_flag
    bool f1 = ((1-rand_flag) & hash_dir);
    bool f2 = (rand_flag & rand_dir);
    FOAV_SAFE_CNTXT(permOrRand, f1)
    FOAV_SAFE_CNTXT(permOrRand, f2)
    if ((f1 | f2) == 1) {
      end = mid;
      tot_unselected_cnt = left_unselected_cnt;
    } else {
      start = mid+1;
      tot_unselected_cnt -= left_unselected_cnt;
      rand_val -= left_unselected_cnt;
    }
  }
}

// If this is defined, set it to the smallest N you want to see
// profiling data for
// #define PROFILE_SETPERM_N 32768

// Define this to show the intermediate states of setPermutation
// #define SHOW_SETPERM

// Produce the partner of x; that is, x+Nleft if x < Nleft, or x-Nleft
// if x >= Nleft
static inline uint32_t PARTNER(uint32_t x, uint32_t Nleft)
{
    uint32_t side = (x >= Nleft) * (Nleft<<1);
    return x + Nleft - side;
}

// The elements of the permutation array start off as just 32-bit
// integers, where if j = permutation[i], then the item in position i
// will move to position j.  This will sort the permutation; that is, it
// will apply the inverse of the given permutation.  So if we want to
// apply the given permutation, we will first use the permutation to set
// the control bits of the Waksman network in a way that will sort it,
// and then apply the inverse permutation by applying the Waksman
// switches in reverse order.

// The strategy of setPermutation is as follows. The invariant is that
// we are given as input a permutation of 0..2k-1, and we will set the
// Waksman network control bits to output the sorted list 0..2k-1.  (If
// we are given an input of odd length, so a permutation of 0..2k-2, we
// implicitly append an entry permutation[2k-1] = 2k-1 to it.)  We then
// find a setting of the k-1 input switches (switch i OSWAPs
// permutation[i] with permutation[i+k] for i=0..k-2; permutation[k-1]
// never gets swapped, whether or not there was a permutation[2k-1] in
// the original input) such that permutation[0..k-1] mod k ends up being
// a permutation of 0..k-1, and permutation[k..2k-1] mod k ends up being
// a permultation of 0..k-1.  (If we were given an odd input initially,
// then it will necessarily be the case that permutation[2k-1] = 2k-1
// and so permutation[2k-1] mod k = k-1, so permutation[k..2k-2] mod k
// will be a permutation of 0..k-2.) We recurse on the left and the
// right, which will set the input and output switches of the
// subnetworks such that, after applying the switches, on the left,
// permutation[0..k-1] mod k will be 0..k-1 (in order), and similarly on
// the right either permutation[k..2k-2] mod k or permutation[k..2k-1]
// mod k, depending on the input size, will be 0..k-2 or 0..k-1
// respectively.

// Then we set the k-1 or k output switches (depending on the length of
// the right side), where switch i again OSWAPs permutation[i] with
// permutation[i+k].  Note that both of these values will necessarily be
// i mod k at this point, so the switch just has to be set to the "high
// bit" of permutation[i]; that is, the bit that is 1 iff
// (permutation[i] >= k).  This will yield the desired sorted list.

// Note that when recursing, we only consider the permutation values mod
// k, but we need to remember whether the value v represented the
// original v or the original v+k, so that we can use that bit to set
// output switch v correctly.  To keep track of this, when we recurse,
// for each v = permutation[i] in the array, we attach to it a stack of
// the "high bits" it's gone through so far (initially empty).  At
// recursive depth d (the initial call is d=0), we have the values in
// the permutation array being (v, [b_0, ..., b_{d-1}]) where v is an
// integer 0 <= v <= 2k-1, and each b_i is a bit.  When we recurse to
// depth d+1, we push the new high bit onto the stack (the top of the
// stack is on the right in this notation), to yield (v mod k, [b_0,
// ..., b_{d-1}, b_d]).  The recursive call then uses the v mod k values,
// which, as above, will be a permutation of 0..k-1.  When the
// recursions finish, the topmost high bit on the stack will be popped
// off to yield (v mod k + b_d*k (= v), [b_0, ..., b_{d-1}]).

// The way we actually internally represent the value at depth d
// (v, [b_0, ..., b_{d-1}]) is by packing that into a single integer,
// with v followed by the d bits: x = v<<d | b_0<<(d-1) | ... | b_{d-1}.

// For example, suppose initially we have N=14, and v = permutation[i] =
// 12.  Then the initial representation of v (with depth d=0) is just
// x = v = 12.
// At the first level, k=7, so when we recurse, (12, []) will become
// x = (12 mod 7, [(12 >= 7)]) = (5, [1]) for d=1, which we represent as
// [101][1] in binary (brackets for clarity only) = 11.  At the next
// level, k=4 and d=2, so x = (5 mod 4, [1, (5 >= 4)]) = (1, [1,1]),
// which we represent as [1][11] = 7.  At the next level (suppose this
// entry ends up in the left recursion), k=2 and d=3, so x = (1 mod 2,
// [1, 1, (1 >= 2)]) = (1, [1,1,0]), which we represent as [1][110] =
// 14.  When k<4, there are no more recursive calls.  As each layer of
// recursion ends, at k=2 and d=3, 14 = [1][110] becomes [(1+0*2)][11] =
// [1][11] = 7. At k=4 and d=2, [1][11] becomes [(1+1*4)][1] = [101][1]
// = 11. At k=7 and d=1, [101][1] becomes [(5+1*7)][] = 12.

// The following functions manipulate this representation.  Note that
// they must all be oblivious to x, but need not be to depth or k.

// Return the value v encoded in the representation x at depth d
static inline uint32_t GET(uint32_t x, uint32_t depth)
{
    return x>>depth;
}

// Turn a representation x of a value v between 0 and 2k-1 at depth d
// (so with d extra bits) into one at depth d+1 (with v between 0 and
// k-1). Pass kd = k<<d. k will be Nleft.
static inline uint32_t PUSH(uint32_t x, uint32_t kd)
{
    // If the effective value is v and the d extra bits are s,
    // then x = v<<d | s.  We want to turn that into
    // ((v%k) << (d+1)) | (s<<1) | (floor(v/k))
    // Recall v < 2*k, so floor(v/k) is the bit b that indicates
    // whether v >= k, or equivalently, that x >= (k<<d)
    uint32_t b = (x >= kd);
    // Now (v%k) = (v - b*k), which avoids taking a potentially
    // non-oblivious mod.  So ((v%k) << (d+1)) | (s<<1) | b
    // = (((v%k) << d) | s) << 1 | b
    // = (((v - b*k)<<d) | s) << 1 | b
    // = (((v<<d)|s) - ((b*k)<<d)) << 1 | b
    // = (x - ((b*k)<<d)) << 1 | b
    // = ((x<<1) - ((b*k)<<(d+1))) | b
    // = ((x<<1) - b*(k<<(d+1)) | b
    x = ((x<<1) - b*(kd<<1)) | b;
    return x;
}

// Turn a representation x of a value v between 0 and k-1 at depth d+1
// (so with d+1 extra bits) into one at depth d (with x between 0 and
// 2*k-1). It should always be that POP(PUSH(x, d, k), d, k) = x
// whenever 0 <= x < (k<<(d+1)). Pass kd = k<<d.  k weill be Nleft.
static inline uint32_t POP(uint32_t x, uint32_t kd)
{
    uint32_t b = x&1;
    x = (x>>1) + b*kd;
    return x;
}


/* Input:
    permutation: points to array of integers 0, ..., N-1 in some order, indicating i->permutation[i]
  Note: This function modifies the input permutation (and actually sorts it).
*/
void WaksmanNetwork::setPermutation(uint32_t *permutation) {
    FOAV_SAFE_CNTXT(WN_SetPerm, Ntotal)
    if (Ntotal > 1) {
        WNTraversal traversal(*this);
        WNMem mem(*this);
        setPermutation(permutation, Ntotal, 0, traversal, mem);
    }
}

/* Input:
    permutation: points to array of integers 0, ..., N-1 in some order, indicating i->permutation[i]
  Note: This function modifies the input permutation (and actually sorts it).
*/
void WaksmanNetwork::setPermutation(uint32_t *permutation, uint32_t N,
  uint32_t depth, WNTraversal &traversal, const WNMem &mem) {
  //printf("Start setPermutation(): N=%d\n", N);

#ifdef SHOW_SETPERM
  printf("S");
  for(uint32_t i=0;i<N;++i) {
    printf(" %2d", permutation[i]);
  }
  printf("\n ");
  for(uint32_t i=0;i<N;++i) {
    printf(" %2d", GET(permutation[i], depth));
  }
  printf("\n");
#endif

  // Handle N<=2 as special cases

  FOAV_SAFE_CNTXT(setPermutation, N)
  if (N < 2) return;

  traversal.subnetNumber += 1;

  FOAV_SAFE_CNTXT(setPermutation, N)
  if (N == 2) {
    // Store output switch value
    traversal.outSwitches[0] = GET(permutation[0], depth);
    //printf("Set outSwitches[0] to %d\n", outSwitches[0]);
    // Apply output switch
    oswap_buffer<OSWAP_4>((unsigned char *) permutation,
      (unsigned char *) (permutation + 1), 4, traversal.outSwitches[0]);
#ifdef SHOW_SETPERM
    printf("O");
    for(uint32_t i=0;i<N/2;++i) {
      printf(" %s", traversal.outSwitches[i] ? " X" : "||");
    }
    printf("\n");

    printf("E");
    for(uint32_t i=0;i<N;++i) {
      printf(" %2d", permutation[i]);
    }
    printf("\n ");
    for(uint32_t i=0;i<N;++i) {
      printf(" %2d", GET(permutation[i],depth));
    }
    printf("\n");
#endif
    traversal.outSwitches += 1;
    return;
  }

#ifdef PROFILE_SETPERM_N
  unsigned long prof_all, prof_before, prof_flt, prof_sflt, prof_unsel, prof_rlt,
    prof_setsw, prof_srtsw, prof_appsw, prof_rec1, prof_rec2, prof_outsw;

  if (N >= PROFILE_SETPERM_N) {
    prof_all = printf_with_rtclock("begin setPermutation N=%u\n", N);
    prof_before = printf_with_rtclock("begin before recursion N=%u\n", N);
  }
#endif

  // The size of the left recursive half.  If N is odd, this is the
  // larger half
  const uint32_t Nleft = (N+1)/2;
  // The size of the right recursive half.  This is also the number of
  // output switches.
  const uint32_t Nright = N/2;

  // N, rounded up to an even number
  const uint32_t Neven = (Nleft<<1);

  if (N > 4) {
#ifdef PROFILE_SETPERM_N
  if (N >= PROFILE_SETPERM_N) {
    prof_flt = printf_with_rtclock("begin forward lookup table N=%u\n", N);
  }
#endif

  const uint64_t snNum = traversal.subnetNumber;

  // Create forward lookup using pseudorandom permutation (PRP)
  // Produced as PRP(i)->(i, GET(permutation[i])) sorted by PRP(i)
  // Note: i and permutation[i] are represented as uint32_t values to pack into one uint64_t
  unsigned char *cur_forward_hash = mem.forward_perm;
  uint32_t *cur_forward_map = (uint32_t *) (mem.forward_perm + sizeof(randkey_t));
  // Generate key for forward-lookup PRP
  __uint128_t forward_perm_hash;
  //printf("Creating forward lookup table\n");
  for (uint32_t i=0; i<Neven; i++) {
    forward_perm_hash = prp128(mem.forward_key, snNum, (uint64_t) i);
    FOAV_SAFE_CNTXT(setPermutation, snNum)
    FOAV_SAFE_CNTXT(setPermutation, forward_perm_hash)
    memcpy(cur_forward_hash, &forward_perm_hash, sizeof(__uint128_t));
    cur_forward_hash += sizeof(randkey_t) + 8;
    *cur_forward_map = i;
    cur_forward_map += 1;
    *cur_forward_map = i < N ? GET(permutation[i], depth) : N;
    cur_forward_map = (uint32_t *) (cur_forward_hash + sizeof(randkey_t));
  }

#ifdef PROFILE_SETPERM_N
  if (N >= PROFILE_SETPERM_N) {
    printf_with_rtclock_diff(prof_flt, "end forward lookup table N=%u\n", N);
    prof_sflt = printf_with_rtclock("begin sort forward lookup table N=%u\n", N);
  }
#endif
  BitonicSort<FPERM_OSWAP_STYLE, randkey_t>(mem.forward_perm, (size_t) N, sizeof(randkey_t) + 8, true);
  // Print forward lookup table
  /*
  unsigned char *tmp_cur_forward_hash = forward_perm;
  uint32_t *tmp_cur_forward_map = (uint32_t *) (forward_perm + sizeof(randkey_t));
  __uint128_t tmp_forward_perm_hash;
  for (uint32_t i=0; i<N; i++) {
    memcpy(&tmp_forward_perm_hash, tmp_cur_forward_hash, sizeof(__uint128_t));
    printf("\t (");
    print_u128(tmp_forward_perm_hash);
    printf(") %d -> %d\n", *tmp_cur_forward_map, *(tmp_cur_forward_map+1));
    tmp_cur_forward_hash += sizeof(randkey_t) + 8;
    tmp_cur_forward_map = (uint32_t *) (tmp_cur_forward_hash + sizeof(randkey_t));
  }
  */

#ifdef PROFILE_SETPERM_N
  if (N >= PROFILE_SETPERM_N) {
    printf_with_rtclock_diff(prof_sflt, "end sort forward lookup table N=%u\n", N);
    prof_unsel = printf_with_rtclock("begin unselected count N=%u\n", N);
  }
#endif
  // Create cumulative count of unselected items
  initUnselectedCnt(mem.unselected_cnt, N);
#ifdef PROFILE_SETPERM_N
  if (N >= PROFILE_SETPERM_N) {
    printf_with_rtclock_diff(prof_unsel, "end unselected count N=%u\n", N);
    prof_rlt = printf_with_rtclock("begin reverse lookup table N=%u\n", N);
  }
#endif

  // Create reverse lookup using hash table
  // Maps \pi(i) to i and index of i->\pi(i) in forward_perm
  mem.reverse_perm->reserve(N);

  // Lookup done on keyed hash of \pi(i) with reverse key
  cur_forward_hash = mem.forward_perm;
  cur_forward_map = (uint32_t *) (mem.forward_perm + sizeof(randkey_t));
  randkey_t reverse_perm_hash;
  //printf("Creating reverse-permutation hash table\n");
  FOAV_SAFE_CNTXT(setPermutation, Neven)
  for (uint32_t i=0; i<Neven; i++) {
    FOAV_SAFE_CNTXT(setPermutation, i)
    reverse_perm_hash = prp128(mem.reverse_key, snNum, (uint64_t) *(cur_forward_map+1));
    FOAV_SAFE_CNTXT(setPermutation, snNum)
    FOAV_SAFE_CNTXT(setPermutation, reverse_perm_hash)
    std::pair<uint32_t, uint32_t> reverse_val(*cur_forward_map, i);
    //printf("Inserting prp128(%d) = ", *(cur_forward_map+1));
    //print_u128(reverse_perm_hash);
    //printf(" -> (%d, %d)\n", reverse_val.first, reverse_val.second);
    mem.reverse_perm->insert(std::make_pair(reverse_perm_hash, reverse_val));
    cur_forward_hash += sizeof(randkey_t) + 8;
    cur_forward_map = (uint32_t *) (cur_forward_hash + sizeof(randkey_t));
  }
#ifdef PROFILE_SETPERM_N
  if (N >= PROFILE_SETPERM_N) {
    printf_with_rtclock_diff(prof_rlt, "end reverse lookup table N=%u\n", N);
    prof_setsw = printf_with_rtclock("begin set switches N=%u\n", N);
  }
#endif

  // Set input switch values
  uint32_t cycle_start = Neven-1; // start of current permutation cycle
  uint32_t forward = 0; // item defining switch to set
  uint32_t forward_partner; // forward permutation "partner" (i.e. same input switch)
  randkey_t forward_partner_hash;
  uint32_t perm_idx;
  uint32_t forward_partner_map; // permutation map applied to forward partner
  uint32_t switch_num;
  uint32_t switch_val;
  uint32_t forward_partner_map_partner; // "partner" (i.e. same residue class) of forward_partner_map
  randkey_t forward_partner_map_partner_hash;
  //const uint32_t input_switch_bit = N >> 1; // bit pattern determining input switch partners
  //const uint32_t switch_mask = (N-1) >> 1; // mask to compute input switch number via AND
  //const uint32_t crp_xor = N >> 1; // bit pattern to compute composite residue partner via XOR
  uint32_t *cur_switch = traversal.inSwitches;
  uint8_t rand_flag = 0; // Indicate if next forward lookup should be random (due to cycle end)

  // Perform first back-and-forth lookups on items Neven-1 and Nleft-1, which have no input switch
  //printf("forward = %d\n", Neven-1);
  //printf("forward partner = %d\n", Nleft-1);
  forward_partner_hash = (randkey_t) prp128(mem.forward_key, snNum, Nleft-1);
  FOAV_SAFE_CNTXT(setPermutation, snNum)
  FOAV_SAFE_CNTXT(setPermutation, forward_partner_hash)
  perm_idx = permOrRand(N, mem.forward_perm, forward_partner_hash, mem.unselected_cnt, 0);
  cur_forward_map = (uint32_t *) (mem.forward_perm + perm_idx*(sizeof(randkey_t) + 8) +
    sizeof(randkey_t));
  forward_partner_map = *(cur_forward_map+1);
  //printf("forward_partner_map = %d\n", forward_partner_map);
  updateUnselectedCnt(mem.unselected_cnt, N, perm_idx);
  forward_partner_map_partner = PARTNER(forward_partner_map, Nleft);
  //printf("forward_partner_map_partner = %d\n", forward_partner_map_partner);
  forward_partner_map_partner_hash = prp128(mem.reverse_key, snNum, (uint64_t) forward_partner_map_partner);
  FOAV_SAFE_CNTXT(setPermutation, snNum)
  FOAV_SAFE_CNTXT(setPermutation, forward_partner_map_partner_hash)

  //printf("looking up ");
  //print_u128(forward_partner_map_partner_hash);
  //printf("\n");
  std::pair<uint32_t, uint32_t>& reverse_perm_ret = mem.reverse_perm->at(forward_partner_map_partner_hash);
  forward = reverse_perm_ret.first;
  perm_idx = reverse_perm_ret.second;
  updateUnselectedCnt(mem.unselected_cnt, N, perm_idx);
  rand_flag = oe_set_flag(forward, cycle_start);

  // Perform remaining back-and-forth lookups and input switch settings
  for (uint32_t i=0; i<Nleft-1; i++) {
    //printf("forward = %d\n", forward);
    // Forward map partner (ignored if random lookup)
    forward_partner = PARTNER(forward, Nleft);
    //printf("forward partner = %d\n", forward_partner);
    // Either map forward_partner under permutation or perform random lookup
    forward_partner_hash = (randkey_t) prp128(mem.forward_key, snNum, (uint64_t) forward_partner);
    FOAV_SAFE_CNTXT(setPermutation, snNum)
    FOAV_SAFE_CNTXT(setPermutation, forward_partner_hash)
    perm_idx = permOrRand(N, mem.forward_perm, forward_partner_hash, mem.unselected_cnt, rand_flag);
    cur_forward_map = (uint32_t *) (mem.forward_perm + perm_idx*(sizeof(randkey_t) + 8) +
      sizeof(randkey_t));
    forward_partner_map = *(cur_forward_map+1);
    //printf("forward_partner_map = %d\n", forward_partner_map);
    // update unselected_cnt with forward lookup
    updateUnselectedCnt(mem.unselected_cnt, N, perm_idx);
    // Write out current switch setting (need to do after potentially random permOrRand lookup)
    switch_val = ((*cur_forward_map) >= Nleft); // value of current input switch
    //printf("switch_val = %d\n", switch_val);
    switch_num = (*cur_forward_map) - switch_val * Nleft; // number of current input switch
    //printf("switch_num = %d\n", switch_num);
    *cur_switch = (switch_num<<1) | switch_val;
    cur_switch++;
    // If random, update cycle_start
    oset_value_uint32_t(&cycle_start, PARTNER((*cur_forward_map),Nleft), rand_flag);
    // Reverse map the residue-class partner
    forward_partner_map_partner = PARTNER(forward_partner_map, Nleft);
    //printf("forward_partner_map_partner = %d\n", forward_partner_map_partner);
    forward_partner_map_partner_hash = prp128(mem.reverse_key, snNum, (uint64_t) forward_partner_map_partner);
    FOAV_SAFE_CNTXT(setPermutation, snNum)
    FOAV_SAFE_CNTXT(setPermutation, forward_partner_map_partner_hash)
    std::pair<uint32_t, uint32_t>& reverse_perm_ret = mem.reverse_perm->at(forward_partner_map_partner_hash);
    forward = reverse_perm_ret.first;
    perm_idx = reverse_perm_ret.second;
    //printf("forward = %d, perm_idx = %d\n", forward, perm_idx);
    // Update unselected_cnt with reverse lookup
    updateUnselectedCnt(mem.unselected_cnt, N, perm_idx);
    // Indicate random lookup needed if cycle start has been reached
    rand_flag = 0; // Needed because oe_set_flag() only sets (i.e. doesn't unset)
    rand_flag = oe_set_flag(forward, cycle_start);
    //printf("rand_flag = %d\n", rand_flag);
  }
  // Clear reverse lookup for use by any recursive call
  mem.reverse_perm->clear();

#ifdef PROFILE_SETPERM_N
  if (N >= PROFILE_SETPERM_N) {
    printf_with_rtclock_diff(prof_setsw, "end set switches N=%u\n", N);
    prof_srtsw = printf_with_rtclock("begin sort switches N=%u\n", N);
  }
#endif
  // Put switches in order
  BitonicSort<OSWAP_4, uint32_t>((unsigned char *) traversal.inSwitches,
    (size_t) Nleft-1, 4, true);
#ifdef PROFILE_SETPERM_N
  if (N >= PROFILE_SETPERM_N) {
    printf_with_rtclock_diff(prof_srtsw, "end sort switches N=%u\n", N);
  }
#endif
  // Print switches
  /*
  printf("Switch\tVal\n");
  cur_switch = (uint32_t *) inSwitches.data();
  for (uint64_t i : inSwitches) {
    printf("%d\t%d\n", (*cur_switch)>>1, *(cur_switch)&1);
    cur_switch += 1;
  }
  */
  } else {
    // N == 3 or N == 4
    // If (GET(permutation[0]) & 1) == (GET(permutation[1]) & 1), set
    // the switch to 1 (so that permutation[0] and permutation[2] get
    // swapped, otherwise 0.  The switch setting is actually stored in
    // the low bit of inSwitches[0].
    traversal.inSwitches[0] = uint64_t((GET(permutation[0],depth) ^
        GET(permutation[1],depth) ^ 1) & 1);
  }
#ifdef PROFILE_SETPERM_N
  if (N >= PROFILE_SETPERM_N) {
    prof_appsw = printf_with_rtclock("begin apply switches N=%u\n", N);
  }
#endif
#ifdef SHOW_SETPERM
  printf("I");
  for(uint32_t i=0;i<Nleft-1;++i) {
    printf(" %s", (traversal.inSwitches[i]&1) ? " X" : "||");
  }
  printf("\n");
#endif

  // Apply input switches to permutation
  uint32_t *cur_switch_val = traversal.inSwitches;
  uint32_t kd = Nleft << depth;
  FOAV_SAFE_CNTXT(setPermutation, Nleft)
  for (uint32_t i=0; i<Nleft-1; i++) {
    FOAV_SAFE2_CNTXT(setPermutation, i, Nleft)
    permutation[i] = PUSH(permutation[i], kd);
    permutation[i+Nleft] = PUSH(permutation[i+Nleft], kd);
    oswap_buffer<OSWAP_4>((unsigned char *) (permutation+i),
      (unsigned char *) (permutation+Nleft+i), 4, (*cur_switch_val)&1);
    cur_switch_val += 1;
  }
  permutation[Nleft-1] = PUSH(permutation[Nleft-1], kd);
  if (N == Neven) {
      permutation[2*Nleft-1] = PUSH(permutation[2*Nleft-1], kd);
  }
#ifdef PROFILE_SETPERM_N
  if (N >= PROFILE_SETPERM_N) {
    printf_with_rtclock_diff(prof_appsw, "end apply switches N=%u\n", N);
    printf_with_rtclock_diff(prof_before, "end before recursion N=%u\n", N);
    prof_rec1 = printf_with_rtclock("begin recursion1 N=%u\n", N);
  }
#endif
#ifdef SHOW_SETPERM
  printf(" ");
  for(uint32_t i=0;i<N;++i) {
    printf(" %2d", permutation[i]);
  }
  printf("\n ");
  for(uint32_t i=0;i<N;++i) {
    printf(" %2d", GET(permutation[i], depth+1));
  }
  printf("\n");
#endif

  traversal.inSwitches += (Nleft-1);
  uint8_t *outSwitch = traversal.outSwitches;
  traversal.outSwitches += Nright;

  // Recursively set switches of subnetworks and propagate permutation through network
  setPermutation(permutation, Nleft, depth+1, traversal, mem);
#ifdef PROFILE_SETPERM_N
  if (N >= PROFILE_SETPERM_N) {
    printf_with_rtclock_diff(prof_rec1, "end recursion1 N=%u\n", N);
    prof_rec2 = printf_with_rtclock("begin recursion2 N=%u\n", N);
  }
#endif
  setPermutation(permutation + Nleft, Nright, depth+1, traversal, mem);

#ifdef SHOW_SETPERM
  printf("R");
  for(uint32_t i=0;i<N;++i) {
    printf(" %2d", permutation[i]);
  }
  printf("\n ");
  for(uint32_t i=0;i<N;++i) {
    printf(" %2d", GET(permutation[i],depth+1));
  }
  printf("\n");
#endif
#ifdef PROFILE_SETPERM_N
  if (N >= PROFILE_SETPERM_N) {
    printf_with_rtclock_diff(prof_rec2, "end recursion2 N=%u\n", N);
    prof_outsw = printf_with_rtclock("begin output switches N=%u\n", N);
  }
#endif
  // Store output switch values and apply to permutation values
  //printf("Setting output switches\n");
  for (uint32_t i=0; i<Nright; i++) {
    outSwitch[i] = permutation[i] & 1;
    permutation[i] = POP(permutation[i], kd);
    permutation[i+Nleft] = POP(permutation[i+Nleft], kd);
    //printf("\toutSwitch[%d] = %d\n", i, outSwitch[i]);
    oswap_buffer<OSWAP_4>((unsigned char *) (permutation + i),
      (unsigned char *) (permutation + Nleft + i), 4, outSwitch[i]);
  }
  if (N != Neven) {
    permutation[Nright] = POP(permutation[Nright], kd);
  }
#ifdef PROFILE_SETPERM_N
  if (N >= PROFILE_SETPERM_N) {
    printf_with_rtclock_diff(prof_outsw, "end output switches N=%u\n", N);
    printf_with_rtclock_diff(prof_all, "end setPermutation N=%u\n", N);
  }
#endif
#ifdef SHOW_SETPERM
  printf("O");
  for(uint32_t i=0;i<Nright;++i) {
    printf(" %s", outSwitch[i] ? " X" : "||");
  }
  printf("\n");

  printf("E");
  for(uint32_t i=0;i<N;++i) {
    printf(" %2d", permutation[i]);
  }
  printf("\n ");
  for(uint32_t i=0;i<N;++i) {
    printf(" %2d", GET(permutation[i],depth));
  }
  printf("\n");
#endif

}

/*
void generateRandomPermutation(uint32_t N, uint32_t *random_permutation){
  //Initialize random permutation as 1,...,N
  for(uint32_t i=0; i<N; i++) {
    random_permutation[i]=i;
  }

  //Convert it to a random permutation of [1,N]
  RecursiveShuffle_M2((unsigned char *) random_permutation, (uint32_t) N, sizeof(uint32_t));
  // To parallelize: RecursiveShuffle_M2_parallel(buf, N, block_size, 1);
}
*/
/* 
void OblivWaksmanShuffle(unsigned char *buffer, uint32_t N, size_t block_size, enc_ret *ret) { 
  uint32_t *random_permutation;
  try {
    random_permutation = new uint32_t[N];
  } catch (std::bad_alloc&) {
    printf("Allocating memory failed in OblivWaksmanShuffle\n");
  }
  // Generate random permutation
  long t1, t2;
  ocall_clock(&t1);
  generateRandomPermutation(N, random_permutation);

  ocall_clock(&t2);
  ret->gen_perm_time = ((double)(t2-t1))/1000.0;
  #ifdef COUNT_OSWAPS
  ret->OSWAP_gp = OSWAP_COUNTER;
    OSWAP_COUNTER=0;
  #endif

  #ifdef TEST_WN_OA
    uint32_t *correct_permuted_keys = new uint32_t[N];
    printf("perm    =");
    for(size_t i=0; i<N; i++) {
        printf(" %2d", random_permutation[i]);
    }
    printf("\norig    =");
    for(size_t i=0; i<N; i++) {
      printf(" %2d", *((uint32_t*)(buffer + (block_size * i))));
    }
    printf("\ncorrect =");
    for(size_t i=0; i<N; i++) {
      uint32_t buffer_key = *((uint32_t*)(buffer + (block_size * random_permutation[i])));
      correct_permuted_keys[i] = buffer_key;
      printf(" %2d", buffer_key);
    }
    printf("\n");
  #endif  

  // Set control bits to implement randomly generated permutation
  ocall_clock(&t1);
  FOAV_SAFE_CNTXT(OWShuffle, N)
  WaksmanNetwork wnet((uint32_t) N);
  //printf("\nSetting control bits\n");
  wnet.setPermutation(random_permutation);
  ocall_clock(&t2);
  ret->control_bits_time = ((double)(t2-t1))/1000.0;
  #ifdef COUNT_OSWAPS
    ret->OSWAP_cb=OSWAP_COUNTER;
    OSWAP_COUNTER=0;
  #endif

  // Apply the permutation
  //printf("\n Applying permutation\n");
  ocall_clock(&t1);
  if (block_size == 4) {
    wnet.applyInversePermutation<OSWAP_4>(buffer, block_size);
  } else if (block_size == 8) {
    wnet.applyInversePermutation<OSWAP_8>(buffer, block_size);
  } else if (block_size == 12) {
    wnet.applyInversePermutation<OSWAP_12>(buffer, block_size);
  } else if (block_size%16 == 0) {
    wnet.applyInversePermutation<OSWAP_16X>(buffer, block_size);
  } else {
    wnet.applyInversePermutation<OSWAP_8_16X>(buffer, block_size);
  }
  ocall_clock(&t2);
  ret->apply_perm_time = ((double)(t2-t1))/1000.0;
  #ifdef COUNT_OSWAPS
    ret->OSWAP_ap = OSWAP_COUNTER;
  #endif

  #ifdef TEST_WN_OA
    printf("output  =");
    for(size_t i=0; i<N; i++) {
      printf(" %2d", *((uint32_t*)(buffer + (block_size * i))));
    }
    printf("\n");
    unsigned char *buffer_ptr = buffer;
    for(size_t i=0; i<N; i++) {
      uint32_t buffer_key = *((uint32_t*)(buffer_ptr));
      if(correct_permuted_keys[i]!=buffer_key) {
        printf("TEST_WN_OA: Shuffle Correctness Failed\n");
        break;
      }
      buffer_ptr+=block_size;
    }
    delete []correct_permuted_keys;
  #endif

  delete[] random_permutation;
} */

/* 
void DecryptAndOblivWaksmanShuffle(unsigned char *encrypted_buffer, uint32_t N,
  size_t encrypted_block_size, unsigned char *result_buffer, enc_ret *ret) {
  long t1, t2;

  // Decrypt buffer to decrypted_buffer
  unsigned char *decrypted_buffer = NULL;
  size_t decrypted_block_size = decryptBuffer(encrypted_buffer, (uint64_t) N, encrypted_block_size,
    &decrypted_buffer);
  // Set the Waksman control bits to implement the permutation
  ocall_clock(&t1);
  PRB_pool_init(1);
  OblivWaksmanShuffle(decrypted_buffer, N, decrypted_block_size, ret);
  ocall_clock(&t2);
  ret->ptime = ((double)(t2-t1))/1000.0;
  #ifdef COUNT_OSWAPS
    ret->OSWAP_count = OSWAP_COUNTER;
  #endif

  // Encrypt buffer to result_buffer
  encryptBuffer(decrypted_buffer, (uint64_t) N, decrypted_block_size, result_buffer);
  PRB_pool_shutdown();
  free(decrypted_buffer);

  return;
}
 *//* 
void OblivWaksmanSort(unsigned char *buffer, uint32_t N, size_t block_size, enc_ret *ret) { 
  uint32_t *sort_permutation;
  try {
    FOAV_SAFE_CNTXT(OWSort, N)
    sort_permutation = new uint32_t[N];
  } catch (std::bad_alloc&) {
    printf("Allocating memory failed in OblivWaksmanSort\n");
  }
  // Generate sort permutation
  long t1, t2;
  ocall_clock(&t1);
  generateSortPermutation_OA(N, buffer, block_size, sort_permutation);

  ocall_clock(&t2);
  ret->gen_perm_time = ((double)(t2-t1))/1000.0;
  #ifdef COUNT_OSWAPS
    ret->OSWAP_gp = OSWAP_COUNTER;
    OSWAP_COUNTER=0;
  #endif

  // Set control bits to implement randomly generated permutation
  ocall_clock(&t1);
#ifdef PROFILE_SETPERM_N
  unsigned long x = printf_with_rtclock("Creating network\n");
#endif
  FOAV_SAFE_CNTXT(OblivWaksmanSort, N)
  WaksmanNetwork wnet = WaksmanNetwork((uint32_t) N);
  FOAV_SAFE_CNTXT(OblivWaksmanSort, wnet)
#ifdef PROFILE_SETPERM_N
  printf_with_rtclock_diff(x, "Created network\n");
#endif
  //printf("\nSetting control bits\n");
  wnet.setPermutation(sort_permutation);
  ocall_clock(&t2);
  ret->control_bits_time = ((double)(t2-t1))/1000.0;
  #ifdef COUNT_OSWAPS
    ret->OSWAP_cb=OSWAP_COUNTER;
    OSWAP_COUNTER=0;
  #endif

  // Apply the permutation
  //printf("\nApplying permutation\n");
  ocall_clock(&t1);
  FOAV_SAFE_CNTXT(AP, block_size)
  if (block_size == 4) {
    wnet.applyInversePermutation<OSWAP_4>(buffer, block_size);
  } else if (block_size == 8) {
    wnet.applyInversePermutation<OSWAP_8>(buffer, block_size);
  } else if (block_size == 12) {
    wnet.applyInversePermutation<OSWAP_12>(buffer, block_size);
  } else if (block_size%16 == 0) {
    wnet.applyInversePermutation<OSWAP_16X>(buffer, block_size);
  } else {
    wnet.applyInversePermutation<OSWAP_8_16X>(buffer, block_size);
  }
  ocall_clock(&t2);
  ret->apply_perm_time = ((double)(t2-t1))/1000.0;
  #ifdef COUNT_OSWAPS
    ret->OSWAP_ap = OSWAP_COUNTER;
  #endif

  delete[] sort_permutation;
} */
/* 
void DecryptAndOblivWaksmanSort(unsigned char *encrypted_buffer, uint32_t N,
  size_t encrypted_block_size, unsigned char *result_buffer, enc_ret *ret) {
  long t1, t2;

  // Decrypt buffer to decrypted_buffer
  unsigned char *decrypted_buffer = NULL;
  size_t decrypted_block_size = decryptBuffer(encrypted_buffer, (uint64_t) N, encrypted_block_size,
    &decrypted_buffer);
  // Set the Waksman control bits to implement the permutation
  ocall_clock(&t1);
  PRB_pool_init(1);
  OblivWaksmanSort(decrypted_buffer, N, decrypted_block_size, ret);
  ocall_clock(&t2);
  ret->ptime = ((double)(t2-t1))/1000.0;
  #ifdef COUNT_OSWAPS
    ret->OSWAP_count = OSWAP_COUNTER;
  #endif

  // Encrypt buffer to result_buffer
  encryptBuffer(decrypted_buffer, (uint64_t) N, decrypted_block_size, result_buffer);
  PRB_pool_shutdown();
  free(decrypted_buffer);

  return;
}

void DecryptAndOWSS(unsigned char *encrypted_buffer, uint32_t N,
  size_t encrypted_block_size, unsigned char *result_buffer, enc_ret *ret) {
  long t1, t2, t3;

  // Decrypt buffer to decrypted_buffer
  unsigned char *decrypted_buffer = NULL;
  size_t decrypted_block_size = decryptBuffer(encrypted_buffer, (uint64_t) N, encrypted_block_size,
    &decrypted_buffer);
  // Set the Waksman control bits to implement the permutation
  ocall_clock(&t1);
  PRB_pool_init(1);
  OblivWaksmanShuffle(decrypted_buffer, N, decrypted_block_size, ret);
  #ifdef COUNT_OSWAPS
    ret->OSWAP_count = OSWAP_COUNTER;
  #endif

  ocall_clock(&t2);
  qsort(decrypted_buffer, N, decrypted_block_size, compare);
  ocall_clock(&t3);
  ret->qsort_time = ((double)(t3-t2))/1000.0;
  ret->ptime = ((double)(t3-t1))/1000.0;

  // Encrypt buffer to result_buffer
  encryptBuffer(decrypted_buffer, (uint64_t) N, decrypted_block_size, result_buffer);
  PRB_pool_shutdown();
  free(decrypted_buffer);

  return;
}
 */

#endif