#include "psrr/wakson/utils.hpp"

void PRB_buffer::getRandomBytes(unsigned char *buffer, size_t size) {
    if (size < random_bytes_left) {
        // 从缓冲区中提供随机字节
        std::memcpy(buffer, random_bytes_ptr, size);
        random_bytes_ptr += size;
        random_bytes_left -= size;
    } else {
        // 消耗所有剩余的随机字节
        std::memcpy(buffer, random_bytes_ptr, random_bytes_left);
        size_t size_left_for_req = size - random_bytes_left;

        // 生成新的随机字节
        std::generate(random_bytes, random_bytes + PRB_BUFFER_SIZE, std::ref(random_seed));
        random_bytes_left = PRB_BUFFER_SIZE;
        random_bytes_ptr = random_bytes;

        // 将剩余的随机字节拷贝到缓冲区
        std::memcpy(buffer + random_bytes_left, random_bytes_ptr, size_left_for_req);
        random_bytes_ptr += size_left_for_req;
        random_bytes_left -= size_left_for_req;
    }
}

void PRB_buffer::getBulkRandomBytes(unsigned char *buffer, size_t size) {
    // 直接生成随机字节填充到buffer
    std::generate(buffer, buffer + size, std::ref(random_seed));
}

bool PRB_buffer::getRandomBit() {
    if (PRB_rand_bits_remaining == 0) {
        getRandomBytes((unsigned char *)&PRB_rand_bits,
            sizeof(PRB_rand_bits));
        PRB_rand_bits_remaining = 64;
    }
    bool ret = PRB_rand_bits & 1;
    PRB_rand_bits >>= 1;
    PRB_rand_bits_remaining -= 1;
    return ret;
}

// Returns log2 rounded up.
int calculatelog2(uint64_t value){
  int log2v = 0;
  uint64_t temp = 1;
  while(temp<value){
    temp=temp<<1;
    log2v+=1;
  }
  return log2v;
}

int calculatelog2_floor(uint64_t value){
  int log2v = 0;
  uint64_t temp = 1;
  while(temp<value){
    temp=temp<<1;
    log2v+=1;
  }
  if(temp==value)
    return log2v;
  else
    return log2v-1;
}

// Returns largest power of two less than N
uint64_t pow2_lt(uint64_t N) {
  uint64_t N1 = 1;
  while (N1 < N) {
    N1 <<= 1;
  }
  N1 >>= 1;
  return N1;
}


// Returns largest power of two greater than N
uint64_t pow2_gt(uint64_t N) {
  uint64_t N1 = 1;
  while (N1 < N) {
    N1 <<= 1;
  }
  return N1;
}
