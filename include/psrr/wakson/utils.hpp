#pragma once
#include <cstdint>
#include <random>
#include <vector>
#include <cstring>
#include <algorithm>
#include <functional>
#include <iostream>

class PRB_buffer {
public:
    static constexpr size_t PRB_BUFFER_SIZE = 100000;

    PRB_buffer() : random_bytes_left(0), PRB_rand_bits_remaining(0) {
      std::random_device rd;
      random_seed = std::mt19937(rd());
        
      std::generate(random_bytes, random_bytes + PRB_BUFFER_SIZE, std::ref(random_seed));
      random_bytes_left = PRB_BUFFER_SIZE;
      random_bytes_ptr = random_bytes;
    }

    // 获取单例实例
    static PRB_buffer& getInstance() {
        static PRB_buffer instance; // 懒汉式单例
        return instance;
    }

    void getRandomBytes(unsigned char *buffer, size_t size);
    void getBulkRandomBytes(unsigned char *buffer, size_t size);
    bool getRandomBit();

private:
    std::mt19937 random_seed; // C++ 随机数生成器
    size_t random_bytes_left;
    unsigned char random_bytes[PRB_BUFFER_SIZE];
    unsigned char* random_bytes_ptr;
    
    // Return a random bit
    uint64_t PRB_rand_bits;
    uint32_t PRB_rand_bits_remaining;
};

template <typename t>
void print_array(t array, size_t N) {
    for(size_t i = 0; i < N; i++)
        std::cout << array[i] << " ";
    std::cout << std::endl;
}


// Other utility functions
int calculatelog2(uint64_t value);
int calculatelog2_floor(uint64_t value);
uint64_t pow2_lt(uint64_t N);
uint64_t pow2_gt(uint64_t N);
