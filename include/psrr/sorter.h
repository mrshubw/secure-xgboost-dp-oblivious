#pragma once
#include <vector>
#include <string>
#include <iostream>
#include "wakson/SortingNetwork/SortingNetwork.hpp"

namespace obl {

    class OSorter {
    public:
        std::string method = "bitonic"; // 默认使用 bitonic 排序

        OSorter(std::string method = "bitonic") : method(method) {};
        ~OSorter() = default;

        template<typename KeyType>
        void sort(std::vector<KeyType>& array){
            bitonicSort(array);
        };

        // wakson库提供的BitonicSort接口有误，没有正确地传入KeyType，这里重新实现一个接口
        template<typename KeyType>
        void bitonicSort(std::vector<KeyType>& array){
            BitonicSort(array.data(),  array.size(), true);
        };
    };

    
    // 实现一个BubbleSort接口，buffer是一个由N个长为block_size的块组成的数组，N是数组的长度，block_size是块具有的字节的大小，ascend表示升序还是降序，将每个块的前sizeof(KeyType)个char作为一个Key，进行排序
    template<typename KeyType>
    void BubbleSort(unsigned char *buffer, size_t N, size_t block_size, bool ascend){
        for (size_t i = 0; i < N - 1; ++i) {
            for (size_t j = 0; j < N - i - 1; ++j) {
                // 获取当前块和下一块的 Key
                KeyType key1 = *reinterpret_cast<KeyType*>(buffer + j * block_size);
                KeyType key2 = *reinterpret_cast<KeyType*>(buffer + (j + 1) * block_size);
    
                // 比较并交换块
                // if ((ascend && key1 > key2) || (!ascend && key1 < key2)) {
                //     // 交换整个块
                //     for (size_t k = 0; k < block_size; ++k) {
                //         std::swap(buffer[j * block_size + k], buffer[(j + 1) * block_size + k]);
                //     }
                // }
                oswap_buffer<OSWAP_ANY>(buffer + j * block_size, buffer + (j + 1) * block_size, block_size, (ascend && key1 > key2) || (!ascend && key1 < key2));
            }
        }
    }
}
