#include "psrr/shuffle.h"
#include "psrr/wakson/WaksmanNetwork/WaksmanNetwork.hpp"
#include "enclave/obl_primitives.h"
#include <iostream>

namespace obl
{
    // bool BitonicShuffler::isRegistered = registerOShufflerCreator<BitonicShuffler>();
    // bool RecursiveShuffler::isRegistered = registerOShufflerCreator<RecursiveShuffler>();

    // 在数组的每一项前面添加一个tag，tag的类型为size_t。若tag随机选取，则根据tag排序能够实现shuffle输入buf
    size_t attachTags(uint8_t *buf, size_t N, size_t block_size, TagType *tags, uint8_t *tags_buf)
    {
        size_t tag_size = sizeof(TagType);
        size_t block_size_with_tag = block_size + tag_size;

        // 如果 tags_buf 尚未分配内存，则分配所需的内存以容纳添加tag后的block
        if (buf == NULL)
        {
            buf = reinterpret_cast<uint8_t *>(malloc(N * block_size_with_tag));
            if (buf == NULL)
            {
                printf("Malloc failed in attacheTag\n");
                return -1; // 添加返回值以处理内存分配失败的情况
            }
        }

        uint8_t *tags_buf_ptr = tags_buf;
        uint8_t *buf_ptr = buf;
        uint8_t *tags_ptr = reinterpret_cast<uint8_t *>(tags);

        for (size_t i = 0; i < N; i++)
        {
            // 复制随机tag到tags_buf
            memcpy(tags_buf_ptr, tags_ptr, tag_size);
            tags_ptr += tag_size;
            tags_buf_ptr += tag_size; 

            // 复制block到tags_buf后面
            memcpy(tags_buf_ptr, buf_ptr, block_size);
            buf_ptr += block_size;
            tags_buf_ptr += block_size;
        }

        return block_size_with_tag;
    }

    // 去除attachTags函数中添加的tag，并将结果输出到buf中。
    size_t detachTags(uint8_t *tags_buf, size_t N, size_t block_size, uint8_t *buf){
        size_t tag_size = sizeof(TagType);
        size_t real_block_size = block_size - tag_size;

        uint8_t *tags_buf_ptr = tags_buf;
        uint8_t *buf_ptr = buf;

        for (size_t i = 0; i < N; i++)
        {
            // 跳过随机tag
            tags_buf_ptr += tag_size;

            // 复制block到buf
            memcpy(buf_ptr, tags_buf_ptr, real_block_size);
            buf_ptr += real_block_size;
            tags_buf_ptr += real_block_size;
        }

        return real_block_size;
    }

    size_t detachTags(uint8_t *tags_buf, size_t N, size_t block_size, uint8_t *buf, TagType *tags){
        size_t tag_size = sizeof(TagType);
        size_t real_block_size = block_size - tag_size;

        uint8_t *tags_buf_ptr = tags_buf;
        uint8_t *buf_ptr = buf;
        TagType *tags_ptr = tags;

        for (size_t i = 0; i < N; i++)
        {
            // 取出tag
            memcpy(tags_ptr, tags_buf_ptr, tag_size);
            tags_ptr += 1;
            tags_buf_ptr += tag_size;

            // 复制block到buf
            memcpy(buf_ptr, tags_buf_ptr, real_block_size);
            buf_ptr += real_block_size;
            tags_buf_ptr += real_block_size;
        }

        return real_block_size;
    }

    // std::unique_ptr<OShuffler> OShuffler::create(std::string method){
    //     auto creator_iter = getOShufflerCreatorMap().find(method);
    //     if (creator_iter == getOShufflerCreatorMap().end()) {
    //         printf("No such shuffler method: %s\n", method.c_str());
    //         return nullptr;
    //     }
    //     return creator_iter->second();
    // }

    void OShufflerWithIndex::shuffle(uint8_t *buf, size_t N, size_t block_size){
        std::cout<<"OShufflerWithIndex::shuffle" << std::endl;
        uint8_t *buffer = buf;
        // 当shuffle方法为bitonic shuffle或recursive shuffle时，需要添加索引idx表示每个block的原始位置
        // 初始化添加了idx的buf
        size_t block_size_with_idx = block_size + sizeof(TagType);
        uint8_t *idx_buf = new uint8_t[N * block_size_with_idx];

        // 初始化idx
        idx.resize(N);
        for (size_t i = 0; i < N; i++)
        {
            idx[i] = i;
        }

        // 添加idx到buf
        attachTags(buf, N, block_size, idx.data(), idx_buf);

        // 使用添加了idx的buf替代原buf进行shuffle
        buffer = idx_buf;
        block_size = block_size_with_idx;


        shuffleKernel(buffer, N, block_size);


        // 当shuffle方法为bitonic shuffle或recursive shuffle时，需要移除idx
        // 移除idx
        block_size = detachTags(buffer, N, block_size, buf, idx.data());

        // 释放idx_buf
        delete[] buffer;
    }

    void OShufflerWithIndex::inverseShuffle(uint8_t *buf, size_t block_size){
        // 需要空间临时存放结果，否则shuffle结果会覆盖原buf
        uint8_t *temp_buf = new uint8_t[idx.size() * block_size];

        // 根据idx将shuffle结果恢复顺序
        for (size_t i = 0; i < idx.size(); i++)
        {
            ObliviousArrayAssignBytes(temp_buf, buf + i * block_size, block_size, idx[i], idx.size());
        }

        // 将temp_buf中的内容复制到输出buf中
        memcpy(buf, temp_buf, idx.size() * block_size);
    }

    void OShufflerWithIndex::inverseShuffle(uint8_t *buf, size_t block_size, uint8_t *out_buf, size_t offset){
        // 需要空间临时存放结果，否则shuffle结果会覆盖原buf
        uint8_t *temp_buf = new uint8_t[idx.size() * block_size];

        // 根据idx将shuffle结果恢复顺序
        for (size_t i = 0; i < idx.size(); i++)
        {
            ObliviousArrayAssignBytes(temp_buf, buf + i * block_size, block_size, idx[i], idx.size());
        }

        // 将temp_buf中的内容复制到输出buf中
        memcpy(out_buf, temp_buf + offset * block_size, (idx.size() - offset) * block_size);
    }

    void OShufflerUsingSorter::shuffleKernel(uint8_t *buf, size_t N, size_t block_size){
        // 添加随机tag
        TagType *tags = new TagType[N];
        PRB_buffer::getInstance().getBulkRandomBytes(reinterpret_cast<uint8_t *>(tags), N * sizeof(TagType));
        uint8_t *tags_buf = new uint8_t[N * (block_size + sizeof(TagType))];
        block_size = attachTags(buf, N, block_size, tags, tags_buf);

        // 以tag为key，block为单位，进行bitonic排序
        sort(tags_buf, N, block_size, true);

        // 移除tag
        block_size = detachTags(tags_buf, N, block_size, buf);

        delete[] tags;
        delete[] tags_buf;
    }

    void BitonicShuffler::sort(uint8_t *buf, size_t N, size_t block_size, bool ascending){
        // 以tag为key，block为单位，进行bitonic排序
        BitonicSort<TagType>(buf, N, block_size, true);

    }

    void BubbleShuffler::sort(uint8_t *buf, size_t N, size_t block_size, bool ascending){
        BubbleSort<TagType>(buf, N, block_size, true);
    }

    void RecursiveShuffler::shuffleKernel(uint8_t *buf, size_t N, size_t block_size){
        RecursiveShuffle_M1(buf, N, block_size);
    }

    #ifdef ENABLE_WAKSMAN_SHUFFLE

    bool WaksmanShuffler::isRegistered = registerOShufflerCreator<WaksmanShuffler>();

    void WaksmanShuffler::shuffle(uint8_t *buf, size_t N, size_t block_size){
        uint32_t *random_permutation;
        try {
            random_permutation = new uint32_t[N];
        } catch (std::bad_alloc&) {
            printf("Allocating memory failed in OblivWaksmanShuffle\n");
        }
        // Generate random permutation
        generateRandomPermutation(N, random_permutation);

        // Set control bits to implement randomly generated permutation
        wnet = std::make_unique<WaksmanNetwork>(((uint32_t) N));
        //printf("\nSetting control bits\n");
        wnet->setPermutation(random_permutation);

        // Apply the permutation
        //printf("\n Applying permutation\n");
        if (block_size == 1) {
            wnet->applyPermutation<OSWAP_1>(buf, block_size);
        } else if (block_size == 2) {
            wnet->applyPermutation<OSWAP_2>(buf, block_size);
        } else if (block_size == 4) {
            wnet->applyPermutation<OSWAP_4>(buf, block_size);
        } else if (block_size == 8) {
            wnet->applyPermutation<OSWAP_8>(buf, block_size);
        } else if (block_size == 12) {
            wnet->applyPermutation<OSWAP_12>(buf, block_size);
        } else if (block_size%16 == 0) {
            wnet->applyPermutation<OSWAP_16X>(buf, block_size);
        } else if (block_size%8 == 0) {
            wnet->applyPermutation<OSWAP_8_16X>(buf, block_size);
        } else {
            wnet->applyPermutation<OSWAP_ANY>(buf, block_size);
        }

        delete[] random_permutation;
    }

    void WaksmanShuffler::inverseShuffle(uint8_t *buf, size_t block_size){
        // 调用WaksmanNetwork的inversePermutation函数
        if (block_size == 1) {
            wnet->applyInversePermutation<OSWAP_1>(buf, block_size);
        } else if (block_size == 2) {
            wnet->applyInversePermutation<OSWAP_2>(buf, block_size);
        } else if (block_size == 4) {
            wnet->applyInversePermutation<OSWAP_4>(buf, block_size);
        } else if (block_size == 8) {
            wnet->applyInversePermutation<OSWAP_8>(buf, block_size);
        } else if (block_size == 12) {
            wnet->applyInversePermutation<OSWAP_12>(buf, block_size);
        } else if (block_size%16 == 0) {
            wnet->applyInversePermutation<OSWAP_16X>(buf, block_size);
        } else if (block_size%8 == 0) {
            wnet->applyInversePermutation<OSWAP_8_16X>(buf, block_size);
        } else {
            wnet->applyInversePermutation<OSWAP_ANY>(buf, block_size);
        }
    }

    void WaksmanShuffler::inverseShuffle(uint8_t *buf, size_t block_size, uint8_t *out_buf, size_t offset){
        // 调用WaksmanNetwork的inversePermutation函数
        inverseShuffle(buf, block_size);

        // 将shuffle结果复制到输出buf中
        memcpy(out_buf, buf + offset * block_size, (wnet->numItems() - offset) * block_size);
    }
    #endif // ENABLE_WAKSON_SHUFFLE

    std::unique_ptr<OShuffler> getShuffler(std::string method){
        if (method == "BitonicShuffler") 
        {
            return std::unique_ptr<OShuffler>(new BitonicShuffler());
        } else if (method == "BubbleShuffler") 
        {
            return std::unique_ptr<OShuffler>(new BubbleShuffler());
        } else if (method == "RecursiveShuffler") 
        {
            return std::unique_ptr<OShuffler>(new RecursiveShuffler());
        } else 
        
    #ifdef ENABLE_WAKSMAN_SHUFFLE
        if (method == "WaksmanShuffler") 
        {
            return std::unique_ptr<OShuffler>(new WaksmanShuffler());
        } else
    #endif // ENABLE_WAKSMAN_SHUFFLE
        {
            std::cout << "No such shuffler method: " << method << std::endl;
            return nullptr;
        }
        
    }
}