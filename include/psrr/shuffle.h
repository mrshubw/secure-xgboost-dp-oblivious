#pragma once
#include <vector>
#include <string>
#include <iostream>
#include <memory> 
#include <unordered_map>
#include <functional>
#include "wakson/RecursiveShuffle/RecursiveShuffle.hpp"
#include "wakson/WaksmanNetwork/WaksmanNetwork.hpp"
#include "sorter.h"

namespace obl
{
    using TagType = uint32_t;

    // enum class Method {BitonicShuffle, RecursiveShuffle, WaksmanShuffle};
    class OShuffler
    {
    public:
        virtual ~OShuffler() = default; // 虚析构函数
        virtual void shuffle(uint8_t *buf, size_t N, size_t block_size) = 0;
        // 逆shuffle,用于恢复原始顺序。BitonicShuffle和RecursiveShuffle的逆shuffle基于obliviousAssign函数，WaksmanShuffle的逆shuffle基于Waksman自带的inversePermutation函数。
        virtual void inverseShuffle(uint8_t *buf, size_t block_size) = 0;
        virtual void inverseShuffle(uint8_t *buf, size_t block_size, uint8_t *out_buf, size_t offset=0) = 0;
    };

    class OShufflerWithIndex : public OShuffler
    {
    public:
        void shuffle(uint8_t *buf, size_t N, size_t block_size);
        void inverseShuffle(uint8_t *buf, size_t block_size);
        void inverseShuffle(uint8_t *buf, size_t block_size, uint8_t *out_buf, size_t offset=0);

    private:
        std::vector<TagType> idx; // 用于记录每一项在输入buf中的位置,BitonicShuffle和RecursiveShuffle需要

        virtual void shuffleKernel(uint8_t *buf, size_t N, size_t block_size) = 0;
    };
    
    using OShufflerCreatorFunc_t = std::function<std::unique_ptr<OShuffler>()>;
    using OShufflerCreatorMap_t = std::unordered_map<std::string, OShufflerCreatorFunc_t>;
    static OShufflerCreatorMap_t& getOShufflerCreatorMap() {
        static OShufflerCreatorMap_t creatorMap;
        return creatorMap;
    }
    template<typename T>
    bool registerOShufflerCreator() {
        std::cout<<"registerOShufflerCreator" << std::endl;
        
        getOShufflerCreatorMap()[T::ClassName()] = []() { return std::unique_ptr<OShuffler>(new T()); };
        return true;
    }

    static std::unique_ptr<OShuffler> create(std::string method){
        auto creator_iter = getOShufflerCreatorMap().find(method);
        if (creator_iter == getOShufflerCreatorMap().end()) {
            printf("No such shuffler method: %s\n", method.c_str());
            return nullptr;
        }
        return creator_iter->second();
    }

    class BitonicShuffler : public OShufflerWithIndex
    {
    public:
        static std::string ClassName() {
            return "BitonicShuffler";
        }
        // static bool isRegistered;
    private:

        void shuffleKernel(uint8_t *buf, size_t N, size_t block_size) override;
    };

    class RecursiveShuffler : public OShufflerWithIndex
    {
    public:
        static std::string ClassName() {
            return "RecursiveShuffler";
        }
        // static bool isRegistered;
    private:

        void shuffleKernel(uint8_t *buf, size_t N, size_t block_size) override;
    };

    #ifdef ENABLE_WAKSMAN_SHUFFLE

    class WaksmanShuffler : public OShuffler
    {
    public:
        static std::string ClassName() {
            return "WaksmanShuffler";
        }
        void shuffle(uint8_t *buf, size_t N, size_t block_size) override;
        void inverseShuffle(uint8_t *buf, size_t block_size) override;
        void inverseShuffle(uint8_t *buf, size_t block_size, uint8_t *out_buf, size_t offset=0) override;

    private:
        static bool isRegistered;

        std::unique_ptr<WaksmanNetwork> wnet; // WaksmanShuffle需要的网络结构
    };
    #endif // ENABLE_WAKSMAN_SHUFFLE

    std::unique_ptr<OShuffler> getShuffler(std::string method);
}