#pragma once

#include <cstddef>
#include <cstdint>
#include <cstring>
#include <type_traits>

#include "enclave/obl_primitives.h"
#include "xgboost/base.h"
#include "xgboost/data.h"

constexpr int cache_size = 4 * 1024;

/**
 * private memory interface
 * consist of multiple bukkets, one access one bukket
 */
class SMemory {
 private:
 protected:
  int nbukkets_;
  int capacity_bukket;
  int capacity_;
  int begin_;
  int end_;
  int* positions_;
 public:
  struct Position
  {
    int bukket;
    int offset;
  };
  
  SMemory(int nbukkets): nbukkets_(nbukkets) { 
    positions_ = new int[nbukkets_]{0}; 
    clear();
  }
  ~SMemory(){
    delete[] positions_;
    positions_ = nullptr;
  }
  inline int capacity() { return capacity_; }
  inline int size() { return end_ - begin_; }
  inline bool full() { return size() == capacity_; }
  inline bool empty() { return begin_ == end_; }
  inline void clear() {
    begin_ = 0;
    end_ = 0;
  }
  inline Position locate(int index){
    index = index % capacity_;
    return Position{index / capacity_bukket, index % capacity_bukket + 1};
  }
  inline void positionFill(Position pos){
    positions_[pos.bukket] = pos.offset;
  }
  inline void positionDrop(Position pos){
    positions_[pos.bukket] = 0;
  }
};

template <typename T>
class SQueue: public SMemory {
 private:
  T** data_;
  T* readBuf_;

 public:
  SQueue(int nbukkets = 1) : SMemory(nbukkets) {
    capacity_bukket = cache_size / sizeof(T) - 1;
    capacity_ = nbukkets_ * capacity_bukket;

    data_ = new T*[nbukkets_];
    for (int i = 0; i < nbukkets_; i++) {
      data_[i] = new T[capacity_bukket + 1];
    }
    readBuf_ = new T[nbukkets_];
  };
  ~SQueue() {
    for (int i = 0; i < nbukkets_; i++) {
      delete[] data_[i];
    }
    delete[] data_;
    data_ = nullptr;
    delete[] readBuf_;
    readBuf_ = nullptr;
  };

  inline void push_back(T& val, bool real) {
    CHECK_LT(size(), capacity_);

    // int index = end_ % capacity_;
    // int pid = index / (capacity_ / nbukkets_);
    // int vid = index % (capacity_ / nbukkets_) + 1;
    // int ids[nbukkets_] = {0};

    // ids[pid] = ObliviousChoose(real, vid, 0);

    // for (int page = 0; page < nbukkets_; page++) {
    //   data_[page][ids[page]] = val;
    // }

    Position pos = locate(end_);
    pos.offset = ObliviousChoose(real, pos.offset, 0);
    positionFill(pos);
    for(int i=0;i<nbukkets_;i++){
      data_[i][positions_[i]] = val;
    }
    positionDrop(pos);
    end_ = ObliviousChoose(real, end_ + 1, end_);
  }

  inline bool pop_font(T& val, bool real = true) {
    real = real && (begin_ < end_);

    // int index = begin_ % capacity_;
    // int pid = index / (capacity_ / nbukkets_);
    // int vid = index % (capacity_ / nbukkets_) + 1;
    // int ids[nbukkets_] = {0};
    // ids[pid] = ObliviousChoose(real, vid, 0);

    // // for (int page=0; page<nbukkets_; page++){
    // //     ObliviousAssign(page==pid, data_[page][ids[page]], val, &val);
    // // }

    // // val = data_[pid][ids[pid]];
    // // 使用下述代码时间延长比较明显
    // T temp[nbukkets_];
    // for (size_t i = 0; i < nbukkets_; i++) {
    //   temp[i] = data_[i][ids[i]];
    // }
    // val = temp[pid];
    
    Position pos = locate(begin_);
    pos.offset = ObliviousChoose(real, pos.offset, 0);
    positionFill(pos);
    for(int i=0;i<nbukkets_;i++){
      readBuf_[i] = data_[i][positions_[i]];
    }
    val = readBuf_[pos.bukket];
    positionDrop(pos);

    begin_ = ObliviousChoose(real, begin_ + 1, begin_);

    if (begin_ > capacity_) {
      begin_ -= capacity_;
      end_ -= capacity_;
    }
    return real;
  }
};

constexpr float epsilon = 1;
constexpr float delta = 0.0001;

inline int DPPrefixSum(int prefix_sum, int error) {
  int min = -error + 1;
  int max = error - 1;
  int noise = (rand() % (max - min + 1)) + min;  // 范围[min,max]
  // return prefix_sum + noise;
  return prefix_sum + noise;
}