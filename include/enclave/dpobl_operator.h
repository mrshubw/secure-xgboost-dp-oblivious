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

template <typename T>
class SQueue<std::vector<T>>: public SMemory {
 private:
  T** data_;
  T* readBuf_;
  int vec_capacity_;

  int** vec_size_;
  int** index_;
  int** nid_;
  int* readBufInt_;

 public:
  SQueue(int vec_capacity, int nbukkets = 1, bool hasNid=true) : SMemory(nbukkets), vec_capacity_(vec_capacity) {
    capacity_bukket = cache_size / sizeof(T) - 1;
    capacity_ = nbukkets_ * capacity_bukket;

    data_ = new T*[nbukkets_];
    index_ = new int*[nbukkets_];
    for (int i = 0; i < nbukkets_; i++) {
      data_[i] = new T[(capacity_bukket + 1)*vec_capacity_];
      index_[i] = new int[capacity_bukket + 1];
    }
    readBuf_ = new T[nbukkets_];
    readBufInt_ = new int[nbukkets_];
    if (hasNid)
    {
      nid_ = new int*[nbukkets_];
      for (int i = 0; i < nbukkets_; i++) {
        nid_[i] = new int[capacity_bukket + 1];
      }
    }
    
  };
  ~SQueue() {
    for (int i = 0; i < nbukkets_; i++) {
      delete[] data_[i];
      delete[] index_[i];
    }
    delete[] data_;
    data_ = nullptr;
    delete[] index_;
    index_ = nullptr;
    delete[] readBuf_;
    readBuf_ = nullptr;
    delete[] readBufInt_;
    readBufInt_ = nullptr;
  };

  inline int vecSize(){
    return vec_capacity_;
  }

  inline void writeData(T& val, int vec_index=0){
    for (int i = 0; i < nbukkets_; i++)
    {
      data_[i][(capacity_bukket + 1)*vec_index+positions_[i]] = val;
    }
  }
  inline void writeInt(int** int_, int& val){
    for (int i = 0; i < nbukkets_; i++)
    {
      int_[i][positions_[i]] = val;
    }
  }

  inline void push_back(std::vector<T>& val, int index, bool real) {
    CHECK_LT(size(), capacity_);

    Position pos = locate(end_);
    pos.offset = ObliviousChoose(real, pos.offset, 0);
    positionFill(pos);
    for (int i = 0; i < vec_capacity_; i++)
    {
      writeData(val[i], i);
    }
    writeInt(index_, index);
    
    positionDrop(pos);
    end_ = ObliviousChoose(real, end_ + 1, end_);
  }
  inline void push_back(std::vector<T>& val, int index, int nid, bool real=true) {
    CHECK_LT(size(), capacity_);

    Position pos = locate(end_);
    pos.offset = ObliviousChoose(real, pos.offset, 0);
    positionFill(pos);
    for (int i = 0; i < vec_capacity_; i++)
    {
      writeData(val[i], i);
    }
    writeInt(index_, index);
    writeInt(nid_, nid);
    
    positionDrop(pos);
    end_ = ObliviousChoose(real, end_ + 1, end_);
  }

  inline void readData(int vec_index=0){
    for (int i = 0; i < nbukkets_; i++)
    {
      readBuf_[i] = data_[i][(capacity_bukket + 1)*vec_index+positions_[i]];
    }
  }
  inline void readInt(int** int_){
    for (int i = 0; i < nbukkets_; i++)
    {
      readBufInt_[i] = int_[i][positions_[i]];
    }
  }

  inline bool pop_font(std::vector<T>& val, int& index, bool real = true) {
    real = real && (begin_ < end_);
    
    Position pos = locate(begin_);
    pos.offset = ObliviousChoose(real, pos.offset, 0);
    positionFill(pos);
    for (int i = 0; i < vec_capacity_; i++)
    {
      readData(i);
      val[i] = readBuf_[pos.bukket];
    }
    readInt(index_);
    index = readBufInt_[pos.bukket];
    
    positionDrop(pos);

    begin_ = ObliviousChoose(real, begin_ + 1, begin_);

    if (begin_ > capacity_) {
      begin_ -= capacity_;
      end_ -= capacity_;
    }
    return real;
  }
  inline bool pop_font(std::vector<T>& val, int& index, int& nid, bool real = true) {
    real = real && (begin_ < end_);
    
    Position pos = locate(begin_);
    pos.offset = ObliviousChoose(real, pos.offset, 0);
    positionFill(pos);
    for (int i = 0; i < vec_capacity_; i++)
    {
      readData(i);
      val[i] = readBuf_[pos.bukket];
    }
    readInt(index_);
    index = readBufInt_[pos.bukket];
    readInt(nid_);
    nid = readBufInt_[pos.bukket];
    
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