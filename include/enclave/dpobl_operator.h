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
  struct Position {
    int bukket;
    int offset;
  };

  SMemory(int nbukkets) : nbukkets_(nbukkets) {
    positions_ = new int[nbukkets_]{0};
    clear();
  }
  ~SMemory() {
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
  inline Position locate(int index) {
    index = index % capacity_;
    return Position{index / capacity_bukket, index % capacity_bukket + 1};
  }
  inline void positionFill(Position pos) {
    positions_[pos.bukket] = pos.offset;
  }
  inline void positionDrop(Position pos) { positions_[pos.bukket] = 0; }
};

template <typename T>
class SQueue : public SMemory {
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
    for (int i = 0; i < nbukkets_; i++) {
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
    for (int i = 0; i < nbukkets_; i++) {
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
class SQueue<std::vector<T>> : public SMemory {
 private:
  T** data_;
  T* readBuf_;
  int vec_capacity_;

  int** vec_size_;
  int** index_;
  int** nid_;
  int* readBufInt_;

 public:
  SQueue(int vec_capacity, int nbukkets = 1, bool hasNid = true)
      : SMemory(nbukkets), vec_capacity_(vec_capacity) {
    capacity_bukket = cache_size / sizeof(T) - 1;
    capacity_ = nbukkets_ * capacity_bukket;

    data_ = new T*[nbukkets_];
    index_ = new int*[nbukkets_];
    for (int i = 0; i < nbukkets_; i++) {
      data_[i] = new T[(capacity_bukket + 1) * vec_capacity_];
      index_[i] = new int[capacity_bukket + 1];
    }
    readBuf_ = new T[nbukkets_];
    readBufInt_ = new int[nbukkets_];
    if (hasNid) {
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

  inline int vecSize() { return vec_capacity_; }

  inline void writeData(T& val, int vec_index = 0) {
    for (int i = 0; i < nbukkets_; i++) {
      data_[i][(capacity_bukket + 1) * vec_index + positions_[i]] = val;
    }
  }
  inline void writeInt(int** int_, int& val) {
    for (int i = 0; i < nbukkets_; i++) {
      int_[i][positions_[i]] = val;
    }
  }

  inline void push_back(std::vector<T>& val, int index, bool real) {
    CHECK_LT(size(), capacity_);

    Position pos = locate(end_);
    pos.offset = ObliviousChoose(real, pos.offset, 0);
    positionFill(pos);
    for (int i = 0; i < vec_capacity_; i++) {
      writeData(val[i], i);
    }
    writeInt(index_, index);

    positionDrop(pos);
    end_ = ObliviousChoose(real, end_ + 1, end_);
  }
  inline void push_back(std::vector<T>& val, int index, int nid,
                        bool real = true) {
    CHECK_LT(size(), capacity_);

    Position pos = locate(end_);
    pos.offset = ObliviousChoose(real, pos.offset, 0);
    positionFill(pos);
    for (int i = 0; i < vec_capacity_; i++) {
      writeData(val[i], i);
    }
    writeInt(index_, index);
    writeInt(nid_, nid);

    positionDrop(pos);
    end_ = ObliviousChoose(real, end_ + 1, end_);
  }

  inline void readData(int vec_index = 0) {
    for (int i = 0; i < nbukkets_; i++) {
      readBuf_[i] = data_[i][(capacity_bukket + 1) * vec_index + positions_[i]];
    }
  }
  inline void readInt(int** int_) {
    for (int i = 0; i < nbukkets_; i++) {
      readBufInt_[i] = int_[i][positions_[i]];
    }
  }

  inline bool pop_font(std::vector<T>& val, int& index, bool real = true) {
    real = real && (begin_ < end_);

    Position pos = locate(begin_);
    pos.offset = ObliviousChoose(real, pos.offset, 0);
    positionFill(pos);
    for (int i = 0; i < vec_capacity_; i++) {
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
  inline bool pop_font(std::vector<T>& val, int& index, int& nid,
                       bool real = true) {
    real = real && (begin_ < end_);

    Position pos = locate(begin_);
    pos.offset = ObliviousChoose(real, pos.offset, 0);
    positionFill(pos);
    for (int i = 0; i < vec_capacity_; i++) {
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



// class PageBucket : Bucket {
//  private:
//   static constexpr size_t capability_bytes{4096};
//   static constexpr size_t capability_{20};
//   xgboost::SparsePage data_;
//   std::vector<size_t> samples_index_;
//   NodeInfo<PageBucket> node_info_;

//  public:
//   PageBucket(/* args */) : node_info_(this) {
//     data_.data.Resize(capability_bytes / sizeof(xgboost::Entry));
//     samples_index_.resize(capability_, 0);
//   };
//   PageBucket(PageBucket* parent) : PageBucket() { node_info_.setParent(parent); };
//   ~PageBucket(){};
//   inline size_t size() { return data_.Size(); }
//   inline bool isFull(){
//     return size() >= capability_;
//   }
//   inline bool isEmpty(){
//     return size() <= 0;
//   }
//   // 清除内容，size重置为0
//   void clear(){
//     data_.offset.HostVector().resize(1);
//   }
//   // 将一条数据写入到bucket中
//   void pull(PageBucket* bucket, size_t index){

//   };
//   void pull(xgboost::SparsePage::Inst inst, size_t sample_index=0) {
//     // std::cout<<"pull to bucket("<<this<<")"<<std::endl;
//     if (isFull())
//     {
//       pushToChildren();
//     }
    
//     auto& data_vec = data_.data.HostVector();
//     auto& offset_vec = data_.offset.HostVector();
//     memcpy(data_vec.data() + offset_vec.back(), inst.data(),
//            inst.size() * sizeof(xgboost::Entry));
//     samples_index_[data_.Size()] = sample_index;
//     offset_vec.push_back(offset_vec.back() + inst.size());
//   }
//   // 从孩子bucket中读取数据
//   void pullFromChildren(){
//     for (size_t i = 0; i < capability_; i++)
//     {
//       node_info_.children_[i]->push(data_);
//     }
    
//   }
//   // 将bucket中的数据全部写入到孩子bucket中
//   void pushToChildren() {
//     node_info_.createChildren(capability_);
//     for (size_t i = 0; i < capability_; i++)
//     {
//       node_info_.children_[i]->pull(data_[i], samples_index_[i]);
//     }
//     clear();
//   }
//   // 将第index项写入bucket中
//   void push(PageBucket* bucket, size_t index){

//   };
//   void push(xgboost::SparsePage& page){
//     if (isEmpty())
//     {
//       pullFromChildren();
//     }
    
//     auto inst = data_[size()];
//     auto& data_vec = page.data.HostVector();
//     auto& offset_vec = page.offset.HostVector();
//     memcpy(data_vec.data() + offset_vec.back(), inst.data(), inst.size()*sizeof(xgboost::Entry));
//     offset_vec.push_back(offset_vec.back() + inst.size());
//     data_.offset.HostVector().pop_back();
//   }
// };

template<typename DataType>
class Bucket {
 private:
  static constexpr size_t capability_bytes{4096};
  size_t capability_{20};
  /*! \brief the data of the segments */
  std::vector<DataType> data;
  size_t size_{0};
 public:
  Bucket(/* args */){
    capability_ = capability_bytes/sizeof(DataType);
    data.resize(capability_);
  };
  ~Bucket(){};
  
  /* \brief clear the bucket*/
  inline void clear(){
    size_ = 0;
  }
  
  /*! \return Number of instances in the page. */
  inline size_t size() {
    return size_;
  }

  inline size_t capability(){
    return capability_;
  }
  
  inline bool isFull(){
    return size() >= capability();
  }
  inline bool isEmpty(){
    return size() <= 0;
  }

  /*! \brief get i-th row from the batch */
  inline DataType& operator[](size_t i) {
    return data[i];
  }

  inline DataType& back(){
    return data[size()-1];
  }

  inline void pop_back(){
    size_--;
  }

  inline void push(DataType& inst){
    data[size_] = inst;
    size_++;
  }
};

template<typename DataType>
class Bucket<xgboost::common::Span<DataType>> {
 private:
  static constexpr size_t capability_bytes{4096};
  size_t capability_{15};
  /*! \brief the data of the segments */
  std::vector<DataType> data;
  // Offset for each row.
  std::vector<xgboost::bst_row_t> offset;
 public:
  Bucket(/* args */){
    data.resize(capability_bytes/sizeof(DataType));
    clear();
  };
  ~Bucket(){};

  /* \brief clear the bucket*/
  inline void clear(){
    offset.resize(1, 0);
  }

  /*! \return Number of instances in the page. */
  inline size_t size() const {
    return offset.size() == 0 ? 0 : offset.size() - 1;
  }
  
  inline size_t capability(){
    return capability_;
  }
  
  inline bool isFull(){
    return size() >= capability();
  }
  inline bool isEmpty(){
    return size() <= 0;
  }

  /*! \brief get i-th row from the batch */
  inline xgboost::common::Span<DataType> operator[](size_t i) {
    size_t size = offset[i + 1] - offset[i];
    return {data.data() + offset[i], size};
  }

  inline xgboost::common::Span<DataType> back() {
    return (*this)[size()-1];
  }

  inline void pop_back(){
    offset.pop_back();
  }

  inline void push(xgboost::common::Span<DataType> inst){
    memcpy(data.data()+offset.back(), inst.data(), inst.size()*sizeof(DataType));
    offset.push_back(offset.back()+inst.size());
  }
};
class Shuffler {
 private:
  Shuffler(/* args */){};
  ~Shuffler(){};
 public:
  class Node
  {
  private:
    struct Sample
    {
      xgboost::common::Span<xgboost::Entry> inst;
      int sample_index;
    };
    
    
    Node* parent_{nullptr};
    std::vector<Node*> children_;
    size_t capability_;
    Bucket<xgboost::common::Span<xgboost::Entry>> bucket;
    std::vector<int> samples_index;
    int num_samples{0};
  public:
    Node(/* args */){
      capability_ = bucket.capability();
      samples_index.resize(capability_, -1);
    };
    ~Node(){};
    inline bool isRoot() { return parent_ == nullptr; }
    inline void setParent(Node* parent) { this->parent_ = parent; }
    inline void createChildren(size_t capability) {
      if (children_.size()==0)
      {
        children_.resize(capability);
        for (size_t i = 0; i < capability; i++)
        {
          children_[i] = new Node();
          children_[i]->setParent(this);
        }
      }
    }

    xgboost::common::Span<xgboost::Entry> pop(){
      if (bucket.isEmpty())
      {
        pullFromChildren();
      }
      auto inst = bucket.back();
      bucket.pop_back();
      return inst;
    }

    void push(xgboost::common::Span<xgboost::Entry> inst){
      if (bucket.isFull())
      {
        pushToChildren();
      }
      bucket.push(inst);
    }

    
    Sample pop(bool isSample){
      if (bucket.isEmpty())
      {
        pullFromChildren();
      }
      auto inst = bucket.back();
      bucket.pop_back();
      return {inst, samples_index[bucket.size()]};
    }

    void push(Sample sample){
      if (bucket.isFull())
      {
        pushToChildren();
      }
      bucket.push(sample.inst);
      samples_index[bucket.size() - 1] = sample.sample_index;
    }
    
    // 从孩子bucket中读取数据
    void pullFromChildren(){
      for (size_t i = 0; i < capability_; i++)
      {
        push(children_[i]->pop(true));
      }
      
    }
    // 将bucket中的数据全部写入到孩子bucket中
    void pushToChildren() {
      createChildren(capability_);
      for (size_t i = 0; i < capability_; i++)
      {
        children_[i]->push(pop(true));
      }
      // bucket.clear();
    }

    void read(xgboost::SparsePage const & in_page){
      num_samples = in_page.Size();
      const auto& data_vec = in_page.data.HostVector();
      const auto& offset_vec = in_page.offset.HostVector();

      for (size_t i = 0; i < in_page.Size(); i++)
      {
        size_t size = offset_vec[i + 1] - offset_vec[i];
        Sample sample{{const_cast<xgboost::Entry*>(data_vec.data() + offset_vec[i]), size}, (int)i};
        push(sample);
      }
      
    }

    void write(xgboost::SparsePage& out_page, std::vector<int>& out_preds){
      auto& data_vec = out_page.data.HostVector();
      auto& offset_vec = out_page.offset.HostVector();

      for (size_t i = 0; i < num_samples; i++)
      {
        auto sample = pop(true);
        memcpy(data_vec.data()+offset_vec.back(), sample.inst.data(), sample.inst.size()*sizeof(xgboost::Entry));
        offset_vec.push_back(offset_vec.back()+sample.inst.size());
        out_preds[i] = sample.sample_index;
      }
      
    }
  };

  Node root;

  static Shuffler& getInstance(){
    static Shuffler instance;
    return instance;
  }

  xgboost::SparsePage* shuffleForwardRandom(
      xgboost::SparsePage const& in_page, std::vector<int>& shuffle_index) {
    auto& in_data = in_page.data.HostVector();
    auto& in_offset = in_page.offset.HostVector();
    auto out_page = new xgboost::SparsePage;
    auto& out_data = out_page->data.HostVector();
    auto& out_offset = out_page->offset.HostVector();
    out_page->SetBaseRowId(in_page.base_rowid);
    out_data.resize(in_data.size());
    out_offset.resize(1, 0);

    // for (size_t i = 0; i < in_page.Size(); i++)
    // {
    //   // std::cout<<"read "<<i<<" from sparse page.............."<<std::endl;
    //   root.pull(in_page[i]);
    // }
    root.read(in_page);
    root.write(*out_page, shuffle_index);
    // for (size_t i = 0; i < out_preds.size(); i++)
    // {
    //   std::cout<<out_preds[i]<<" ";
    // }
    // std::cout<<std::endl;
    
    // for (size_t i = 0; i < out_page->Size(); i++)
    // {
    //   root.push(*out_page);
    // }
    
    return out_page;
  };
  // xgboost::SparsePage* shuffleForwardRandom(
  //     xgboost::SparsePage const& in_page) {
  //   auto& in_data = in_page.data.HostVector();
  //   auto& in_offset = in_page.offset.HostVector();
  //   auto out_page = new xgboost::SparsePage;
  //   auto& out_data = out_page->data.HostVector();
  //   auto& out_offset = out_page->offset.HostVector();
  //   out_page->SetBaseRowId(in_page.base_rowid);
  //   out_data.resize(in_data.size());
  //   out_offset.resize(in_offset.size());
  //   size_t top{0};
  //   out_offset[0] = 0;
  //   for (size_t i = 0; i < in_page.Size(); i++) {
  //     auto inst = in_page[i];
  //     memcpy(out_data.data() + top, inst.data(),
  //            inst.size() * sizeof(xgboost::Entry));
  //     top += inst.size();
  //     out_offset[i + 1] = top;
  //   }
  //   // std::cout<<"+++++++++++++++"<<std::endl;
  //   // memcpy(out_offset.data(), in_offset.data(),
  //   // in_offset.size()*sizeof(xgboost::bst_row_t)); memcpy(out_data.data(),
  //   // in_data.data(), in_data.size()*sizeof(xgboost::Entry));
  //   return out_page;
  // };

  template <typename Monitor>
  xgboost::SparsePage* shuffleForwardRandom(xgboost::SparsePage const& in_page, std::vector<int>& shuffle_index,
                                            Monitor* monitor_) {
    monitor_->Start(__func__);
    xgboost::SparsePage* out_page = shuffleForwardRandom(in_page, shuffle_index);
    monitor_->Stop(__func__);
    return out_page;
  };
};
