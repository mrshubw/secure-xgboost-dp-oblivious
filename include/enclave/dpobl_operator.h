#pragma once

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <limits>
#include <type_traits>
#include <iostream>
#include <fstream>
#include <random>
#include <vector>
#include <sstream>

#include "enclave/obl_primitives.h"
#include "xgboost/base.h"
#include "xgboost/data.h"
#include "psrr/shuffle.h"

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


inline void logStr(std::string logfile, std::string outStr){

  // Create an ofstream (output file stream) object
  std::fstream outfile;

  // Open the file in write mode
  outfile.open(logfile, std::ios::out|std::ios::app);
  // Check if the file was opened successfully
  if (!outfile) {
      std::cerr << "Error opening file" << std::endl;
      return;
  }

  outfile<<outStr;
  
  // Close the file
  outfile.close();
}

template <typename VauleType>
inline void logStr(std::string logfile, std::string outStr, VauleType value){

  // Create an ofstream (output file stream) object
  std::fstream outfile;

  // Open the file in write mode
  outfile.open(logfile, std::ios::out|std::ios::app);
  // Check if the file was opened successfully
  if (!outfile) {
      std::cerr << "Error opening file" << std::endl;
      return;
  }

  outfile<<outStr<<value<<std::endl;
  
  // Close the file
  outfile.close();
}



// 从配置文件中读取指定参数的值，支持不同类型
template <typename T>
T ReadParameterFromConfig(const std::string &filename, const std::string &key, const T& default_value = T()) {
    std::ifstream file(filename);
    std::string line;
    T value = default_value; // 默认值

    if (file.is_open()) {
        while (getline(file, line)) {
            if (line.find(key + "=") == 0) { // 检查每一行是否以key开头
                std::string value_str = line.substr(line.find('=') + 1); // 提取等号后的值
                
                if constexpr (std::is_same<T, std::string>::value) {
                    value = value_str; // 字符串类型直接赋值
                } else {
                    std::istringstream iss(value_str);
                    iss >> value; // 将字符串转换为目标类型
                }
                break; // 找到后可以退出
            }
        }
        file.close();
    } else {
        std::cerr << "Cannot open file: " << filename << std::endl;
    }

    return value; // 返回读取的值或默认值
}

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
      // for (size_t i = 0; i < capability_; i++)
      // {
      //   push(children_[i]->pop(true));
      // }
      
      std::vector<size_t> indices(capability_);
      for (size_t i = 0; i < capability_; i++) {
        indices[i] = i;
      }
      std::random_device rd;
      std::mt19937 g(rd());
      std::shuffle(indices.begin(), indices.end(), g);
      for (size_t i = 0; i < capability_; i++)
      {
        push(children_[indices[i]]->pop(true));
      }
    }
    // 将bucket中的数据全部写入到孩子bucket中
    void pushToChildren() {
      createChildren(capability_);

      // for (size_t i = 0; i < capability_; i++)
      // {
      //   children_[i]->push(pop(true));
      // }
      // // bucket.clear();

      std::vector<size_t> indices(capability_);
      for (size_t i = 0; i < capability_; i++) {
        indices[i] = i;
      }
      std::random_device rd;
      std::mt19937 g(rd());
      std::shuffle(indices.begin(), indices.end(), g);
      for (size_t i = 0; i < capability_; i++) {
        children_[indices[i]]->push(pop(true));
      }
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

  void shuffleForwardRandom(xgboost::SparsePage const& in_page, xgboost::SparsePage& out_page, std::vector<int>& shuffle_index) {
    auto& in_data = in_page.data.HostVector();
    auto& in_offset = in_page.offset.HostVector();
    auto& out_data = out_page.data.HostVector();
    auto& out_offset = out_page.offset.HostVector();
    out_page.SetBaseRowId(in_page.base_rowid);
    out_data.resize(in_data.size());
    out_offset.resize(1, 0);

    // std::cout<<"shuffle point"<<std::endl;

    root.read(in_page);
    // std::cout<<"shuffle point"<<std::endl;
    root.write(out_page, shuffle_index);
    // std::cout<<"shuffle point"<<std::endl;
    // for (size_t i = 0; i < shuffle_index.size(); i++)
    // {
    //   std::cout<<"index: "<<shuffle_index[i]<<std::endl;
    // }
    
  };

  // template <typename Monitor>
  // xgboost::SparsePage* shuffleForwardRandom(xgboost::SparsePage const& in_page, std::vector<int>& shuffle_index,
  //                                           Monitor* monitor_) {
  //   monitor_->Start(__func__);
  //   xgboost::SparsePage* out_page = shuffleForwardRandom(in_page, shuffle_index);
  //   monitor_->Stop(__func__);
  //   return out_page;
  // };
};


class DummySamples: public xgboost::SparsePage
{
private:
  int num_samples;
  static std::map<int, std::shared_ptr<DummySamples>> instances_;
public:
  DummySamples(int num_samples, const xgboost::SparsePage::Inst& inst){
    for (size_t i = 0; i < num_samples; i++)
    {
      this->Push(inst);
    }
    
  }
  ~DummySamples(){}

  static std::shared_ptr<DummySamples> getInstance(int num_samples, const xgboost::SparsePage::Inst& inst){
    auto it = instances_.find(num_samples);
    if (it == instances_.end()) {
        std::shared_ptr<DummySamples> instance(new DummySamples(num_samples, inst));
        instances_[num_samples] = instance;
        return instance;
    }
    return it->second;
  }
  
};

// Oblivious Swap
template <typename T>
void obliviousSwap(T& value1, T& value2, bool pred) {
    T temp = value1;
    value1 = ObliviousChoose(pred, value2, value1);
    value2 = ObliviousChoose(pred, temp, value2);
}


template <typename T, typename Ty=T>
class BitonicSorter {
public:
    BitonicSorter(std::vector<T>& data, bool ascending = true, bool oblivious = true)
        : arr(data), ascending(ascending), oblivious(oblivious) {
        originalSize_ = arr.size();
        paddedSize_ = nextPowerOf2(originalSize_);
    }

    void sort() {
        arr.resize(paddedSize_, std::numeric_limits<T>::max());
        #pragma omp parallel
        {
            #pragma omp single
            {
                bitonicSort(0, arr.size(), ascending);
            }
        }
        arr.resize(originalSize_);
    }

    void reorder(std::vector<Ty>& data){
        reorder_data = &data;
        data.resize(paddedSize_, std::numeric_limits<Ty>::max());
        sort();
    }

    double sortAndMeasureTime() {
        auto start = std::chrono::high_resolution_clock::now();
        sort();
        auto end = std::chrono::high_resolution_clock::now();
        std::chrono::duration<double> elapsed = end - start;
        return elapsed.count();
    }

private:
    std::vector<T>& arr;
    bool ascending;
    bool oblivious;
    int paddedSize_;
    int originalSize_;
    std::vector<Ty>* reorder_data = nullptr;

    void compareAndSwap(int i, int j, bool dir) {
      bool pred = dir == (arr[i] > arr[j]);
      if (oblivious)
      {
        obliviousSwap(arr[i], arr[j], pred);
        if (reorder_data!=nullptr)
        {
            obliviousSwap((*reorder_data)[i], (*reorder_data)[j], pred);
        }
      }else
      {
        if (pred) {
            std::swap(arr[i], arr[j]);
            if (reorder_data!=nullptr)
            {
                std::swap((*reorder_data)[i], (*reorder_data)[j]);
            }
        }
      }
    }

    void bitonicMerge(int low, int cnt, bool dir) {
        if (cnt > 1) {
            int k = cnt / 2;
            #pragma omp parallel for schedule(dynamic)
            for (int i = low; i < low + k; i++) {
                compareAndSwap(i, i + k, dir);
            }
            #pragma omp task
            bitonicMerge(low, k, dir);
            #pragma omp task
            bitonicMerge(low + k, k, dir);
            #pragma omp taskwait
        }
    }

    void bitonicSort(int low, int cnt, bool dir) {
        if (cnt > 1) {
            int k = cnt / 2;
            #pragma omp task
            bitonicSort(low, k, true);
            #pragma omp task
            bitonicSort(low + k, k, false);
            #pragma omp taskwait
            bitonicMerge(low, cnt, dir);
        }
    }

    int nextPowerOf2(int n) {
        int p = 1;
        while (p < n) {
            p <<= 1;
        }
        return p;
    }
};

// Set arr[i] += val
template <typename T>
inline void obliviousArrayElementAdd(T *arr, size_t i, size_t n, const T &val){
  size_t nbytes = sizeof(T);
  size_t step = nbytes < CACHE_LINE_SIZE ? CACHE_LINE_SIZE / nbytes : 1;
  T temp;
  i = ObliviousChoose(i<n, i, i+step);
  for (size_t j = 0; j < n; j += step) {
    bool cond = ObliviousEqual(j / step, i / step);
    int pos = ObliviousChoose(cond, i, j);
    void *dst_pos = (char *)(arr) + pos * nbytes;
    temp = arr[pos] + val;
    obl::ObliviousBytesAssign(cond, nbytes, &temp, dst_pos, dst_pos);
  }
}



// 计算标准正态分布的分位数（近似算法）
inline double inverseCDF(double p) {
  // 使用近似公式
  if (p < 0.5) {
      return -std::sqrt(-2.0 * std::log(p));
  } else {
      return std::sqrt(-2.0 * std::log(1.0 - p));
  }
}

inline double calculateSigma(double epsilon, double delta, double sensitivity) {
  double c = std::sqrt(2 * std::log(1.25 / delta));
  return c * sensitivity / epsilon;
}

inline double calculateMean(double sigma, double delta) {
  double z = inverseCDF(delta);
  return -sigma * z;
}

struct PrivacyBudget {
  double epsilon;
  double delta;
};

inline double AdvancedCompositionEpsilon(double epsilon_per_tree,
                                         size_t trees_num,
                                         double delta_prime) {
  const double trees = static_cast<double>(trees_num);
  return std::sqrt(2.0 * trees * std::log(1.0 / delta_prime)) *
             epsilon_per_tree +
         trees * epsilon_per_tree * std::expm1(epsilon_per_tree);
}

inline PrivacyBudget SplitPrivacyBudgetByAdvancedComposition(
    double epsilon, double delta, size_t trees_num) {
  CHECK_GT(epsilon, 0.0);
  CHECK_GT(delta, 0.0);
  CHECK_LT(delta, 1.0);

  if (trees_num <= 1) {
    return PrivacyBudget{epsilon, delta};
  }

  const double delta_prime = delta / 2.0;
  const double delta_per_tree = (delta - delta_prime) /
                                static_cast<double>(trees_num);

  double lower = 0.0;
  double upper = epsilon;
  while (AdvancedCompositionEpsilon(upper, trees_num, delta_prime) < epsilon) {
    upper *= 2.0;
  }

  for (int i = 0; i < 80; ++i) {
    const double mid = (lower + upper) / 2.0;
    if (AdvancedCompositionEpsilon(mid, trees_num, delta_prime) <= epsilon) {
      lower = mid;
    } else {
      upper = mid;
    }
  }

  return PrivacyBudget{lower, delta_per_tree};
}

#define PSRR_OSHUFFLE

class DOoperator
{
private:
  double epsilon;
  double delta;
  double sensitivity;
  double sigma;
  double mean;
public:
  xgboost::SparsePage shuffle_page;
  std::vector<int> shuffle_index;
  std::vector<xgboost::bst_float> shuffle_preds;
  #ifdef PSRR_OSHUFFLE
  std::unique_ptr<obl::OShuffler> oshuffler;
  #endif

  DOoperator(/* args */):DOoperator(1, 0.00001, 1){};
  DOoperator(double epsilon, double delta, double sensitivity):epsilon(epsilon),delta(delta),sensitivity(sensitivity){
    // sigma = calculateSigma(epsilon, delta, sensitivity);
    // mean = calculateMean(sigma, delta);
    // std::cout<<"sigma: "<<sigma<<" mean: "<<mean<<std::endl;
  };
  ~DOoperator(){};

  void clear(){

  }

  // Function to generate an n-dimensional array with elements from a standard normal distribution
  inline std::vector<int> generate_normal_distribution_array(int size) {
    std::vector<int> array(size);

    // Seed with a real random value, if available
    std::random_device rd;
    std::mt19937 gen(rd());

    // Standard normal distribution
    std::normal_distribution<> d(mean, sigma);

    for (int i = 0; i < size; ++i) {
        // Sample from the distribution and convert to positive integer
        array[i] = std::abs(static_cast<int>(std::round(d(gen))));
    }

    return array;
  }
  
  void ProduceDummySamples(xgboost::SparsePagePadding& dummySamples, xgboost::SparsePage& in_page){
    // std::cout<<"ProduceDummySamples size: "<<in_page.Size()<<std::endl;
    size_t nodes_in_cache_page = 200;
    for (size_t i = 0; i < in_page.Size(); i += nodes_in_cache_page)
    {
      dummySamples.ExpandAndWrite(i, in_page);
      // dummySamples.ExpandAndWrite(in_page[i*(in_page.Size()-1)/(samples_num-1)]);
      // dummySamples.Push(in_page[0]);
    }
  }

  /**
   * 在原始数据中添加噪声数量的伪数据
  */
  void AddDummy(xgboost::SparsePage& out_page, xgboost::SparsePage& dummySamples){
    std::vector<int> noise_vec = generate_normal_distribution_array(dummySamples.Size());

    // std::cout<<"noise_vec: ";
    // for (size_t i = 0; i < noise_vec.size(); i++)
    // {
    //   std::cout<<noise_vec[i]<<" ";
    // }
    // std::cout<<std::endl;

    // for (size_t i = 0; i < dummySamples.Size(); i++)
    // {
    //   for (size_t j = 0; j < noise_vec[i]; j++)
    //   {
    //     out_page.PushObliviousSrc(dummySamples, i);
    //   }
    // }
    size_t did=0;
    int noise_value=0;
    int count(0);
    while (did<dummySamples.Size())
    {
      out_page.PushObliviousSrc(dummySamples, did);
      count++;
      noise_value = ObliviousArrayAccess(noise_vec.data(), did, noise_vec.size());
      bool next = count>=noise_value;
      did = ObliviousChoose(next, did+1, did);
      count = ObliviousChoose(next, 0, count);
    }
    
  }

  template <typename Monitor>
  void Preprocess(xgboost::SparsePage& in_page, std::vector<xgboost::SparsePage>& trees_dummy_samples, size_t tree_nodes_num, Monitor* monitor_ = nullptr, int trees_num=1, int num_groups=1){
    CHECK_GT(trees_num, 0);
    PrivacyBudget per_tree_budget =
        SplitPrivacyBudgetByAdvancedComposition(
            epsilon, delta, static_cast<size_t>(trees_num));
    sigma = calculateSigma(epsilon, delta/trees_num,
                           sensitivity);
    mean = calculateMean(sigma, delta/trees_num);
    // size_t samples_num = (tree_nodes_num/2)/200;
    // std::cout<<"trees_num: "<<trees_num<<" samples_num: "<<samples_num<<std::endl;
    
    auto noise_page = xgboost::SparsePagePadding::FromSparsePage(in_page);

    for (size_t i = 0; i < trees_num; i++)
    {
      xgboost::SparsePagePadding dummySamples(noise_page.fixed_row_size);
      ProduceDummySamples(dummySamples, trees_dummy_samples[i]);

      // add dummy
      if (monitor_ != nullptr) monitor_->StartForce("AddDummy");
      AddDummy(noise_page, dummySamples);
      if (monitor_ != nullptr) monitor_->StopForce("AddDummy");
    }

    // std::cout<<"the number of dummies: "<<noise_page.Size()-in_page.Size()<<std::endl;
    
    // for (size_t  i = 0; i < noise_page.Size(); i++)
    // {
    //   std::cout << " "<<i<<": " << noise_page[i].size() << std::endl;
    // }
    
    // shuffle
    if (monitor_ != nullptr) monitor_->StartForce("shuffle");
    shuffle_index.resize(noise_page.Size());
    shuffle_preds.resize(noise_page.Size() * num_groups);
    // std::cout<<"noise_page.Size(): "<<noise_page.Size()<<std::endl;
    #ifdef PSRR_OSHUFFLE
    std::string shuffler_type = ReadParameterFromConfig<std::string>("/root/secure-xgboost/do-enhanced/data/config.txt", "shuffleMethod", "BitonicShuffler");
    logStr("/root/secure-xgboost/do-enhanced/data/time.log", "shuffleMethod: ", shuffler_type);
    oshuffler = obl::getShuffler(shuffler_type);
    // std::cout<<"noise_page.Size(): "<<noise_page.Size()<<" noise_page.fixed_row_size: "<<noise_page.fixed_row_size<<std::endl;
    // oshuffler = obl::create("BitonicShuffler");
    // auto temp_shuffler = new obl::BitonicShuffler;
    // for (size_t  i = 0; i < noise_page.Size(); i++)
    // {
    //   std::cout << " "<<i<<": " << noise_page[i].size() << std::endl;
    // }
    
    // std::cout<<"noise_page.data.HostVector().size(): "<<noise_page.data.HostVector().size()<<std::endl;
    // std::cout<<"noise_page.Size()*noise_page.fixed_row_size: "<<noise_page.Size()*noise_page.fixed_row_size<<std::endl;
    oshuffler->shuffle((uint8_t*)noise_page.data.HostVector().data(), noise_page.Size(), noise_page.fixed_row_size*sizeof(xgboost::Entry));
    // std::cout<<"shuffle_page.Size(): "<<shuffle_page.Size()<<std::endl;
    shuffle_page.Push(noise_page);
    // std::cout<<"shuffle_page.Size(): "<<shuffle_page.Size()<<std::endl;
    #else
    Shuffler& shuffler = Shuffler::getInstance();
    shuffler.shuffleForwardRandom(noise_page, shuffle_page, shuffle_index);
    #endif
    if (monitor_ != nullptr) monitor_->StopForce("shuffle");
  }

  template <typename Monitor>
  void Preprocess(xgboost::SparsePage& in_page,
                  xgboost::SparsePage& dummy_samples,
                  size_t dummy_max_entries,
                  Monitor* monitor_ = nullptr, int num_groups=1) {
    const size_t fixed_row_size =
        std::max(in_page.MaxNumberOfEntries(), dummy_max_entries);
    xgboost::SparsePagePadding noise_page(fixed_row_size);
    noise_page.FrommSparsePage(in_page);

    if (monitor_ != nullptr) monitor_->StartForce("AddDummy");
    for (size_t i = 0; i < dummy_samples.Size(); ++i) {
      noise_page.ExpandAndWrite(i, dummy_samples);
    }
    if (monitor_ != nullptr) monitor_->StopForce("AddDummy");

    if (monitor_ != nullptr) monitor_->StartForce("shuffle");
    shuffle_index.resize(noise_page.Size());
    shuffle_preds.resize(noise_page.Size() * num_groups);
    #ifdef PSRR_OSHUFFLE
    std::string shuffler_type = ReadParameterFromConfig<std::string>("/root/secure-xgboost/do-enhanced/data/config.txt", "shuffleMethod", "BitonicShuffler");
    logStr("/root/secure-xgboost/do-enhanced/data/time.log", "shuffleMethod: ", shuffler_type);
    oshuffler = obl::getShuffler(shuffler_type);
    oshuffler->shuffle((uint8_t*)noise_page.data.HostVector().data(), noise_page.Size(), noise_page.fixed_row_size*sizeof(xgboost::Entry));
    shuffle_page.Push(noise_page);
    #else
    Shuffler& shuffler = Shuffler::getInstance();
    shuffler.shuffleForwardRandom(noise_page, shuffle_page, shuffle_index);
    #endif
    if (monitor_ != nullptr) monitor_->StopForce("shuffle");
  }

  template <typename Monitor>
  void Preprocess(xgboost::SparsePage& in_page,
                  xgboost::SparsePage& dummy_samples,
                  Monitor* monitor_ = nullptr, int num_groups=1) {
    Preprocess(in_page, dummy_samples, dummy_samples.MaxNumberOfEntries(),
               monitor_, num_groups);
  }

  // another method of post process, deprecated
  template <typename Monitor>
  void PostProcessAdd(std::vector<xgboost::bst_float>* out_preds, Monitor* monitor_ = nullptr){
    if (monitor_ != nullptr) monitor_->StartForce("PostProcess");
    // for (size_t i = 0; i < shuffle_index.size(); i++)
    // {
    //   if (shuffle_index[i]>=out_preds->size())
    //   {
    //     continue;
    //   }
    //   (*out_preds)[shuffle_index[i]] += shuffle_preds[i];
    // }


    for (size_t i = 0; i < shuffle_index.size(); i++)
    {
      obliviousArrayElementAdd(out_preds->data(), shuffle_index[i], out_preds->size(), shuffle_preds[i]);
    }


    // BitonicSorter<int, xgboost::bst_float> sorter(shuffle_index, true, true);
    // sorter.reorder(shuffle_preds);
    
    // for (size_t i = 0; i < out_preds->size(); i++)
    // {
    //   (*out_preds)[i] += shuffle_preds[i];
    // }
    
    if (monitor_ != nullptr) monitor_->StopForce("PostProcess");
  }

  template <typename Monitor>
  void PostProcess(std::vector<xgboost::bst_float>* out_preds, Monitor* monitor_ = nullptr){
    if (monitor_ != nullptr) monitor_->StartForce("PostProcess");

    int num_groups = shuffle_preds.size() / shuffle_index.size();

    // std::cout<<"shuffle_index.size(): "<<shuffle_index.size()<<" num_groups: "<<num_groups<<std::endl;
    #ifdef PSRR_OSHUFFLE
    oshuffler->inverseShuffle((uint8_t*)shuffle_preds.data(), num_groups*sizeof(xgboost::bst_float));
    memcpy(out_preds->data(), shuffle_preds.data(), out_preds->size()*sizeof(xgboost::bst_float));
    #else
    for (size_t i = 0; i < shuffle_index.size(); i++)
    {
      // if (shuffle_index[i]>=out_preds->size()/num_groups)
      // {
      //   continue;
      // }

      int n = out_preds->size()/num_groups;
      int index = ObliviousChoose<int>(shuffle_index[i]<n, shuffle_index[i], shuffle_index[i]+CACHE_LINE_SIZE);

      ObliviousArrayAssignBytes(out_preds->data(), shuffle_preds.data() + i * num_groups, num_groups*sizeof(xgboost::bst_float), index, n);
    }
    #endif
    
    if (monitor_ != nullptr) monitor_->StopForce("PostProcess");
  }
};
