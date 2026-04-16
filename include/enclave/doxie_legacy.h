// Legacy differentially oblivious tree traversal experiments.
// This header is included from xgboost/tree_model.h inside namespace xgboost.

constexpr int cache_size = 4 * 1024;

class SMemory {
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

  explicit SMemory(int nbukkets) : nbukkets_(nbukkets) {
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
  explicit SQueue(int nbukkets = 1) : SMemory(nbukkets) {
    capacity_bukket = cache_size / sizeof(T) - 1;
    capacity_ = nbukkets_ * capacity_bukket;

    data_ = new T*[nbukkets_];
    for (int i = 0; i < nbukkets_; i++) {
      data_[i] = new T[capacity_bukket + 1];
    }
    readBuf_ = new T[nbukkets_];
  }
  ~SQueue() {
    for (int i = 0; i < nbukkets_; i++) {
      delete[] data_[i];
    }
    delete[] data_;
    data_ = nullptr;
    delete[] readBuf_;
    readBuf_ = nullptr;
  }

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
  }
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
  }

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

inline int DPPrefixSum(int prefix_sum, int error) {
  int min = -error + 1;
  int max = error - 1;
  int noise = (rand() % (max - min + 1)) + min;
  return prefix_sum + noise;
}

inline bst_float RegTree::DPOGetLeafValue(const RegTree::FVec& feat,
                                          unsigned root_id) {
  xgboost::bst_float out_value;
  bool in_stash = stash_.GetLeafValue(feat, &out_value);
  bst_node_t nid = 0;
  while (!(*this)[nid].IsLeaf()) {
    unsigned split_index = (*this)[nid].SplitIndex();
    bst_node_t true_nid = this->GetNext(nid, feat.GetFvalue(split_index),
                                        feat.IsMissing(split_index));
    float r = static_cast<float>(rand()) / static_cast<float>(RAND_MAX);
    bst_node_t random_nid = ObliviousChoose(r < 0.5, (*this)[nid].LeftChild(),
                                            (*this)[nid].RightChild());
    nid = ObliviousChoose(in_stash, random_nid, true_nid);
  }
  return ObliviousChoose(in_stash, out_value, (*this)[nid].LeafValue());
}

inline void RegTree::OPredictByHistNoCache(DMatrix* p_fmat,
                                           std::vector<bst_float>* out_preds,
                                           int32_t gid, int32_t num_group,
                                           RegTree::FVec& feat) {
  std::vector<int> index;
  index.resize(p_fmat->Info().num_row_);
  std::fill(index.begin(), index.end(), 0);
  std::vector<bst_float>& preds = *out_preds;
  for (int nid = 0; nid < this->GetNodes().size(); nid++) {
    for (auto const& batch : p_fmat->GetBatches<SparsePage>()) {
      auto nsize = batch.Size();
      for (int i = 0; i < nsize; i++) {
        bool is_in_node = index[batch.base_rowid + i] == nid;
        if (!(*this)[nid].IsLeaf()) {
          feat.Fill(batch[i]);
          unsigned split_index = (*this)[nid].SplitIndex();
          auto next = OGetNext(nid, feat.GetFvalue(split_index),
                               feat.IsMissing(split_index));
          index[batch.base_rowid + i] =
              ObliviousChoose(is_in_node, next, index[batch.base_rowid + i]);
          feat.Drop(batch[i]);
        } else {
          auto leaf_value =
              ObliviousChoose(is_in_node, (*this)[nid].LeafValue(), 0.0f);
          preds[(batch.base_rowid + i) * num_group + gid] += leaf_value;
        }
      }
    }
  }
}

inline void RegTree::OPredictByHist(DMatrix* p_fmat,
                                    std::vector<bst_float>* out_preds,
                                    int32_t gid, int32_t num_group,
                                    RegTree::FVec& feat) {
  std::vector<int> index;
  index.resize(p_fmat->Info().num_row_);
  std::fill(index.begin(), index.end(), 0);

  std::vector<bool> preded;
  preded.resize(p_fmat->Info().num_row_);
  std::fill(preded.begin(), preded.end(), false);

  std::vector<bst_float>& preds = *out_preds;
  int num_nodes_page = 4096 / sizeof(Node);
  for (int npage = 0; npage <= this->GetNodes().size() / num_nodes_page;
       npage++) {
    for (auto const& batch : p_fmat->GetBatches<SparsePage>()) {
      auto nsize = batch.Size();
      for (int i = 0; i < nsize; i++) {
        if (npage == 0) {
          while (index[batch.base_rowid + i] >= npage * num_nodes_page &&
                 index[batch.base_rowid + i] < (npage + 1) * num_nodes_page &&
                 index[batch.base_rowid + i] < this->GetNodes().size()) {
            feat.Fill(batch[i]);
            bool is_leaf = (*this)[index[batch.base_rowid + i]].IsLeaf();
            auto leaf_value = ObliviousChoose(
                is_leaf, (*this)[index[batch.base_rowid + i]].LeafValue(),
                0.0f);
            preds[(batch.base_rowid + i) * num_group + gid] += leaf_value;

            preded[batch.base_rowid + i] = ObliviousChoose(
                is_leaf, true, bool(preded[batch.base_rowid + i]));
            unsigned split_index = ObliviousChoose(
                preded[batch.base_rowid + i], 0u,
                (*this)[index[batch.base_rowid + i]].SplitIndex());
            auto next = OGetNext(index[batch.base_rowid + i],
                                 feat.GetFvalue(split_index),
                                 feat.IsMissing(split_index));

            index[batch.base_rowid + i] =
                ObliviousChoose(preded[batch.base_rowid + i],
                                index[batch.base_rowid + i] * 2 + 1, next);
            feat.Drop(batch[i]);
          }
        } else {
          bool in_page =
              index[batch.base_rowid + i] >= npage * num_nodes_page &&
              index[batch.base_rowid + i] < (npage + 1) * num_nodes_page &&
              index[batch.base_rowid + i] < this->GetNodes().size();
          int nid = ObliviousChoose(in_page, int(index[batch.base_rowid + i]),
                                    npage * num_nodes_page);
          feat.Fill(batch[i]);
          bool is_leaf = (*this)[nid].IsLeaf();
          auto leaf_value = ObliviousChoose(is_leaf && in_page,
                                            (*this)[nid].LeafValue(), 0.0f);
          preds[(batch.base_rowid + i) * num_group + gid] += leaf_value;

          preded[batch.base_rowid + i] = ObliviousChoose(
              is_leaf && in_page, true, bool(preded[batch.base_rowid + i]));
          unsigned split_index =
              ObliviousChoose(preded[batch.base_rowid + i] || !in_page, 0u,
                              (*this)[nid].SplitIndex());
          auto next = OGetNext(nid, feat.GetFvalue(split_index),
                               feat.IsMissing(split_index));

          index[batch.base_rowid + i] = ObliviousChoose(
              in_page,
              ObliviousChoose(preded[batch.base_rowid + i],
                              index[batch.base_rowid + i] * 2 + 1, next),
              index[batch.base_rowid + i]);
          feat.Drop(batch[i]);
        }
      }
    }
  }
}

inline void RegTree::DPOPredictByHist(DMatrix* p_fmat,
                                      std::vector<bst_float>* out_preds,
                                      int32_t gid, int32_t num_group,
                                      RegTree::FVec& feat) {
  std::vector<std::vector<FVecIndex>*> hist;
  hist.resize(this->GetNodes().size());
  for (int i = 0; i < hist.size(); i++) {
    hist[i] = new std::vector<FVecIndex>;
  }

  int num_nodes_page = 4096 / sizeof(Node);
  for (auto const& batch : p_fmat->GetBatches<SparsePage>()) {
    auto nsize = batch.Size();
    for (int i = 0; i < nsize; i++) {
      FVecIndex entry;
      auto& p_feat = entry.feat;
      p_feat.Init(feat.Size());
      p_feat.Fill(batch[i]);
      int nid = 0;
      while (nid >= 0 && nid < num_nodes_page && !(*this)[nid].IsLeaf()) {
        unsigned split_index = (*this)[nid].SplitIndex();
        auto next = GetNext(nid, p_feat.GetFvalue(split_index),
                            p_feat.IsMissing(split_index));
        nid = next;
      }
      entry.index = batch.base_rowid + i;
      hist[nid]->push_back(std::move(entry));
    }
  }
  std::vector<bst_float>& preds = *out_preds;
  for (int nid = 0; nid < this->GetNodes().size(); nid++) {
    if ((*this)[nid].IsLeaf()) {
      for (auto& entry : *hist[nid]) {
        preds[entry.index * num_group + gid] += (*this)[nid].LeafValue();
      }
    } else {
      while (!hist[nid]->empty()) {
        FVecIndex& entry = hist[nid]->back();
        auto& p_feat = entry.feat;
        unsigned split_index = (*this)[nid].SplitIndex();
        auto next = GetNext(nid, p_feat.GetFvalue(split_index),
                            p_feat.IsMissing(split_index));
        hist[next]->push_back(std::move(entry));
        hist[nid]->pop_back();
      }
    }
    std::vector<FVecIndex>().swap(*hist[nid]);
  }
}

inline void RegTree::DPOPredictByHist1(DMatrix* p_fmat,
                                       std::vector<bst_float>* out_preds,
                                       int32_t gid, int32_t num_group,
                                       RegTree::FVec& feat) {
  std::vector<std::vector<FVecIndex>*> hist;
  hist.resize(this->GetNodes().size());
  for (int i = 0; i < hist.size(); i++) {
    hist[i] = new std::vector<FVecIndex>;
  }

  int num_nodes_page = 4096 / sizeof(Node);
  for (auto const& batch : p_fmat->GetBatches<SparsePage>()) {
    auto nsize = batch.Size();
    std::cout << "memcost:" << batch.MemCostBytes() << " size:" << nsize
              << " mem per sample:" << batch.MemCostBytes() / nsize
              << std::endl;
    for (int i = 0; i < nsize; i++) {
      FVecIndex entry;
      auto& p_feat = entry.feat;
      p_feat.Init(feat.Size());
      p_feat.Fill(batch[i]);
      entry.index = batch.base_rowid + i;
      hist[0]->push_back(std::move(entry));
    }
  }
  std::vector<bst_float>& preds = *out_preds;
  SQueue<FVecIndex> lbuf{25};
  SQueue<FVecIndex> rbuf{25};
  int num_round = lbuf.capacity() / 4;
  for (int nid = 0; nid < this->GetNodes().size(); nid++) {
    if ((*this)[nid].IsLeaf()) {
      bst_float leaf_value = (*this)[nid].LeafValue();
      for (auto& entry : *hist[nid]) {
        bst_float leaf_value_temp =
            ObliviousChoose(entry.index < 0, 0.0f, leaf_value);
        int index = ObliviousChoose(entry.index < 0, -entry.index, entry.index);
        preds[index * num_group + gid] += leaf_value_temp;
      }
    } else {
      int prefix_sum_l = 0;
      int prefix_sum_r = 0;
      for (int round_base = 0; round_base < hist[nid]->size();
           round_base += num_round) {
        prefix_sum_l -= lbuf.size();
        prefix_sum_r -= rbuf.size();
        int s = std::min(num_round, int(hist[nid]->size() - round_base));
        for (int i = 0; i < s; i++) {
          FVecIndex& entry = hist[nid]->at(round_base + i);
          auto& p_feat = entry.feat;
          unsigned split_index = (*this)[nid].SplitIndex();
          auto next = GetNext(nid, p_feat.GetFvalue(split_index),
                              p_feat.IsMissing(split_index));

          lbuf.push_back(entry, (next == (*this)[nid].LeftChild()));
          rbuf.push_back(entry, (next == (*this)[nid].RightChild()));
        }
        prefix_sum_l += lbuf.size();
        prefix_sum_r += rbuf.size();

        FillOutput(lbuf, hist[(*this)[nid].LeftChild()],
                   DPPrefixSum(prefix_sum_l, num_round) - num_round);
        FillOutput(rbuf, hist[(*this)[nid].RightChild()],
                   DPPrefixSum(prefix_sum_r, num_round) - num_round);
      }
      FillOutput(lbuf, hist[(*this)[nid].LeftChild()],
                 DPPrefixSum(prefix_sum_l, num_round) + num_round);
      FillOutput(rbuf, hist[(*this)[nid].RightChild()],
                 DPPrefixSum(prefix_sum_r, num_round) + num_round);
    }
    std::vector<FVecIndex>().swap(*hist[nid]);
  }
}

inline void RegTree::FillOutput(SQueue<FVecIndex>& buffer,
                                std::vector<FVecIndex>* output, int out_size) {
  while (int(output->size()) < out_size) {
    FVecIndex temp;
    auto real = buffer.pop_font(temp);
    temp.index = ObliviousChoose(real, temp.index, -1);
    output->push_back(std::move(temp));
  }
}

inline void RegTree::DPOPredictByHist2(DMatrix* p_fmat,
                                       std::vector<bst_float>* out_preds,
                                       int32_t gid, int32_t num_group,
                                       RegTree::FVec& feat) {
  std::vector<std::vector<FVecIndex>*> hist;
  hist.resize(this->GetNodes().size());
  for (int i = 0; i < hist.size(); i++) {
    hist[i] = new std::vector<FVecIndex>;
  }

  int num_nodes_page = 4096 / sizeof(Node);
  for (auto const& batch : p_fmat->GetBatches<SparsePage>()) {
    auto nsize = batch.Size();
    for (int i = 0; i < nsize; i++) {
      FVecIndex entry;
      auto& p_feat = entry.feat;
      p_feat.Init(feat.Size());
      p_feat.Fill(batch[i]);
      entry.index = batch.base_rowid + i;
      hist[0]->push_back(std::move(entry));
    }
  }
  std::vector<bst_float>& preds = *out_preds;
  int vec_size = feat.Size();
  SQueue<std::vector<RegTree::FVec::Entry>> lbuf{vec_size};
  SQueue<std::vector<RegTree::FVec::Entry>> rbuf{vec_size};
  int num_round = lbuf.capacity() / 4;
  for (int nid = 0; nid < this->GetNodes().size(); nid++) {
    if ((*this)[nid].IsLeaf()) {
      bst_float leaf_value = (*this)[nid].LeafValue();
      for (auto& entry : *hist[nid]) {
        bst_float leaf_value_temp =
            ObliviousChoose(entry.index < 0, 0.0f, leaf_value);
        int index = ObliviousChoose(entry.index < 0, -entry.index, entry.index);
        preds[index * num_group + gid] += leaf_value_temp;
      }
    } else {
      int prefix_sum_l = 0;
      int prefix_sum_r = 0;
      for (int round_base = 0; round_base < hist[nid]->size();
           round_base += num_round) {
        prefix_sum_l -= lbuf.size();
        prefix_sum_r -= rbuf.size();
        int s = std::min(num_round, int(hist[nid]->size() - round_base));
        for (int i = 0; i < s; i++) {
          FVecIndex& entry = hist[nid]->at(round_base + i);
          auto& p_feat = entry.feat;
          unsigned split_index = (*this)[nid].SplitIndex();
          auto next = GetNext(nid, p_feat.GetFvalue(split_index),
                              p_feat.IsMissing(split_index));

          lbuf.push_back(entry.feat.Data(), entry.index,
                         (next == (*this)[nid].LeftChild()));
          rbuf.push_back(entry.feat.Data(), entry.index,
                         (next == (*this)[nid].RightChild()));
        }
        prefix_sum_l += lbuf.size();
        prefix_sum_r += rbuf.size();

        FillOutput2(lbuf, hist[(*this)[nid].LeftChild()],
                    DPPrefixSum(prefix_sum_l, num_round) - num_round);
        FillOutput2(rbuf, hist[(*this)[nid].RightChild()],
                    DPPrefixSum(prefix_sum_r, num_round) - num_round);
      }
      FillOutput2(lbuf, hist[(*this)[nid].LeftChild()],
                  DPPrefixSum(prefix_sum_l, num_round) + num_round);
      FillOutput2(rbuf, hist[(*this)[nid].RightChild()],
                  DPPrefixSum(prefix_sum_r, num_round) + num_round);
    }
    std::vector<FVecIndex>().swap(*hist[nid]);
  }
}

inline void RegTree::FillOutput2(
    SQueue<std::vector<RegTree::FVec::Entry>>& buffer,
    std::vector<FVecIndex>* output, int out_size) {
  while (int(output->size()) < out_size) {
    FVecIndex temp;
    temp.feat.Init(buffer.vecSize());
    auto real = buffer.pop_font(temp.feat.Data(), temp.index);
    temp.index = ObliviousChoose(real, temp.index, -1);
    output->push_back(std::move(temp));
  }
}

inline int LeftPage(int page_num, int nodes_in_page) {
  return (page_num * nodes_in_page * 2 + 1) / nodes_in_page;
}

inline int RightPage(int page_num, int nodes_in_page) {
  return (((page_num + 1) * nodes_in_page - 1) * 2 + 2) / nodes_in_page;
}

inline void RegTree::DPOPredictByHist3(DMatrix* p_fmat,
                                       std::vector<bst_float>* out_preds,
                                       int32_t gid, int32_t num_group,
                                       RegTree::FVec& feat) {
  int nodes_in_page = 4;
  int npages = this->GetNodes().size() / nodes_in_page + 1;
  std::vector<SQueue<std::vector<RegTree::FVec::Entry>>*> hist;
  hist.resize(npages, nullptr);
  int vec_size = feat.Size();
  for (int i = 0; i < hist.size(); i++) {
    hist[i] = new SQueue<std::vector<RegTree::FVec::Entry>>(vec_size, 6);
  }
  std::vector<bst_float>& preds = *out_preds;
  int num_round = hist[0]->capacity() / 3;

  for (auto const& batch : p_fmat->GetBatches<SparsePage>()) {
    auto nsize = batch.Size();
    for (int round_base = 0; round_base < nsize; round_base += num_round) {
      int s = std::min(num_round, int(nsize - round_base));
      for (int i = 0; i < s; i++) {
        int index = batch.base_rowid + round_base + i;
        feat.Fill(batch[round_base + i]);
        int nid = 0;
        while (nid < nodes_in_page) {
          unsigned split_index = (*this)[nid].SplitIndex();
          nid = GetNext(nid, feat.GetFvalue(split_index),
                        feat.IsMissing(split_index));
          if ((*this)[nid].IsLeaf()) {
            (*out_preds)[index * num_group + gid] += (*this)[nid].LeafValue();
            break;
          }
        }
        for (int page_num_next = std::max(LeftPage(0, nodes_in_page), 1);
             page_num_next <= RightPage(0, nodes_in_page); page_num_next++) {
          if (page_num_next < hist.size() && !(*this)[nid].IsLeaf()) {
            hist[page_num_next]->push_back(
                feat.Data(), index, nid, page_num_next == int(nid / nodes_in_page));
          }
        }
        feat.Drop(batch[round_base + i]);
      }
      PagePredict(hist, feat, out_preds, gid, num_group, 3, num_round,
                  nodes_in_page);
      PagePredict(hist, feat, out_preds, gid, num_group, 2, num_round,
                  nodes_in_page);
      PagePredict(hist, feat, out_preds, gid, num_group, 1, num_round,
                  nodes_in_page);
      PagePredict(hist, feat, out_preds, gid, num_group, 0, num_round,
                  nodes_in_page);
    }
  }
  for (int i = 0; i < hist.size(); i++) {
    PagePredict(hist, feat, out_preds, gid, num_group, i, hist[i]->size(),
                nodes_in_page);
  }
}

inline void RegTree::PagePredict(
    std::vector<SQueue<std::vector<RegTree::FVec::Entry>>*>& hist,
    RegTree::FVec& feat, std::vector<bst_float>* out_preds, int32_t gid,
    int32_t num_group, int page_num, int num_round, int nodes_in_page) {
  if (hist.size() <= page_num) return;
  if (hist[page_num]->size() < num_round) return;

  for (int i = 0; i < num_round; i++) {
    int index;
    int nid;
    hist[page_num]->pop_font(feat.Data(), index, nid);
    unsigned split_index = (*this)[nid].SplitIndex();
    auto next = GetNext(nid, feat.GetFvalue(split_index),
                        feat.IsMissing(split_index));
    if ((*this)[next].IsLeaf()) {
      (*out_preds)[index * num_group + gid] += (*this)[next].LeafValue();
    } else {
      for (int page_num_next =
               std::max(LeftPage(page_num, nodes_in_page), page_num + 1);
           page_num_next <= RightPage(page_num, nodes_in_page);
           page_num_next++) {
        if (page_num_next < hist.size()) {
          hist[page_num_next]->push_back(
              feat.Data(), index, next,
              page_num_next == int(next / nodes_in_page));
        }
      }
    }
  }
  for (int page_num_next =
           std::max(LeftPage(page_num, nodes_in_page), page_num + 1);
       page_num_next <= RightPage(page_num, nodes_in_page); page_num_next++) {
    PagePredict(hist, feat, out_preds, gid, num_group, page_num_next,
                num_round, nodes_in_page);
  }
}
