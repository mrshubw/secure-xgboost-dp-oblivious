#pragma once

#include <cstddef>
#include <cstdint>
#include <cstring>
#include <type_traits>

#include "xgboost/base.h"
#include "xgboost/data.h"

constexpr int cache_size = 4 * 1024;

// template <typename T>
// class Stash
// {
// private:
//     T[]* stash_;
//     std::vector<T>* array_;
//     uint16_t[]* array_to_stash;
//     bool inited;
// public:
//     Stash(std::vector<T>* array):array_{array}{

//     };
// };

// template <typename T>
// class NodeStash {
//  private:
//   using index_type = uint16_t;
//   static const size_t capacity_ = 256;
//   static const index_type kInvalidIndex = capacity_;
//   xgboost::RegTree::Node stash_[capacity_];
//   index_type free_stash[capacity_];
//   size_t size_ = 0;
//   index_type* tree2stash;
//   xgboost::RegTree* tree_;
//   float rate = 0;
//   bool inited = false;

//  public:
//   NodeStash() {
//   };
//   ~NodeStash() { delete[] tree2stash; };

//   inline void SetTree(xgboost::RegTree* tree) {
//     tree_ = tree;
//     tree2stash = new index_type[tree->GetNodes().size()]{kInvalidIndex};
//     rate = static_cast<float>(capacity_) /
//            static_cast<float>(tree->GetNodes().size());
//     for (index_type i = 0; i < capacity_; i++) {
//       free_stash[i] = i;
//     }
//     size_ = capacity_;
//   };

//   inline void InitStash() {
//     if (inited) {
//       return;
//     }
//     inited = true;
//     for (size_t nid = 0; nid < tree_->GetNodes().size(); nid++) {
//       if ((*tree_)[nid].IsLeaf()) {
//         float r = static_cast<float>(rand()) / static_cast<float>(RAND_MAX);
//         if (r < rate) {
//           size_t nid_ = nid;
//           while (!(*tree_)[nid_].IsRoot()) {
//             Set(nid_, (*tree_)[nid_]);
//             nid_ = (*tree_)[nid_].Parent();
//           }
//           Set(nid_, (*tree_)[nid_]);
//         }
//       }
//     }
//   }

//   inline index_type Alloc() {
//     if (size_ > 0) {
//       size_--;
//       return free_stash[size_];
//     }
//     return kInvalidIndex;
//   }

//   inline bool Set(int nid, xgboost::RegTree::Node& node) {
//     if (tree2stash[nid] == kInvalidIndex) {
//       tree2stash[nid] = Alloc();
//       if (tree2stash[nid] == kInvalidIndex) return false;
//     }
//     stash_[tree2stash[nid]] = node;
//     return true;
//   }

//   inline bool Get(int nid, xgboost::RegTree::Node* out) {
//     if (tree2stash[nid] == kInvalidIndex) {
//       return false;
//     }
//     *out = stash_[tree2stash[nid]];
//     return true;
//   }

//   inline bool GetLeafValue(const xgboost::RegTree::FVec& feat,
//                            xgboost::bst_float* out_value) {
//     xgboost::bst_node_t nid = 0;
//     xgboost::RegTree::Node node;
//     while (Get(nid, &node)) {
//       if (node.IsLeaf()) {
//         *out_value = node.LeafValue();
//         return true;
//       }

//       unsigned split_index = node.SplitIndex();
//       xgboost::bst_float split_value = node.SplitCond();
//       if (feat.IsMissing(split_index)) {
//         nid = node.DefaultChild();
//       } else {
//         if (feat.GetFvalue(split_index) < split_value) {
//           nid = node.LeftChild();
//         } else {
//           nid = node.RightChild();
//         }
//       }
//     }
//     return false;
//   }
// };
