/*!
 * Copyright by Contributors 2017-2020
 * Modifications Copyright 2020-22 by Secure XGBoost Contributors
 */
#include "enclave/doxie_inference.h"

#include <dmlc/omp.h>

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <limits>
#include <random>
#include <type_traits>
#include <vector>

#include "../common/timer.h"
#include "../gbm/gbtree_model.h"
#include "psrr/doxie_memory.h"
#include "xgboost/logging.h"

namespace xgboost {
namespace doxie {
namespace {

template <size_t kUnrollLen = 8>
struct SparsePageView {
  SparsePage const* page;
  bst_row_t base_rowid;
  static size_t constexpr kUnroll = kUnrollLen;

  explicit SparsePageView(SparsePage const* p)
      : page{p}, base_rowid{page->base_rowid} {
    page->data.HostVector();
    page->offset.HostVector();
  }
  SparsePage::Inst operator[](size_t i) { return (*page)[i]; }
  size_t Size() const { return page->Size(); }
};

struct FeatureDomain {
  unsigned feature{0};
  bst_float lower{std::numeric_limits<bst_float>::lowest()};
  bst_float upper{std::numeric_limits<bst_float>::max()};
};

using NodeDomain = std::vector<FeatureDomain>;

struct NoisePosition {
  size_t tree_pos;
  size_t representative_pos;
  bool found;

  NoisePosition() : tree_pos(0), representative_pos(0), found(false) {}
  NoisePosition(size_t tree, size_t representative)
      : tree_pos(tree), representative_pos(representative), found(true) {}
};

struct DummySamples {
  SparsePage page;
  size_t max_entries{0};
};

struct DenseWorkingDomain {
  std::vector<unsigned> features;
  std::vector<bst_float> lower;
  std::vector<bst_float> upper;
  std::vector<uint8_t> active;
};

struct IndexedFeatureDomain {
  size_t index{0};
  unsigned feature{0};
  bst_float lower{std::numeric_limits<bst_float>::lowest()};
  bst_float upper{std::numeric_limits<bst_float>::max()};
};

using IndexedNodeDomain = std::vector<IndexedFeatureDomain>;
using IndexedDomains = std::vector<std::vector<IndexedNodeDomain>>;

void InitThreadTemp(int nthread, int num_feature,
                    std::vector<RegTree::FVec>* out) {
  int prev_thread_temp_size = out->size();
  if (prev_thread_temp_size < nthread) {
    out->resize(nthread, RegTree::FVec());
    for (int i = prev_thread_temp_size; i < nthread; ++i) {
      (*out)[i].Init(num_feature);
    }
  }
}

bst_float PredValue(const SparsePage::Inst& inst,
                    const std::vector<std::unique_ptr<RegTree>>& trees,
                    const std::vector<int>& tree_info, int bst_group,
                    RegTree::FVec* p_feats, unsigned tree_begin,
                    unsigned tree_end, common::Monitor* monitor = nullptr) {
  bst_float psum = 0.0f;
  p_feats->Fill(inst);
  for (size_t i = tree_begin; i < tree_end; ++i) {
    if (tree_info[i] == bst_group) {
      if (trees[i]->HasDoxieMemory()) {
        psum += trees[i]->GetLeafValueDoxie(*p_feats);
      } else {
        int tid = trees[i]->GetLeafIndex(*p_feats);
        psum += (*trees[i])[tid].LeafValue();
      }
    }
  }
  p_feats->Drop(inst);
  return psum;
}

template <typename DataView>
void PredictBatchKernel(DataView batch, std::vector<bst_float>* out_preds,
                        gbm::GBTreeModel const& model, int32_t tree_begin,
                        int32_t tree_end,
                        std::vector<RegTree::FVec>* p_thread_temp,
                        common::Monitor* monitor = nullptr) {
  if (monitor != nullptr) monitor->Start(__func__);
  auto& thread_temp = *p_thread_temp;
  int32_t const num_group = model.learner_model_param->num_output_group;

  std::vector<bst_float>& preds = *out_preds;
  CHECK_EQ(model.param.size_leaf_vector, 0)
      << "size_leaf_vector is enforced to 0 so far";

  const auto nsize = static_cast<bst_omp_uint>(batch.Size());
  auto constexpr kUnroll = DataView::kUnroll;
  const bst_omp_uint rest = nsize % kUnroll;
  if (nsize >= kUnroll) {
#pragma omp parallel for schedule(static)
    for (bst_omp_uint i = 0; i < nsize - rest; i += kUnroll) {
      const int tid = omp_get_thread_num();
      RegTree::FVec& feats = thread_temp[tid];
      int64_t ridx[kUnroll];
      SparsePage::Inst inst[kUnroll];
      for (size_t k = 0; k < kUnroll; ++k) {
        ridx[k] = static_cast<int64_t>(batch.base_rowid + i + k);
      }
      for (size_t k = 0; k < kUnroll; ++k) {
        inst[k] = batch[i + k];
      }
      for (size_t k = 0; k < kUnroll; ++k) {
        for (int gid = 0; gid < num_group; ++gid) {
          const size_t offset = ridx[k] * num_group + gid;
          preds[offset] += PredValue(inst[k], model.trees, model.tree_info, gid,
                                     &feats, tree_begin, tree_end, monitor);
        }
      }
    }
  }
  for (bst_omp_uint i = nsize - rest; i < nsize; ++i) {
    RegTree::FVec& feats = thread_temp[0];
    const auto ridx = static_cast<int64_t>(batch.base_rowid + i);
    auto inst = batch[i];
    for (int gid = 0; gid < num_group; ++gid) {
      const size_t offset = ridx * num_group + gid;
      preds[offset] += PredValue(inst, model.trees, model.tree_info, gid,
                                 &feats, tree_begin, tree_end, monitor);
    }
  }
  if (monitor != nullptr) monitor->Stop(__func__);
}

template <typename DataView>
void PredictBatchKernelBlocked(DataView batch,
                               std::vector<bst_float>* out_preds,
                               gbm::GBTreeModel const& model,
                               int32_t tree_begin, int32_t tree_end,
                               common::Monitor* monitor = nullptr) {
  if (monitor != nullptr) monitor->Start(__func__);
  int32_t const num_group = model.learner_model_param->num_output_group;
  int32_t const num_feature = model.learner_model_param->num_feature;

  std::vector<bst_float>& preds = *out_preds;
  CHECK_EQ(model.param.size_leaf_vector, 0)
      << "size_leaf_vector is enforced to 0 so far";

  const auto nsize = static_cast<bst_omp_uint>(batch.Size());
  bst_omp_uint constexpr kBlockSize = 64;

#pragma omp parallel
  {
    std::vector<RegTree::FVec> feats(kBlockSize);
    std::vector<SparsePage::Inst> inst(kBlockSize);
    for (auto& feat : feats) {
      feat.Init(num_feature);
    }

#pragma omp for schedule(static)
    for (bst_omp_uint block_begin = 0; block_begin < nsize;
         block_begin += kBlockSize) {
      const bst_omp_uint block_end =
          std::min<bst_omp_uint>(nsize, block_begin + kBlockSize);
      const bst_omp_uint block_len = block_end - block_begin;

      for (bst_omp_uint k = 0; k < block_len; ++k) {
        inst[k] = batch[block_begin + k];
        feats[k].Fill(inst[k]);
      }

      for (int32_t tree_id = tree_begin; tree_id < tree_end; ++tree_id) {
        const int gid = model.tree_info[tree_id];
        CHECK_LT(gid, num_group);
        const RegTree& tree = *model.trees[tree_id];
        for (bst_omp_uint k = 0; k < block_len; ++k) {
          const auto ridx =
              static_cast<int64_t>(batch.base_rowid + block_begin + k);
          const size_t offset = ridx * num_group + gid;
          preds[offset] += tree.GetLeafValueDoxie(feats[k]);
        }
      }

      for (bst_omp_uint k = 0; k < block_len; ++k) {
        feats[k].Drop(inst[k]);
      }
    }
  }

  if (monitor != nullptr) monitor->Stop(__func__);
}

std::vector<int> CollectLeafPageRepresentatives(
    gbm::GBTreeModel const& model, int32_t tree_begin, int32_t tree_end) {
  CHECK_GE(tree_begin, 0);
  CHECK_GE(tree_end, tree_begin);
  CHECK_LE(static_cast<size_t>(tree_end), model.trees.size());

  std::vector<int> representatives;
  if (tree_begin == tree_end) {
    return representatives;
  }

  const auto& nodes = model.trees[tree_begin]->GetNodes();
  const size_t leaf_begin = nodes.size() / 2;
  const size_t leaf_count = nodes.size() - leaf_begin;
  const size_t leaves_per_page = DOXIE_NODES_PER_PAGE;
  const size_t page_count =
      (leaf_count + leaves_per_page - 1) / leaves_per_page;

  representatives.reserve(page_count);
  for (size_t page = 0; page < page_count; ++page) {
    const size_t page_leaf_begin = page * leaves_per_page;
    const size_t page_leaf_end =
        std::min(leaf_count, page_leaf_begin + leaves_per_page);

    size_t subtree_leaf_begin = page_leaf_begin;
    size_t subtree_leaf_count = 1;
    for (size_t count = 2; count <= page_leaf_end - page_leaf_begin;
         count <<= 1) {
      const size_t aligned_begin =
          ((page_leaf_begin + count - 1) / count) * count;
      if (aligned_begin + count <= page_leaf_end) {
        subtree_leaf_begin = aligned_begin;
        subtree_leaf_count = count;
      }
    }

    size_t nid = leaf_begin + subtree_leaf_begin;
    for (size_t count = subtree_leaf_count; count > 1; count >>= 1) {
      nid = (nid - 1) / 2;
    }
    representatives.push_back(static_cast<int>(nid));
  }
  return representatives;
}

void AddDomainConstraint(NodeDomain* domain, unsigned feature,
                         bst_float split_value, bool go_left) {
  for (auto& item : *domain) {
    if (item.feature == feature) {
      if (go_left) {
        item.upper = std::min(item.upper, split_value);
      } else {
        item.lower = std::max(item.lower, split_value);
      }
      return;
    }
  }

  FeatureDomain item;
  item.feature = feature;
  if (go_left) {
    item.upper = split_value;
  } else {
    item.lower = split_value;
  }
  domain->push_back(item);
}

NodeDomain BuildNodeDomain(RegTree const& tree, int representative_nid) {
  NodeDomain domain;
  domain.reserve(8);
  bst_node_t nid = representative_nid;
  while (nid != 0) {
    const auto& node = tree[nid];
    const bst_node_t parent_nid = node.Parent();
    const auto& parent = tree[parent_nid];
    const bool go_left = nid == parent.LeftChild();
    AddDomainConstraint(&domain, parent.SplitIndex(), parent.SplitCond(),
                        go_left);
    nid = parent_nid;
  }
  return domain;
}

std::vector<std::vector<NodeDomain>> CollectRepresentativeDomains(
    gbm::GBTreeModel const& model, int32_t tree_begin, int32_t tree_end,
    const std::vector<int>& representatives) {
  CHECK_GE(tree_begin, 0);
  CHECK_GE(tree_end, tree_begin);
  CHECK_LE(static_cast<size_t>(tree_end), model.trees.size());

  std::vector<std::vector<NodeDomain>> domains(tree_end - tree_begin);
  for (int32_t tree_id = tree_begin; tree_id < tree_end; ++tree_id) {
    auto& tree_domains = domains[tree_id - tree_begin];
    tree_domains.reserve(representatives.size());
    for (int nid : representatives) {
      tree_domains.push_back(BuildNodeDomain(*model.trees[tree_id], nid));
    }
  }
  return domains;
}

std::vector<std::vector<size_t>> SampleRepresentativeNoise(
    int32_t tree_begin, int32_t tree_end,
    const std::vector<int>& representatives, double epsilon, double delta,
    double sensitivity) {
  CHECK_GE(tree_begin, 0);
  CHECK_GE(tree_end, tree_begin);
  CHECK_GT(epsilon, 0.0);
  CHECK_GT(delta, 0.0);

  const size_t num_trees = static_cast<size_t>(tree_end - tree_begin);
  std::vector<std::vector<size_t>> noise(num_trees);
  if (num_trees == 0 || representatives.empty()) {
    return noise;
  }

  const PrivacyBudget per_tree_budget =
      SplitPrivacyBudgetByAdvancedComposition(epsilon, delta, num_trees);
  const double sigma =
      calculateSigma(per_tree_budget.epsilon, per_tree_budget.delta,
                     sensitivity);
  const double mean =
      calculateMean(sigma, per_tree_budget.delta, representatives.size());

  std::random_device rd;
  std::mt19937 gen(rd());
  std::normal_distribution<double> gaussian(mean, sigma);

  for (auto& tree_noise : noise) {
    tree_noise.reserve(representatives.size());
    for (size_t i = 0; i < representatives.size(); ++i) {
      tree_noise.push_back(
          static_cast<size_t>(std::round(std::max(gaussian(gen), 0.0))));
    }
  }
  return noise;
}

size_t CountRemainingNoise(const std::vector<std::vector<size_t>>& noise) {
  size_t remaining_noise = 0;
  for (const auto& tree_noise : noise) {
    for (size_t value : tree_noise) {
      remaining_noise += value;
    }
  }
  return remaining_noise;
}

bool IsDefaultLower(bst_float value) {
  return value == std::numeric_limits<bst_float>::lowest();
}

bool IsDefaultUpper(bst_float value) {
  return value == std::numeric_limits<bst_float>::max();
}

template <typename T>
typename std::enable_if<std::is_integral<T>::value && !std::is_same<T, bool>::value,
                        T>::type
FastObliviousChoose(bool pred, T t_val, T f_val) {
  const T mask = static_cast<T>(0) - static_cast<T>(pred);
  return (t_val & mask) | (f_val & ~mask);
}

bool FastObliviousChoose(bool pred, bool t_val, bool f_val) {
  return FastObliviousChoose<uint8_t>(
             pred, static_cast<uint8_t>(t_val), static_cast<uint8_t>(f_val)) !=
         0;
}

bst_float FastObliviousChoose(bool pred, bst_float t_val, bst_float f_val) {
  static_assert(sizeof(bst_float) == sizeof(uint32_t),
                "Fast bst_float select expects 32-bit float.");
  uint32_t t_bits;
  uint32_t f_bits;
  std::memcpy(&t_bits, &t_val, sizeof(t_bits));
  std::memcpy(&f_bits, &f_val, sizeof(f_bits));
  const uint32_t mask = 0U - static_cast<uint32_t>(pred);
  const uint32_t out_bits = (t_bits & mask) | (f_bits & ~mask);
  bst_float out;
  std::memcpy(&out, &out_bits, sizeof(out));
  return out;
}

std::vector<unsigned> CollectWorkingDomainFeatures(
    const std::vector<std::vector<NodeDomain>>& domains) {
  std::vector<unsigned> features;
  for (const auto& tree_domains : domains) {
    for (const auto& domain : tree_domains) {
      for (const auto& item : domain) {
        features.push_back(item.feature);
      }
    }
  }
  std::sort(features.begin(), features.end());
  features.erase(std::unique(features.begin(), features.end()),
                 features.end());
  return features;
}

void InitDenseWorkingDomain(std::vector<unsigned> features,
                            DenseWorkingDomain* working_domain) {
  working_domain->features = std::move(features);
  const size_t size = working_domain->features.size();
  working_domain->lower.resize(size);
  working_domain->upper.resize(size);
  working_domain->active.resize(size);
}

size_t DenseWorkingDomainFeatureIndex(const DenseWorkingDomain& working_domain,
                                      unsigned feature) {
  const auto begin = working_domain.features.begin();
  const auto end = working_domain.features.end();
  const auto iter = std::lower_bound(begin, end, feature);
  CHECK(iter != end);
  CHECK_EQ(*iter, feature);
  return static_cast<size_t>(iter - begin);
}

IndexedDomains BuildIndexedDomains(
    const std::vector<std::vector<NodeDomain>>& domains,
    const DenseWorkingDomain& working_domain) {
  IndexedDomains indexed_domains(domains.size());
  for (size_t tree_pos = 0; tree_pos < domains.size(); ++tree_pos) {
    indexed_domains[tree_pos].reserve(domains[tree_pos].size());
    for (const auto& domain : domains[tree_pos]) {
      IndexedNodeDomain indexed_domain;
      indexed_domain.reserve(domain.size());
      for (const auto& item : domain) {
        IndexedFeatureDomain indexed_item;
        indexed_item.index =
            DenseWorkingDomainFeatureIndex(working_domain, item.feature);
        indexed_item.feature = item.feature;
        indexed_item.lower = item.lower;
        indexed_item.upper = item.upper;
        indexed_domain.push_back(indexed_item);
      }
      indexed_domains[tree_pos].push_back(std::move(indexed_domain));
    }
  }
  return indexed_domains;
}

void ResetDenseWorkingDomain(DenseWorkingDomain* working_domain) {
  std::fill(working_domain->lower.begin(), working_domain->lower.end(),
            std::numeric_limits<bst_float>::lowest());
  std::fill(working_domain->upper.begin(), working_domain->upper.end(),
            std::numeric_limits<bst_float>::max());
  std::fill(working_domain->active.begin(), working_domain->active.end(), 0);
}

bool DenseWorkingDomainCanIntersect(const DenseWorkingDomain& working_domain,
                                    const IndexedNodeDomain& candidate) {
  bool can_intersect = true;
  for (const auto& candidate_item : candidate) {
    const size_t index = candidate_item.index;
    const bool found = working_domain.active[index] != 0;
    const bool candidate_valid = candidate_item.lower < candidate_item.upper;
    const bst_float lower =
        std::max(working_domain.lower[index], candidate_item.lower);
    const bst_float upper =
        std::min(working_domain.upper[index], candidate_item.upper);
    can_intersect =
        can_intersect && ((found && lower < upper) ||
                          (!found && candidate_valid));
  }
  return can_intersect;
}

void ApplyDenseWorkingDomainIntersect(DenseWorkingDomain* working_domain,
                                      const IndexedNodeDomain& candidate,
                                      bool enabled) {
  for (const auto& candidate_item : candidate) {
    const size_t index = candidate_item.index;
    const bool found = working_domain->active[index] != 0;
    const bst_float current_lower = working_domain->lower[index];
    const bst_float current_upper = working_domain->upper[index];
    const bst_float next_lower = FastObliviousChoose(
        found, std::max(current_lower, candidate_item.lower),
        candidate_item.lower);
    const bst_float next_upper = FastObliviousChoose(
        found, std::min(current_upper, candidate_item.upper),
        candidate_item.upper);
    working_domain->lower[index] =
        FastObliviousChoose(enabled, next_lower, current_lower);
    working_domain->upper[index] =
        FastObliviousChoose(enabled, next_upper, current_upper);
    working_domain->active[index] =
        FastObliviousChoose(enabled, static_cast<uint8_t>(1),
                            working_domain->active[index]);
  }
}

void TouchIndexedDomain(const IndexedNodeDomain& candidate) {
  if (candidate.empty()) {
    return;
  }

  constexpr std::uintptr_t kPageSize = 4096;
  const auto* data =
      reinterpret_cast<const unsigned char*>(candidate.data());
  const std::uintptr_t begin = reinterpret_cast<std::uintptr_t>(data);
  const std::uintptr_t end =
      begin + candidate.size() * sizeof(IndexedFeatureDomain);
  const std::uintptr_t first_page = begin & ~(kPageSize - 1);

  uint8_t sink = 0;
  for (std::uintptr_t page = first_page; page < end; page += kPageSize) {
    const std::uintptr_t address = std::max(page, begin);
    const volatile unsigned char* byte =
        reinterpret_cast<const volatile unsigned char*>(address);
    sink ^= *byte;
  }
  volatile uint8_t keep_alive = sink;
  static_cast<void>(keep_alive);
}

template <typename T>
void TouchVectorPages(const std::vector<T>& values) {
  if (values.empty()) {
    return;
  }

  constexpr std::uintptr_t kPageSize = 4096;
  const auto* data = reinterpret_cast<const unsigned char*>(values.data());
  const std::uintptr_t begin = reinterpret_cast<std::uintptr_t>(data);
  const std::uintptr_t end = begin + values.size() * sizeof(T);
  const std::uintptr_t first_page = begin & ~(kPageSize - 1);

  uint8_t sink = 0;
  for (std::uintptr_t page = first_page; page < end; page += kPageSize) {
    const std::uintptr_t address = std::max(page, begin);
    const volatile unsigned char* byte =
        reinterpret_cast<const volatile unsigned char*>(address);
    sink ^= *byte;
  }
  volatile uint8_t keep_alive = sink;
  static_cast<void>(keep_alive);
}

void TouchDenseWorkingDomainPages(const DenseWorkingDomain& working_domain) {
  TouchVectorPages(working_domain.active);
  TouchVectorPages(working_domain.lower);
  TouchVectorPages(working_domain.upper);
}

void ApplySelectedDenseWorkingDomainIntersect(
    DenseWorkingDomain* working_domain, const IndexedNodeDomain& candidate,
    bool enabled) {
  TouchDenseWorkingDomainPages(*working_domain);
  TouchIndexedDomain(candidate);
  ApplyDenseWorkingDomainIntersect(working_domain, candidate, enabled);
}

NoisePosition SelectFirstPositiveNoiseOblivious(
    const std::vector<std::vector<size_t>>& noise) {
  NoisePosition selected;
  for (size_t tree_pos = 0; tree_pos < noise.size(); ++tree_pos) {
    for (size_t representative_pos = 0;
         representative_pos < noise[tree_pos].size(); ++representative_pos) {
      const bool should_select =
          !selected.found && noise[tree_pos][representative_pos] > 0;
      selected.tree_pos =
          FastObliviousChoose(should_select, tree_pos, selected.tree_pos);
      selected.representative_pos =
          FastObliviousChoose(should_select, representative_pos,
                              selected.representative_pos);
      selected.found =
          FastObliviousChoose(should_select, true, selected.found);
    }
  }
  return selected;
}

void DecrementSelectedNoiseOblivious(
    std::vector<std::vector<size_t>>* noise, const NoisePosition& selected) {
  for (size_t tree_pos = 0; tree_pos < noise->size(); ++tree_pos) {
    for (size_t representative_pos = 0;
         representative_pos < (*noise)[tree_pos].size();
         ++representative_pos) {
      const bool should_decrement =
          selected.found && tree_pos == selected.tree_pos &&
          representative_pos == selected.representative_pos;
      const size_t value = (*noise)[tree_pos][representative_pos];
      (*noise)[tree_pos][representative_pos] =
          value - FastObliviousChoose<size_t>(should_decrement, 1, 0);
    }
  }
}

void DecrementSelectedNoiseInTreeOblivious(
    std::vector<std::vector<size_t>>* noise, size_t tree_pos,
    const NoisePosition& selected) {
  for (size_t representative_pos = 0;
       representative_pos < (*noise)[tree_pos].size(); ++representative_pos) {
    const bool should_decrement =
        selected.found && representative_pos == selected.representative_pos;
    const size_t value = (*noise)[tree_pos][representative_pos];
    (*noise)[tree_pos][representative_pos] =
        value - FastObliviousChoose<size_t>(should_decrement, 1, 0);
  }
}

void ApplySelectedDomainInTreeOblivious(
    DenseWorkingDomain* working_domain,
    const IndexedDomains& domains, size_t tree_pos,
    const NoisePosition& selected) {
  for (size_t representative_pos = 0;
       representative_pos < domains[tree_pos].size(); ++representative_pos) {
    const bool enabled =
        selected.found && representative_pos == selected.representative_pos;
    ApplySelectedDenseWorkingDomainIntersect(
        working_domain, domains[tree_pos][representative_pos], enabled);
  }
}

void InitializeWorkingDomainFromSelected(
    DenseWorkingDomain* working_domain,
    const IndexedDomains& domains,
    const NoisePosition& selected) {
  ResetDenseWorkingDomain(working_domain);
  for (size_t tree_pos = 0; tree_pos < domains.size(); ++tree_pos) {
    for (size_t representative_pos = 0;
         representative_pos < domains[tree_pos].size();
         ++representative_pos) {
      const bool enabled =
          selected.found && tree_pos == selected.tree_pos &&
          representative_pos == selected.representative_pos;
      ApplySelectedDenseWorkingDomainIntersect(
          working_domain, domains[tree_pos][representative_pos], enabled);
    }
  }
}

NoisePosition SelectCompatibleRepresentativeOblivious(
    const std::vector<std::vector<size_t>>& noise,
    const IndexedDomains& domains,
    const DenseWorkingDomain& working_domain, size_t tree_pos,
    const NoisePosition& seed) {
  NoisePosition selected;
  for (size_t representative_pos = 0;
       representative_pos < noise[tree_pos].size(); ++representative_pos) {
    const auto& candidate_domain = domains[tree_pos][representative_pos];
    const bool can_intersect =
        DenseWorkingDomainCanIntersect(working_domain, candidate_domain);
    const bool is_seed_tree = seed.found && tree_pos == seed.tree_pos;
    const bool should_select =
        !selected.found && !is_seed_tree &&
        noise[tree_pos][representative_pos] > 0 && can_intersect;
    selected.tree_pos =
        FastObliviousChoose(should_select, tree_pos, selected.tree_pos);
    selected.representative_pos =
        FastObliviousChoose(should_select, representative_pos,
                            selected.representative_pos);
    selected.found = FastObliviousChoose(should_select, true, selected.found);
  }
  return selected;
}

void BuildDummyEntries(const DenseWorkingDomain& domain,
                       std::vector<Entry>* entries) {
  TouchDenseWorkingDomainPages(domain);
  entries->clear();
  entries->reserve(domain.active.size());
  for (size_t feature = 0; feature < domain.active.size(); ++feature) {
    if (domain.active[feature] == 0) {
      continue;
    }

    const bool has_lower = !IsDefaultLower(domain.lower[feature]);
    const bool has_upper = !IsDefaultUpper(domain.upper[feature]);
    if (!has_lower && !has_upper) {
      continue;
    }

    bst_float value;
    if (has_lower && has_upper) {
      value = domain.lower[feature] +
              (domain.upper[feature] - domain.lower[feature]) / 2;
    } else if (has_lower) {
      value = std::nextafter(domain.lower[feature],
                             std::numeric_limits<bst_float>::max());
    } else {
      value = std::nextafter(domain.upper[feature],
                             std::numeric_limits<bst_float>::lowest());
    }
    entries->push_back(Entry{domain.features[feature], value});
  }
}

void PadDummyEntries(size_t fixed_row_size, std::vector<Entry>* entries) {
  CHECK_LE(entries->size(), fixed_row_size);
  const Entry padding_entry{std::numeric_limits<bst_feature_t>::max(), 0.0f};
  while (entries->size() < fixed_row_size) {
    entries->push_back(padding_entry);
  }
}

DummySamples BuildDummySamplesFromNoiseAndDomains(
    std::vector<std::vector<size_t>>* noise,
    const std::vector<std::vector<NodeDomain>>& domains,
    size_t fixed_row_size) {
  CHECK(noise != nullptr);
  CHECK_EQ(noise->size(), domains.size());
  for (size_t tree_pos = 0; tree_pos < noise->size(); ++tree_pos) {
    CHECK_EQ((*noise)[tree_pos].size(), domains[tree_pos].size());
  }

  DummySamples dummy_samples;
  dummy_samples.max_entries = fixed_row_size;
  size_t remaining_noise = CountRemainingNoise(*noise);
  std::vector<Entry> entries;
  DenseWorkingDomain dummy_domain;
  InitDenseWorkingDomain(CollectWorkingDomainFeatures(domains),
                         &dummy_domain);
  const IndexedDomains indexed_domains =
      BuildIndexedDomains(domains, dummy_domain);
  while (remaining_noise > 0) {
    const NoisePosition seed = SelectFirstPositiveNoiseOblivious(*noise);
    CHECK(seed.found);
    InitializeWorkingDomainFromSelected(&dummy_domain, indexed_domains, seed);
    DecrementSelectedNoiseOblivious(noise, seed);
    remaining_noise--;

    for (size_t tree_pos = 0; tree_pos < noise->size(); ++tree_pos) {
      const NoisePosition selected =
          SelectCompatibleRepresentativeOblivious(*noise, indexed_domains,
                                                  dummy_domain, tree_pos,
                                                  seed);
      ApplySelectedDomainInTreeOblivious(&dummy_domain, indexed_domains,
                                         tree_pos, selected);
      DecrementSelectedNoiseInTreeOblivious(noise, tree_pos, selected);
      remaining_noise -= FastObliviousChoose<size_t>(selected.found, 1, 0);
    }

    BuildDummyEntries(dummy_domain, &entries);
    PadDummyEntries(fixed_row_size, &entries);
    SparsePage::Inst inst{entries.data(), entries.size()};
    dummy_samples.page.Push(inst);
  }
  return dummy_samples;
}

void ConfigureDoxieMemory(gbm::GBTreeModel const& model, int32_t tree_begin,
                          int32_t tree_end, bool enabled) {
  for (size_t i = tree_begin; i < static_cast<size_t>(tree_end); ++i) {
    if (enabled) {
      model.trees[i]->EnableDoxieMemory();
    } else {
      model.trees[i]->DisableDoxieMemory();
    }
  }
}

}  // namespace

void PredictDMatrix(DMatrix* p_fmat, std::vector<bst_float>* out_preds,
                    gbm::GBTreeModel const& model, int32_t tree_begin,
                    int32_t tree_end, InferenceContext* context) {
  CHECK(context != nullptr);
  CHECK(context->metrics != nullptr);
  CHECK(context->thread_temp != nullptr);
  CHECK(context->lock != nullptr);

  PredictionMetrics& metrics = *context->metrics;
  ConfigureDoxieMemory(model, tree_begin, tree_end,
                       metrics.doxie_memory_alignment);

  common::Timer total_timer;
  std::lock_guard<std::mutex> guard(*context->lock);
  const int threads = omp_get_max_threads();
  InitThreadTemp(threads, model.learner_model_param->num_feature,
                 context->thread_temp);

  for (auto& batch : p_fmat->GetBatches<SparsePage>()) {
    CHECK_EQ(out_preds->size(),
             p_fmat->Info().num_row_ *
                 model.learner_model_param->num_output_group);
    size_t constexpr kUnroll = 8;

    double epsilon = metrics.epsilon;
    double delta = metrics.delta;
    int32_t const num_group = model.learner_model_param->num_output_group;

    auto representatives =
        CollectLeafPageRepresentatives(model, tree_begin, tree_end);
    auto domains = CollectRepresentativeDomains(model, tree_begin, tree_end,
                                                representatives);
    auto noise = SampleRepresentativeNoise(tree_begin, tree_end,
                                           representatives, epsilon, delta, 1);
    const size_t fixed_dummy_row_size =
        std::max(batch.MaxNumberOfEntries(),
                 CollectWorkingDomainFeatures(domains).size());
    DummySamples dummy_samples =
        BuildDummySamplesFromNoiseAndDomains(&noise, domains,
                                             fixed_dummy_row_size);
    ::DoxieInference doxie_inference(epsilon, delta, 1,
                                     metrics.shuffle_method);
    doxie_inference.Preprocess(batch, dummy_samples.page,
                               dummy_samples.max_entries, &metrics,
                               num_group);

    common::Timer predict_no_timer;
    if (metrics.doxie_memory_alignment && metrics.doxie_blocked_kernel) {
      PredictBatchKernelBlocked(
          SparsePageView<kUnroll>{&doxie_inference.shuffle_page},
          &(doxie_inference.shuffle_preds), model, tree_begin, tree_end,
          context->monitor);
    } else {
      PredictBatchKernel(SparsePageView<kUnroll>{&doxie_inference.shuffle_page},
                         &(doxie_inference.shuffle_preds), model, tree_begin,
                         tree_end, context->thread_temp, context->monitor);
    }
    predict_no_timer.Stop();
    metrics.predict_no_seconds += predict_no_timer.ElapsedSeconds();
    doxie_inference.PostProcess(out_preds, &metrics);
  }

  total_timer.Stop();
  metrics.has_do_metrics = true;
  metrics.predict_dmatrix_do_seconds = total_timer.ElapsedSeconds();
}

}  // namespace doxie
}  // namespace xgboost
