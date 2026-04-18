/*!
 * Copyright by Contributors 2017-2020
 * Modifications Copyright 2020-22 by Secure XGBoost Contributors
 */
#include "enclave/doxie_inference.h"

#include <dmlc/omp.h>

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <limits>
#include <random>
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

  NoisePosition() : tree_pos(0), representative_pos(0) {}
  NoisePosition(size_t tree, size_t representative)
      : tree_pos(tree), representative_pos(representative) {}
};

struct DummySamples {
  SparsePage page;
  size_t max_entries{0};
};

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

NoisePosition FindSeed(const std::vector<std::vector<size_t>>& noise,
                       NoisePosition* cursor) {
  CHECK(cursor != nullptr);
  for (size_t tree_pos = cursor->tree_pos; tree_pos < noise.size();
       ++tree_pos) {
    const size_t representative_begin =
        tree_pos == cursor->tree_pos ? cursor->representative_pos : 0;
    for (size_t representative_pos = representative_begin;
         representative_pos < noise[tree_pos].size(); ++representative_pos) {
      if (noise[tree_pos][representative_pos] > 0) {
        *cursor = NoisePosition(tree_pos, representative_pos);
        return NoisePosition(tree_pos, representative_pos);
      }
    }
  }
  LOG(FATAL) << "FindSeed requires at least one positive noise value.";
  return NoisePosition();
}

bool CanIntersectDomain(const NodeDomain& current,
                        const NodeDomain& candidate) {
  for (const auto& candidate_item : candidate) {
    bool found = false;
    for (const auto& current_item : current) {
      if (current_item.feature == candidate_item.feature) {
        const bst_float lower =
            std::max(current_item.lower, candidate_item.lower);
        const bst_float upper =
            std::min(current_item.upper, candidate_item.upper);
        if (!(lower < upper)) {
          return false;
        }
        found = true;
        break;
      }
    }
    if (!found) {
      if (!(candidate_item.lower < candidate_item.upper)) {
        return false;
      }
    }
  }

  return true;
}

void ApplyIntersectDomain(NodeDomain* current, const NodeDomain& candidate) {
  for (const auto& candidate_item : candidate) {
    bool found = false;
    for (auto& current_item : *current) {
      if (current_item.feature == candidate_item.feature) {
        current_item.lower =
            std::max(current_item.lower, candidate_item.lower);
        current_item.upper =
            std::min(current_item.upper, candidate_item.upper);
        found = true;
        break;
      }
    }
    if (!found) {
      current->push_back(candidate_item);
    }
  }
}

bool IsDefaultLower(bst_float value) {
  return value == std::numeric_limits<bst_float>::lowest();
}

bool IsDefaultUpper(bst_float value) {
  return value == std::numeric_limits<bst_float>::max();
}

void BuildDummyEntries(const NodeDomain& domain,
                       std::vector<Entry>* entries) {
  entries->clear();
  entries->reserve(domain.size());
  for (const auto& item : domain) {
    const bool has_lower = !IsDefaultLower(item.lower);
    const bool has_upper = !IsDefaultUpper(item.upper);
    if (!has_lower && !has_upper) {
      continue;
    }

    bst_float value;
    if (has_lower && has_upper) {
      value = item.lower + (item.upper - item.lower) / 2;
    } else if (has_lower) {
      value = std::nextafter(item.lower,
                             std::numeric_limits<bst_float>::max());
    } else {
      value = std::nextafter(item.upper,
                             std::numeric_limits<bst_float>::lowest());
    }
    entries->push_back(Entry{item.feature, value});
  }
}

DummySamples BuildDummySamplesFromNoiseAndDomains(
    std::vector<std::vector<size_t>>* noise,
    const std::vector<std::vector<NodeDomain>>& domains) {
  CHECK(noise != nullptr);
  CHECK_EQ(noise->size(), domains.size());
  for (size_t tree_pos = 0; tree_pos < noise->size(); ++tree_pos) {
    CHECK_EQ((*noise)[tree_pos].size(), domains[tree_pos].size());
  }

  DummySamples dummy_samples;
  size_t remaining_noise = CountRemainingNoise(*noise);
  std::vector<Entry> entries;
  NodeDomain dummy_domain;
  dummy_domain.reserve(32);
  NoisePosition seed_cursor;
  std::vector<size_t> first_positive_rep(noise->size(), 0);
  while (remaining_noise > 0) {
    const NoisePosition seed = FindSeed(*noise, &seed_cursor);
    dummy_domain = domains[seed.tree_pos][seed.representative_pos];
    CHECK_GT((*noise)[seed.tree_pos][seed.representative_pos], 0U);
    (*noise)[seed.tree_pos][seed.representative_pos]--;
    remaining_noise--;

    for (size_t tree_pos = 0; tree_pos < noise->size(); ++tree_pos) {
      if (tree_pos == seed.tree_pos) {
        continue;
      }

      while (first_positive_rep[tree_pos] < (*noise)[tree_pos].size() &&
             (*noise)[tree_pos][first_positive_rep[tree_pos]] == 0) {
        first_positive_rep[tree_pos]++;
      }

      for (size_t representative_pos = first_positive_rep[tree_pos];
           representative_pos < (*noise)[tree_pos].size();
           ++representative_pos) {
        if ((*noise)[tree_pos][representative_pos] == 0) {
          continue;
        }
        const auto& candidate_domain = domains[tree_pos][representative_pos];
        if (CanIntersectDomain(dummy_domain, candidate_domain)) {
          ApplyIntersectDomain(&dummy_domain, candidate_domain);
          (*noise)[tree_pos][representative_pos]--;
          remaining_noise--;
          break;
        }
      }
    }

    BuildDummyEntries(dummy_domain, &entries);
    SparsePage::Inst inst{entries.data(), entries.size()};
    dummy_samples.page.Push(inst);
    dummy_samples.max_entries =
        std::max(dummy_samples.max_entries, entries.size());
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
    DummySamples dummy_samples =
        BuildDummySamplesFromNoiseAndDomains(&noise, domains);
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
