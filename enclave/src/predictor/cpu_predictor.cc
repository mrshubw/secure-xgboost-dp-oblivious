/*!
 * Copyright by Contributors 2017-2020
 * Modifications Copyright 2020-22 by Secure XGBoost Contributors
 */
#include <dmlc/any.h>
#include <dmlc/omp.h>

#include <cstddef>
#include <limits>
#include <mutex>
#include <thread>

#include "../common/math.h"
#include "../common/timer.h"
#include "../data/adapter.h"
#include "../gbm/gbtree_model.h"
#include "xgboost/base.h"
#include "xgboost/data.h"
#include "xgboost/host_device_vector.h"
#include "xgboost/logging.h"
#include "xgboost/predictor.h"
#include "xgboost/tree_model.h"
#include "xgboost/tree_updater.h"

#ifdef __ENCLAVE_OBLIVIOUS__
#include "enclave/dpobl_operator.h"
#include "../common/quantile.h"
#endif

namespace xgboost {
namespace predictor {

DMLC_REGISTRY_FILE_TAG(cpu_predictor);

bst_float PredValue(const SparsePage::Inst& inst,
                    const std::vector<std::unique_ptr<RegTree>>& trees,
                    const std::vector<int>& tree_info, int bst_group,
                    RegTree::FVec* p_feats, unsigned tree_begin,
                    unsigned tree_end, common::Monitor* monitor_ = nullptr) {
  // if (monitor_ != nullptr) monitor_->Start(__func__);
  bst_float psum = 0.0f;
  p_feats->Fill(inst);
  for (size_t i = tree_begin; i < tree_end; ++i) {
    if (tree_info[i] == bst_group) {
#ifdef __ENCLAVE_OBLIVIOUS__
#ifdef __ENCLAVE_DPOBLIVIOUS__
      // // 最开始的dp方案，面向单个样本的，使用stash进行缓存的方案，隐私损失太大
      // // std::cout<<"Pred Value Tree "<<i<<std::endl;
      // trees[i]->InitStash();
      // // std::cout<<"end position!!!!! "<<std::endl;
      // if (monitor_ != nullptr) monitor_->Start("DPOGetLeafValue");
      // auto leaf_value = trees[i]->DPOGetLeafValue(*p_feats);
      // psum += leaf_value;
      // if (monitor_ != nullptr) monitor_->Stop("DPOGetLeafValue");

      // // int tid = trees[i]->GetLeafIndex(*p_feats);
      // // auto leaf_value2 = (*trees[i])[tid].LeafValue();
      // // CHECK_EQ(leaf_value, leaf_value2) << leaf_value << ", " << leaf_value2;

      // 其他dp方案对此不进行改动
      // if (monitor_ != nullptr) monitor_->Start("GetLeafValue");
      int tid = trees[i]->GetLeafIndex(*p_feats);
      psum += (*trees[i])[tid].LeafValue();
      // if (monitor_ != nullptr) monitor_->Stop("GetLeafValue");
#else
      if (common::ObliviousEnabled()) {
        // if (monitor_ != nullptr) monitor_->Start("OGetLeafValue");
        auto leaf_value = trees[i]->OGetLeafValue(*p_feats);
        // auto leaf_value = trees[i]->OGetLeafValueCache(*p_feats);
        // int tid = trees[i]->GetLeafIndex(*p_feats);
        // auto leaf_value = (*trees[i])[tid].LeafValue();
        // if (monitor_ != nullptr) monitor_->Stop("OGetLeafValue");
        // if (common::ObliviousDebugCheckEnabled()) {
        //   int tid = trees[i]->GetLeafIndex(*p_feats);
        //   auto leaf_value2 = (*trees[i])[tid].LeafValue();
        //   CHECK_EQ(leaf_value, leaf_value2)
        //       << leaf_value << ", " << leaf_value2;
        // }
        psum += leaf_value;
      } else {
        int tid = trees[i]->GetLeafIndex(*p_feats);
        psum += (*trees[i])[tid].LeafValue();
      }
#endif
#else
      // if (monitor_ != nullptr) monitor_->Start("GetLeafValue");
      int tid = trees[i]->GetLeafIndex(*p_feats);
      psum += (*trees[i])[tid].LeafValue();
      // if (monitor_ != nullptr) monitor_->Stop("GetLeafValue");
#endif
    }
  }
  p_feats->Drop(inst);

  // if (monitor_ != nullptr) monitor_->Stop(__func__);
  return psum;
}

template <size_t kUnrollLen = 8>
struct SparsePageView {
  SparsePage const* page;
  bst_row_t base_rowid;
  static size_t constexpr kUnroll = kUnrollLen;

  explicit SparsePageView(SparsePage const* p)
      : page{p}, base_rowid{page->base_rowid} {
    // Pull to host before entering omp block, as this is not thread safe.
    page->data.HostVector();
    page->offset.HostVector();
  }
  SparsePage::Inst operator[](size_t i) { return (*page)[i]; }
  size_t Size() const { return page->Size(); }
};

template <typename Adapter, size_t kUnrollLen = 8>
class AdapterView {
  Adapter* adapter_;
  float missing_;
  common::Span<Entry> workspace_;
  std::vector<size_t> current_unroll_;

 public:
  static size_t constexpr kUnroll = kUnrollLen;

 public:
  explicit AdapterView(Adapter* adapter, float missing,
                       common::Span<Entry> workplace)
      : adapter_{adapter},
        missing_{missing},
        workspace_{workplace},
        current_unroll_(omp_get_max_threads() > 0 ? omp_get_max_threads() : 1,
                        0) {}
  SparsePage::Inst operator[](size_t i) {
    bst_feature_t columns = adapter_->NumColumns();
    auto const& batch = adapter_->Value();
    auto row = batch.GetLine(i);
    auto t = omp_get_thread_num();
    auto const beg = (columns * kUnroll * t) + (current_unroll_[t] * columns);
    size_t non_missing{beg};
    for (size_t c = 0; c < row.Size(); ++c) {
      auto e = row.GetElement(c);
      if (missing_ != e.value && !common::CheckNAN(e.value)) {
        workspace_[non_missing] =
            Entry{static_cast<bst_feature_t>(e.column_idx), e.value};
        ++non_missing;
      }
    }
    auto ret = workspace_.subspan(beg, non_missing - beg);
    current_unroll_[t]++;
    if (current_unroll_[t] == kUnroll) {
      current_unroll_[t] = 0;
    }
    return ret;
  }

  size_t Size() const { return adapter_->NumRows(); }

  bst_row_t const static base_rowid = 0;  // NOLINT
};

template <typename DataView>
void PredictBatchKernel(DataView batch, std::vector<bst_float>* out_preds,
                        gbm::GBTreeModel const& model, int32_t tree_begin,
                        int32_t tree_end,
                        std::vector<RegTree::FVec>* p_thread_temp,
                        common::Monitor* monitor_ = nullptr) {
  if (monitor_ != nullptr) monitor_->Start(__func__);
  auto& thread_temp = *p_thread_temp;
  int32_t const num_group = model.learner_model_param->num_output_group;

  std::vector<bst_float>& preds = *out_preds;
  CHECK_EQ(model.param.size_leaf_vector, 0)
      << "size_leaf_vector is enforced to 0 so far";
  // parallel over local batch
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
                                     &feats, tree_begin, tree_end, monitor_);
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
                                 &feats, tree_begin, tree_end, monitor_);
    }
  }
  if (monitor_ != nullptr) monitor_->Stop(__func__);
}

class CPUPredictor : public Predictor {
 protected:
  // init thread buffers
  static void InitThreadTemp(int nthread, int num_feature,
                             std::vector<RegTree::FVec>* out) {
    int prev_thread_temp_size = out->size();
    if (prev_thread_temp_size < nthread) {
      out->resize(nthread, RegTree::FVec());
      for (int i = prev_thread_temp_size; i < nthread; ++i) {
        (*out)[i].Init(num_feature);
      }
    }
  }

  /**
   * 初始推断方案
   * 将输入数据分批处理
  */
  void PredictDMatrix(DMatrix* p_fmat, std::vector<bst_float>* out_preds,
                      gbm::GBTreeModel const& model, int32_t tree_begin,
                      int32_t tree_end) {
    std::lock_guard<std::mutex> guard(lock_);
    monitor_.Start(__func__);
    const int threads = omp_get_max_threads();
    InitThreadTemp(threads, model.learner_model_param->num_feature,
                   &this->thread_temp_);
    for (auto const& batch : p_fmat->GetBatches<SparsePage>()) {
      CHECK_EQ(out_preds->size(),
               p_fmat->Info().num_row_ *
                   model.learner_model_param->num_output_group);
      size_t constexpr kUnroll = 8;
      PredictBatchKernel(SparsePageView<kUnroll>{&batch}, out_preds, model,
                         tree_begin, tree_end, &thread_temp_, &monitor_);
    }
    std::cout << "num of tree nodes: "<< model.trees[0]->GetNodes().size() << std::endl;
    monitor_.Stop(__func__);
  }

#ifdef __ENCLAVE_DPOBLIVIOUS__
  /**
   * 逐决策树地进行推断
  */
  void PredictDMatrixByTrees(DMatrix* p_fmat, std::vector<bst_float>* out_preds,
                            gbm::GBTreeModel const& model, int32_t tree_begin,
                            int32_t tree_end) {
    monitor_.Start(__func__);
    const int threads = omp_get_max_threads();
    InitThreadTemp(threads, model.learner_model_param->num_feature,
                   &this->thread_temp_);
    int32_t const num_group = model.learner_model_param->num_output_group;
    CHECK_EQ(out_preds->size(), p_fmat->Info().num_row_ * model.learner_model_param->num_output_group);
    for (int gid = 0; gid < num_group; ++gid) {
      for (size_t i = tree_begin; i < tree_end; ++i) {
        if (model.tree_info[i] == gid) {
          model.trees[i]->DPOPredictByHist(p_fmat, out_preds, gid, num_group, thread_temp_[0]);
        }
      }
    }

    monitor_.Stop(__func__);
  }

  void PredictOneTree(xgboost::SparsePage& in_page, std::vector<bst_float>& out_preds, RegTree& tree, RegTree::FVec& feat){
    auto nsize = in_page.Size();
    for (int i = 0; i < nsize; i++) {
      feat.Fill(in_page[i]);
      
      int tid = tree.GetLeafIndex(feat);
      out_preds[i] += tree[tid].LeafValue();

      feat.Drop(in_page[i]);
    }
  }
  // static void printNumbers(int start, int end) {
  //   for (int i = start; i <= end; ++i) {
  //       std::cout << i << " ";
  //   }
  //   std::cout << std::endl;
  // }

  // static void printLetters(char start, char end) {
  //     for (char c = start; c <= end; ++c) {
  //         std::cout << c << " ";
  //     }
  //     std::cout << std::endl;
  // }

  /**
   * 将输入数据添加噪声后混洗，随后正常推断
  */
  void PredictDMatrixDO(DMatrix* p_fmat, std::vector<bst_float>* out_preds,
                            gbm::GBTreeModel const& model, int32_t tree_begin,
                            int32_t tree_end){
    common::Monitor monitor1;
    monitor1.Init("DO");
    monitor1.StartForce(__func__);
    //   #ifdef _OPENMP
    // std::cout << "OpenMP is enabled. Version: " << _OPENMP << std::endl;
    // #else
    // std::cout << "OpenMP is not enabled." << std::endl;
    // #endif
    // std::thread t1(printNumbers, 1, 10); // Thread to print numbers
    // std::thread t2(printLetters, 'a', 'z'); // Thread to print letters

    // // Wait for threads to finish
    // t1.join();
    // t2.join();

    std::lock_guard<std::mutex> guard(lock_);
    const int threads = omp_get_max_threads();
    InitThreadTemp(threads, model.learner_model_param->num_feature,
                   &this->thread_temp_);
    for (auto & batch : p_fmat->GetBatches<SparsePage>()) {
      CHECK_EQ(out_preds->size(),
               p_fmat->Info().num_row_ *
                   model.learner_model_param->num_output_group);
      size_t constexpr kUnroll = 8;

      double epsilon = 0.1;
      double delta = 0.00001;
      logStr("/home/hgtc/secure-xgboost-dp-oblivious/do-enhanced/data/time.log", "epsilon: ", epsilon);
      bool nosie_by_trees = false;
      if (nosie_by_trees)
      {
        int32_t const num_group = model.learner_model_param->num_output_group;
        std::cout<<"num_group: "<<num_group<<std::endl;
        for (int gid = 0; gid < num_group; ++gid) {
          // #pragma omp parallel for num_threads(5)
          for (size_t i = tree_begin; i < tree_end; ++i) {
            if (model.tree_info[i] == gid) {
              // std::cout << "Thread " << omp_get_thread_num() << " processing element " << i << std::endl;
              DOoperator do_operator(epsilon/(tree_end-tree_begin), delta/(tree_end-tree_begin), 1);
              do_operator.Preprocess(batch, model.trees[0]->GetNodes().size(), &monitor1);

              monitor1.StartForce("PredictNO");
              PredictOneTree(do_operator.shuffle_page, do_operator.shuffle_preds, *model.trees[i], thread_temp_[0]);
              monitor1.StopForce("PredictNO");

              do_operator.PostProcessAdd(out_preds, &monitor1);
            }
          }
        }
      }else
      {
        int32_t const num_group = model.learner_model_param->num_output_group;
        std::cout<<"num_group: "<<num_group<<std::endl;
        // std::cout<<"batch size: "<<batch.Size()<<std::endl;
        // std::cout<<"batch data size: "<<batch.data.Size() * sizeof(xgboost::Entry)<<std::endl;
        
        DOoperator do_operator(epsilon, delta, 1);
        do_operator.Preprocess(batch, model.trees[0]->GetNodes().size(), &monitor1, (tree_end-tree_begin), num_group);
        
        monitor1.StartForce("PredictNO");
        PredictBatchKernel(SparsePageView<kUnroll>{&do_operator.shuffle_page}, &(do_operator.shuffle_preds), model,
                          tree_begin, tree_end, &thread_temp_, &monitor_);
        monitor1.StopForce("PredictNO");
        std::cout<<"shuffle_preds: "<<do_operator.shuffle_preds.size()<<std::endl;
        do_operator.PostProcess(out_preds, &monitor1);
        std::cout<<"out_preds: "<<out_preds->size()<<std::endl;
        std::cout<<"shuffle_preds: "<<do_operator.shuffle_preds.size()<<std::endl;
      }
      
    }

    monitor1.StopForce(__func__);
    monitor1.PrintForce("/home/hgtc/secure-xgboost-dp-oblivious/do-enhanced/data/time.log");
  }
#endif

  void InitOutPredictions(const MetaInfo& info,
                          HostDeviceVector<bst_float>* out_preds,
                          const gbm::GBTreeModel& model) const {
    CHECK_NE(model.learner_model_param->num_output_group, 0);
    size_t n = model.learner_model_param->num_output_group * info.num_row_;
    const auto& base_margin = info.base_margin_.HostVector();
    out_preds->Resize(n);
    std::vector<bst_float>& out_preds_h = out_preds->HostVector();
    if (base_margin.size() == n) {
      CHECK_EQ(out_preds->Size(), n);
      std::copy(base_margin.begin(), base_margin.end(), out_preds_h.begin());
    } else {
      if (!base_margin.empty()) {
        std::ostringstream oss;
        oss << "Ignoring the base margin, since it has incorrect length. "
            << "The base margin must be an array of length ";
        if (model.learner_model_param->num_output_group > 1) {
          oss << "[num_class] * [number of data points], i.e. "
              << model.learner_model_param->num_output_group << " * "
              << info.num_row_ << " = " << n << ". ";
        } else {
          oss << "[number of data points], i.e. " << info.num_row_ << ". ";
        }
        oss << "Instead, all data points will use "
            << "base_score = " << model.learner_model_param->base_score;
        LOG(WARNING) << oss.str();
      }
      std::fill(out_preds_h.begin(), out_preds_h.end(),
                model.learner_model_param->base_score);
    }
  }

 public:
  explicit CPUPredictor(GenericParameter const* generic_param)
      : Predictor::Predictor{generic_param} {}
  // ntree_limit is a very problematic parameter, as it's ambiguous in the
  // context of multi-output and forest.  Same problem exists for tree_begin
  void PredictBatch(DMatrix* dmat, PredictionCacheEntry* predts,
                    const gbm::GBTreeModel& model, int tree_begin,
                    uint32_t const ntree_limit = 0) override {
    xgboost::common::Timer timer;
    timer.Start();
    monitor_.Init("CPUPredictor");
    monitor_.Start(__func__);
    // tree_begin is not used, right now we just enforce it to be 0.
    CHECK_EQ(tree_begin, 0);
    auto* out_preds = &predts->predictions;
    CHECK_GE(predts->version, tree_begin);
    if (out_preds->Size() == 0 && dmat->Info().num_row_ != 0) {
      CHECK_EQ(predts->version, 0);
    }
    if (predts->version == 0) {
      // out_preds->Size() can be non-zero as it's initialized here before any
      // tree is built at the 0^th iterator.
      this->InitOutPredictions(dmat->Info(), out_preds, model);
    }

    uint32_t const output_groups = model.learner_model_param->num_output_group;
    CHECK_NE(output_groups, 0);
    // Right now we just assume ntree_limit provided by users means number of
    // tree layers in the context of multi-output model
    uint32_t real_ntree_limit = ntree_limit * output_groups;
    if (real_ntree_limit == 0 || real_ntree_limit > model.trees.size()) {
      real_ntree_limit = static_cast<uint32_t>(model.trees.size());
    }

    uint32_t const end_version =
        (tree_begin + real_ntree_limit) / output_groups;
    // When users have provided ntree_limit, end_version can be lesser, cache is
    // violated
    if (predts->version > end_version) {
      CHECK_NE(ntree_limit, 0);
      this->InitOutPredictions(dmat->Info(), out_preds, model);
      predts->version = 0;
    }
    uint32_t const beg_version = predts->version;
    CHECK_LE(beg_version, end_version);

    // be called when booster.predict in python
    if (beg_version < end_version) {
#ifdef __ENCLAVE_DPOBLIVIOUS__
      this->PredictDMatrixDO(dmat, &out_preds->HostVector(), model,
                           beg_version * output_groups,
                           end_version * output_groups);

      // 检测DO算法
      // std::vector<bst_float> out_check;
      // out_check.resize(out_preds->HostVector().size(), 0);
      // this->PredictDMatrix(dmat, &out_check, model,
      //                      beg_version * output_groups,
      //                      end_version * output_groups);

      // for (size_t i = 0; i < out_check.size(); i++)
      // {
      //   CHECK_EQ(out_preds->HostVector()[i], out_check[i])<<out_preds->HostVector()[i]<<","<< out_check[i];
      // }
      
#else
      this->PredictDMatrix(dmat, &out_preds->HostVector(), model,
                           beg_version * output_groups,
                           end_version * output_groups);
#endif
    }

    // delta means {size of forest} * {number of newly accumulated layers}
    uint32_t delta = end_version - beg_version;
    CHECK_LE(delta, model.trees.size());
    predts->Update(delta);

    CHECK(out_preds->Size() == output_groups * dmat->Info().num_row_ ||
          out_preds->Size() == dmat->Info().num_row_);

    monitor_.Stop(__func__);
    monitor_.Print();
    // std::cout<<"PredictBatch cost: "<<monitor_.GetCost(__func__).second<<std::endl;
    timer.Stop();
    timer.PrintElapsed("PredictBatch: ", "/home/hgtc/secure-xgboost-dp-oblivious/do-enhanced/data/time.log");
  }

  template <typename Adapter>
  void DispatchedInplacePredict(dmlc::any const& x,
                                const gbm::GBTreeModel& model, float missing,
                                PredictionCacheEntry* out_preds,
                                uint32_t tree_begin, uint32_t tree_end) const {
    auto threads = omp_get_max_threads();
    auto m = dmlc::get<std::shared_ptr<Adapter>>(x);
    CHECK_EQ(m->NumColumns(), model.learner_model_param->num_feature)
        << "Number of columns in data must equal to trained model.";
    MetaInfo info;
    info.num_col_ = m->NumColumns();
    info.num_row_ = m->NumRows();
    this->InitOutPredictions(info, &(out_preds->predictions), model);
    std::vector<Entry> workspace(info.num_col_ * 8 * threads);
    auto& predictions = out_preds->predictions.HostVector();
    std::vector<RegTree::FVec> thread_temp;
    InitThreadTemp(threads, model.learner_model_param->num_feature,
                   &thread_temp);
    size_t constexpr kUnroll = 8;
    PredictBatchKernel(AdapterView<Adapter, kUnroll>(
                           m.get(), missing, common::Span<Entry>{workspace}),
                       &predictions, model, tree_begin, tree_end, &thread_temp);
  }

  void InplacePredict(dmlc::any const& x, const gbm::GBTreeModel& model,
                      float missing, PredictionCacheEntry* out_preds,
                      uint32_t tree_begin, unsigned tree_end) const override {
    if (x.type() == typeid(std::shared_ptr<data::DenseAdapter>)) {
      this->DispatchedInplacePredict<data::DenseAdapter>(
          x, model, missing, out_preds, tree_begin, tree_end);
    } else if (x.type() == typeid(std::shared_ptr<data::CSRAdapter>)) {
      this->DispatchedInplacePredict<data::CSRAdapter>(
          x, model, missing, out_preds, tree_begin, tree_end);
    } else {
      LOG(FATAL) << "Data type is not supported by CPU Predictor.";
    }
  }

  void PredictInstance(const SparsePage::Inst& inst,
                       std::vector<bst_float>* out_preds,
                       const gbm::GBTreeModel& model,
                       unsigned ntree_limit) override {
    if (thread_temp_.size() == 0) {
      thread_temp_.resize(1, RegTree::FVec());
      thread_temp_[0].Init(model.learner_model_param->num_feature);
    }
    ntree_limit *= model.learner_model_param->num_output_group;
    if (ntree_limit == 0 || ntree_limit > model.trees.size()) {
      ntree_limit = static_cast<unsigned>(model.trees.size());
    }
    out_preds->resize(model.learner_model_param->num_output_group *
                      (model.param.size_leaf_vector + 1));
    // loop over output groups
    for (uint32_t gid = 0; gid < model.learner_model_param->num_output_group;
         ++gid) {
      (*out_preds)[gid] = PredValue(inst, model.trees, model.tree_info, gid,
                                    &thread_temp_[0], 0, ntree_limit) +
                          model.learner_model_param->base_score;
    }
  }

  void PredictLeaf(DMatrix* p_fmat, std::vector<bst_float>* out_preds,
                   const gbm::GBTreeModel& model,
                   unsigned ntree_limit) override {
    const int nthread = omp_get_max_threads();
    InitThreadTemp(nthread, model.learner_model_param->num_feature,
                   &this->thread_temp_);
    const MetaInfo& info = p_fmat->Info();
    // number of valid trees
    ntree_limit *= model.learner_model_param->num_output_group;
    if (ntree_limit == 0 || ntree_limit > model.trees.size()) {
      ntree_limit = static_cast<unsigned>(model.trees.size());
    }
    std::vector<bst_float>& preds = *out_preds;
    preds.resize(info.num_row_ * ntree_limit);
    // start collecting the prediction
    for (const auto& batch : p_fmat->GetBatches<SparsePage>()) {
      // parallel over local batch
      const auto nsize = static_cast<bst_omp_uint>(batch.Size());
#pragma omp parallel for schedule(static)
      for (bst_omp_uint i = 0; i < nsize; ++i) {
        const int tid = omp_get_thread_num();
        auto ridx = static_cast<size_t>(batch.base_rowid + i);
        RegTree::FVec& feats = thread_temp_[tid];
        feats.Fill(batch[i]);
        for (unsigned j = 0; j < ntree_limit; ++j) {
          int tid = model.trees[j]->GetLeafIndex(feats);
          preds[ridx * ntree_limit + j] = static_cast<bst_float>(tid);
        }
        feats.Drop(batch[i]);
      }
    }
  }

  void PredictContribution(DMatrix* p_fmat,
                           std::vector<bst_float>* out_contribs,
                           const gbm::GBTreeModel& model, uint32_t ntree_limit,
                           std::vector<bst_float>* tree_weights,
                           bool approximate, int condition,
                           unsigned condition_feature) override {
    const int nthread = omp_get_max_threads();
    InitThreadTemp(nthread, model.learner_model_param->num_feature,
                   &this->thread_temp_);
    const MetaInfo& info = p_fmat->Info();
    // number of valid trees
    ntree_limit *= model.learner_model_param->num_output_group;
    if (ntree_limit == 0 || ntree_limit > model.trees.size()) {
      ntree_limit = static_cast<unsigned>(model.trees.size());
    }
    const int ngroup = model.learner_model_param->num_output_group;
    CHECK_NE(ngroup, 0);
    size_t const ncolumns = model.learner_model_param->num_feature + 1;
    CHECK_NE(ncolumns, 0);
    // allocate space for (number of features + bias) times the number of rows
    std::vector<bst_float>& contribs = *out_contribs;
    contribs.resize(info.num_row_ * ncolumns *
                    model.learner_model_param->num_output_group);
    // make sure contributions is zeroed, we could be reusing a previously
    // allocated one
    std::fill(contribs.begin(), contribs.end(), 0);
// initialize tree node mean values
#pragma omp parallel for schedule(static)
    for (bst_omp_uint i = 0; i < ntree_limit; ++i) {
      model.trees[i]->FillNodeMeanValues();
    }
    const std::vector<bst_float>& base_margin = info.base_margin_.HostVector();
    // start collecting the contributions
    for (const auto& batch : p_fmat->GetBatches<SparsePage>()) {
      // parallel over local batch
      const auto nsize = static_cast<bst_omp_uint>(batch.Size());
#pragma omp parallel for schedule(static)
      for (bst_omp_uint i = 0; i < nsize; ++i) {
        auto row_idx = static_cast<size_t>(batch.base_rowid + i);
        RegTree::FVec& feats = thread_temp_[omp_get_thread_num()];
        std::vector<bst_float> this_tree_contribs(ncolumns);
        // loop over all classes
        for (int gid = 0; gid < ngroup; ++gid) {
          bst_float* p_contribs =
              &contribs[(row_idx * ngroup + gid) * ncolumns];
          feats.Fill(batch[i]);
          // calculate contributions
          for (unsigned j = 0; j < ntree_limit; ++j) {
            std::fill(this_tree_contribs.begin(), this_tree_contribs.end(), 0);
            if (model.tree_info[j] != gid) {
              continue;
            }
            if (!approximate) {
              model.trees[j]->CalculateContributions(
                  feats, &this_tree_contribs[0], condition, condition_feature);
            } else {
              model.trees[j]->CalculateContributionsApprox(
                  feats, &this_tree_contribs[0]);
            }
            for (size_t ci = 0; ci < ncolumns; ++ci) {
              p_contribs[ci] +=
                  this_tree_contribs[ci] *
                  (tree_weights == nullptr ? 1 : (*tree_weights)[j]);
            }
          }
          feats.Drop(batch[i]);
          // add base margin to BIAS
          if (base_margin.size() != 0) {
            p_contribs[ncolumns - 1] += base_margin[row_idx * ngroup + gid];
          } else {
            p_contribs[ncolumns - 1] += model.learner_model_param->base_score;
          }
        }
      }
    }
  }

  void PredictInteractionContributions(DMatrix* p_fmat,
                                       std::vector<bst_float>* out_contribs,
                                       const gbm::GBTreeModel& model,
                                       unsigned ntree_limit,
                                       std::vector<bst_float>* tree_weights,
                                       bool approximate) override {
    const MetaInfo& info = p_fmat->Info();
    const int ngroup = model.learner_model_param->num_output_group;
    size_t const ncolumns = model.learner_model_param->num_feature;
    const unsigned row_chunk = ngroup * (ncolumns + 1) * (ncolumns + 1);
    const unsigned mrow_chunk = (ncolumns + 1) * (ncolumns + 1);
    const unsigned crow_chunk = ngroup * (ncolumns + 1);

    // allocate space for (number of features^2) times the number of rows and
    // tmp off/on contribs
    std::vector<bst_float>& contribs = *out_contribs;
    contribs.resize(info.num_row_ * ngroup * (ncolumns + 1) * (ncolumns + 1));
    std::vector<bst_float> contribs_off(info.num_row_ * ngroup *
                                        (ncolumns + 1));
    std::vector<bst_float> contribs_on(info.num_row_ * ngroup * (ncolumns + 1));
    std::vector<bst_float> contribs_diag(info.num_row_ * ngroup *
                                         (ncolumns + 1));

    // Compute the difference in effects when conditioning on each of the
    // features on and off see: Axiomatic characterizations of probabilistic and
    //      cardinal-probabilistic interaction indices
    PredictContribution(p_fmat, &contribs_diag, model, ntree_limit,
                        tree_weights, approximate, 0, 0);
    for (size_t i = 0; i < ncolumns + 1; ++i) {
      PredictContribution(p_fmat, &contribs_off, model, ntree_limit,
                          tree_weights, approximate, -1, i);
      PredictContribution(p_fmat, &contribs_on, model, ntree_limit,
                          tree_weights, approximate, 1, i);

      for (size_t j = 0; j < info.num_row_; ++j) {
        for (int l = 0; l < ngroup; ++l) {
          const unsigned o_offset =
              j * row_chunk + l * mrow_chunk + i * (ncolumns + 1);
          const unsigned c_offset = j * crow_chunk + l * (ncolumns + 1);
          contribs[o_offset + i] = 0;
          for (size_t k = 0; k < ncolumns + 1; ++k) {
            // fill in the diagonal with additive effects, and off-diagonal with
            // the interactions
            if (k == i) {
              contribs[o_offset + i] += contribs_diag[c_offset + k];
            } else {
              contribs[o_offset + k] =
                  (contribs_on[c_offset + k] - contribs_off[c_offset + k]) /
                  2.0;
              contribs[o_offset + i] -= contribs[o_offset + k];
            }
          }
        }
      }
    }
  }

 private:
  std::mutex lock_;
  std::vector<RegTree::FVec> thread_temp_;
  common::Monitor monitor_;
};

XGBOOST_REGISTER_PREDICTOR(CPUPredictor, "cpu_predictor")
    .describe("Make predictions using CPU.")
    .set_body([](GenericParameter const* generic_param) {
      return new CPUPredictor(generic_param);
    });
}  // namespace predictor
}  // namespace xgboost
