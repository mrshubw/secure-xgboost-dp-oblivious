#pragma once

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <limits>
#include <memory>
#include <mutex>
#include <random>
#include <string>
#include <vector>

#include "../../enclave/src/common/timer.h"
#include "enclave/prediction_metrics.h"
#include "enclave/obl_primitives.h"
#include "xgboost/base.h"
#include "xgboost/data.h"
#include "xgboost/tree_model.h"
#include "psrr/shuffle.h"

namespace xgboost {
namespace common {
struct Monitor;
}  // namespace common
namespace gbm {
struct GBTreeModel;
}  // namespace gbm
namespace doxie {

struct InferenceContext {
  PredictionMetrics* metrics{nullptr};
  common::Monitor* monitor{nullptr};
  std::vector<RegTree::FVec>* thread_temp{nullptr};
  std::mutex* lock{nullptr};
};

void PredictDMatrix(DMatrix* p_fmat, std::vector<bst_float>* out_preds,
                    gbm::GBTreeModel const& model, int32_t tree_begin,
                    int32_t tree_end, InferenceContext* context);

}  // namespace doxie
}  // namespace xgboost

// Approximate the quantile function of N(0, 1).
inline double inverseCDF(double p) {
  CHECK_GT(p, 0.0);
  CHECK_LT(p, 1.0);

  static constexpr double a[] = {
      -3.969683028665376e+01, 2.209460984245205e+02,
      -2.759285104469687e+02, 1.383577518672690e+02,
      -3.066479806614716e+01, 2.506628277459239e+00};
  static constexpr double b[] = {
      -5.447609879822406e+01, 1.615858368580409e+02,
      -1.556989798598866e+02, 6.680131188771972e+01,
      -1.328068155288572e+01};
  static constexpr double c[] = {
      -7.784894002430293e-03, -3.223964580411365e-01,
      -2.400758277161838e+00, -2.549732539343734e+00,
      4.374664141464968e+00, 2.938163982698783e+00};
  static constexpr double d[] = {
      7.784695709041462e-03, 3.224671290700398e-01,
      2.445134137142996e+00, 3.754408661907416e+00};
  static constexpr double plow = 0.02425;
  static constexpr double phigh = 1.0 - plow;

  if (p < plow) {
    const double q = std::sqrt(-2.0 * std::log(p));
    return (((((c[0] * q + c[1]) * q + c[2]) * q + c[3]) * q + c[4]) *
                q +
            c[5]) /
           ((((d[0] * q + d[1]) * q + d[2]) * q + d[3]) * q + 1.0);
  }
  if (p > phigh) {
    const double q = std::sqrt(-2.0 * std::log(1.0 - p));
    return -(((((c[0] * q + c[1]) * q + c[2]) * q + c[3]) * q + c[4]) *
                 q +
             c[5]) /
           ((((d[0] * q + d[1]) * q + d[2]) * q + d[3]) * q + 1.0);
  }

  const double q = p - 0.5;
  const double r = q * q;
  return (((((a[0] * r + a[1]) * r + a[2]) * r + a[3]) * r + a[4]) *
              r +
          a[5]) *
         q /
         (((((b[0] * r + b[1]) * r + b[2]) * r + b[3]) * r + b[4]) *
              r +
          1.0);
}

inline double calculateSigma(double epsilon, double delta, double sensitivity) {
  const double c = std::sqrt(2.0 * std::log(2.5 / delta));
  return sensitivity * c / epsilon;
}

inline double calculateMean(double sigma, double delta, size_t representatives) {
  CHECK_GT(representatives, 0U);
  const double log_quantile =
      std::log1p(-delta / 2.0) / static_cast<double>(representatives);
  double quantile = std::exp(log_quantile);
  quantile = std::min(quantile, std::nextafter(1.0, 0.0));
  quantile = std::max(quantile, std::nextafter(0.0, 1.0));
  return sigma * inverseCDF(quantile);
}

inline double calculateMean(double sigma, double delta) {
  return calculateMean(sigma, delta, 1);
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

inline PrivacyBudget SplitPrivacyBudgetByBasicComposition(
    double epsilon, double delta, size_t trees_num) {
  CHECK_GT(epsilon, 0.0);
  CHECK_GT(delta, 0.0);
  CHECK_LT(delta, 1.0);

  if (trees_num <= 1) {
    return PrivacyBudget{epsilon, delta};
  }

  const double trees = static_cast<double>(trees_num);
  return PrivacyBudget{epsilon / trees, delta / trees};
}

#define PSRR_OSHUFFLE

class DoxieInference
{
private:
  double epsilon;
  double delta;
  double sensitivity;
  double sigma;
  double mean;
  std::string shuffle_method_;
  bool use_advanced_composition_;
public:
  xgboost::SparsePage shuffle_page;
  std::vector<int> shuffle_index;
  std::vector<xgboost::bst_float> shuffle_preds;
  #ifdef PSRR_OSHUFFLE
  std::unique_ptr<obl::OShuffler> oshuffler;
  #endif

  DoxieInference(/* args */):DoxieInference(1, 0.00001, 1){};
  DoxieInference(double epsilon, double delta, double sensitivity,
             std::string shuffle_method = "BitonicShuffler",
             bool use_advanced_composition = true)
      : epsilon(epsilon),
        delta(delta),
        sensitivity(sensitivity),
        shuffle_method_(shuffle_method),
        use_advanced_composition_(use_advanced_composition) {
    // sigma = calculateSigma(epsilon, delta, sensitivity);
    // mean = calculateMean(sigma, delta);
    // std::cout<<"sigma: "<<sigma<<" mean: "<<mean<<std::endl;
  };
  ~DoxieInference(){};

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

  void Preprocess(xgboost::SparsePage& in_page,
                  std::vector<xgboost::SparsePage>& trees_dummy_samples,
                  size_t tree_nodes_num,
                  PredictionMetrics* metrics = nullptr, int trees_num=1,
                  int num_groups=1){
    CHECK_GT(trees_num, 0);
    PrivacyBudget per_tree_budget =
        use_advanced_composition_
            ? SplitPrivacyBudgetByAdvancedComposition(
                  epsilon, delta, static_cast<size_t>(trees_num))
            : SplitPrivacyBudgetByBasicComposition(
                  epsilon, delta, static_cast<size_t>(trees_num));
    sigma = calculateSigma(per_tree_budget.epsilon, per_tree_budget.delta,
                           sensitivity);
    mean = calculateMean(sigma, per_tree_budget.delta);
    // size_t samples_num = (tree_nodes_num/2)/200;
    // std::cout<<"trees_num: "<<trees_num<<" samples_num: "<<samples_num<<std::endl;
    
    auto noise_page = xgboost::SparsePagePadding::FromSparsePage(in_page);

    for (size_t i = 0; i < trees_num; i++)
    {
      xgboost::SparsePagePadding dummySamples(noise_page.fixed_row_size);
      ProduceDummySamples(dummySamples, trees_dummy_samples[i]);

      xgboost::common::Timer add_dummy_timer;
      AddDummy(noise_page, dummySamples);
      add_dummy_timer.Stop();
      if (metrics != nullptr) {
        metrics->add_dummy_seconds += add_dummy_timer.ElapsedSeconds();
      }
    }

    // std::cout<<"the number of dummies: "<<noise_page.Size()-in_page.Size()<<std::endl;
    
    // for (size_t  i = 0; i < noise_page.Size(); i++)
    // {
    //   std::cout << " "<<i<<": " << noise_page[i].size() << std::endl;
    // }
    
    // shuffle
    xgboost::common::Timer shuffle_timer;
    shuffle_index.resize(noise_page.Size());
    shuffle_preds.resize(noise_page.Size() * num_groups);
    // std::cout<<"noise_page.Size(): "<<noise_page.Size()<<std::endl;
    #ifdef PSRR_OSHUFFLE
    oshuffler = obl::getShuffler(shuffle_method_);
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
    #endif
    shuffle_timer.Stop();
    if (metrics != nullptr) {
      metrics->shuffle_seconds += shuffle_timer.ElapsedSeconds();
    }
  }

  void Preprocess(xgboost::SparsePage& in_page,
                  xgboost::SparsePage& dummy_samples,
                  size_t dummy_max_entries,
                  PredictionMetrics* metrics = nullptr, int num_groups=1) {
    const size_t fixed_row_size =
        std::max(in_page.MaxNumberOfEntries(), dummy_max_entries);
    xgboost::SparsePagePadding noise_page(fixed_row_size);
    noise_page.FrommSparsePage(in_page);

    xgboost::common::Timer add_dummy_timer;
    for (size_t i = 0; i < dummy_samples.Size(); ++i) {
      noise_page.ExpandAndWrite(i, dummy_samples);
    }
    add_dummy_timer.Stop();
    if (metrics != nullptr) {
      metrics->add_dummy_seconds += add_dummy_timer.ElapsedSeconds();
    }

    xgboost::common::Timer shuffle_timer;
    shuffle_index.resize(noise_page.Size());
    shuffle_preds.resize(noise_page.Size() * num_groups);
    #ifdef PSRR_OSHUFFLE
    oshuffler = obl::getShuffler(shuffle_method_);
    oshuffler->shuffle((uint8_t*)noise_page.data.HostVector().data(), noise_page.Size(), noise_page.fixed_row_size*sizeof(xgboost::Entry));
    shuffle_page.Push(noise_page);
    #endif
    shuffle_timer.Stop();
    if (metrics != nullptr) {
      metrics->shuffle_seconds += shuffle_timer.ElapsedSeconds();
    }
  }

  void Preprocess(xgboost::SparsePage& in_page,
                  xgboost::SparsePage& dummy_samples,
                  PredictionMetrics* metrics = nullptr, int num_groups=1) {
    Preprocess(in_page, dummy_samples, dummy_samples.MaxNumberOfEntries(),
               metrics, num_groups);
  }

  void PostProcess(std::vector<xgboost::bst_float>* out_preds,
                   PredictionMetrics* metrics = nullptr){
    xgboost::common::Timer post_process_timer;

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
    
    post_process_timer.Stop();
    if (metrics != nullptr) {
      metrics->post_process_seconds += post_process_timer.ElapsedSeconds();
    }
  }
};
