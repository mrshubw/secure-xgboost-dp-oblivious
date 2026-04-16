#pragma once

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <limits>
#include <memory>
#include <random>
#include <string>
#include <vector>

#include "../../enclave/src/common/timer.h"
#include "enclave/prediction_metrics.h"
#include "enclave/obl_primitives.h"
#include "xgboost/base.h"
#include "xgboost/data.h"
#include "psrr/shuffle.h"

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

class DoxieInference
{
private:
  double epsilon;
  double delta;
  double sensitivity;
  double sigma;
  double mean;
  std::string shuffle_method_;
public:
  xgboost::SparsePage shuffle_page;
  std::vector<int> shuffle_index;
  std::vector<xgboost::bst_float> shuffle_preds;
  #ifdef PSRR_OSHUFFLE
  std::unique_ptr<obl::OShuffler> oshuffler;
  #endif

  DoxieInference(/* args */):DoxieInference(1, 0.00001, 1){};
  DoxieInference(double epsilon, double delta, double sensitivity,
             std::string shuffle_method = "BitonicShuffler")
      : epsilon(epsilon),
        delta(delta),
        sensitivity(sensitivity),
        shuffle_method_(shuffle_method) {
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
