#pragma once

#include <cstddef>
#include <iomanip>
#include <sstream>
#include <string>
#include <utility>

struct PredictionMetrics {
  std::string algorithm{AlgorithmName()};
  double predict_batch_seconds{0.0};

  bool has_do_metrics{false};
  double epsilon{1.0};
  double delta{0.00001};
  std::string shuffle_method{"BitonicShuffler"};
  bool doxie_memory_alignment{true};
  bool doxie_blocked_kernel{true};
  bool doxie_advanced_composition{true};

  double add_dummy_seconds{0.0};
  double post_process_seconds{0.0};
  double predict_dmatrix_do_seconds{0.0};
  double predict_online_seconds{0.0};
  double predict_no_seconds{0.0};
  double shuffle_seconds{0.0};

  void ResetTimings() {
    algorithm = AlgorithmName();
    predict_batch_seconds = 0.0;
    has_do_metrics = false;
    add_dummy_seconds = 0.0;
    post_process_seconds = 0.0;
    predict_dmatrix_do_seconds = 0.0;
    predict_online_seconds = 0.0;
    predict_no_seconds = 0.0;
    shuffle_seconds = 0.0;
  }

  static double Seconds(std::pair<size_t, size_t> const& metric) {
    return metric.first == 0 ? 0.0 : static_cast<double>(metric.second) / 1e+6;
  }

  static std::string AlgorithmName() {
#ifdef __ENCLAVE_OBLIVIOUS__
#ifdef __ENCLAVE_DPOBLIVIOUS__
    return "DO";
#else
    return "O";
#endif
#else
    return "NO";
#endif
  }

  std::string ToJson() const {
    std::ostringstream os;
    os << std::setprecision(12);
    os << "{\"algorithm\":\"" << EscapeJson(algorithm) << "\"";
    os << ",\"PredictBatch\":" << predict_batch_seconds;
    if (has_do_metrics) {
      os << ",\"epsilon\":" << epsilon;
      os << ",\"delta\":" << delta;
      os << ",\"shuffleMethod\":\"" << EscapeJson(shuffle_method) << "\"";
      os << ",\"doxieMemoryAlignment\":"
         << (doxie_memory_alignment ? "true" : "false");
      os << ",\"doxieBlockedKernel\":"
         << (doxie_blocked_kernel ? "true" : "false");
      os << ",\"doxieAdvancedComposition\":"
         << (doxie_advanced_composition ? "true" : "false");
      os << ",\"AddDummy\":" << add_dummy_seconds;
      os << ",\"PostProcess\":" << post_process_seconds;
      os << ",\"PredictDMatrixDO\":" << predict_dmatrix_do_seconds;
      os << ",\"PredictOnline\":" << predict_online_seconds;
      os << ",\"PredictNO\":" << predict_no_seconds;
      os << ",\"shuffle\":" << shuffle_seconds;
    }
    os << "}";
    return os.str();
  }

 private:
  static std::string EscapeJson(std::string const& value) {
    std::ostringstream os;
    for (char c : value) {
      if (c == '"' || c == '\\') {
        os << '\\';
      }
      os << c;
    }
    return os.str();
  }
};
