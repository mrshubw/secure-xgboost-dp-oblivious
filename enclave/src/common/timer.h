/*!
 * Copyright by Contributors 2017-2019
 * Modifications Copyright (c) 2020-22 by Secure XGBoost Contributors
 */
#pragma once
#include <xgboost/logging.h>
#include <chrono>
#include <iostream>
#include <fstream>
#include <map>
#include <string>
#include <utility>
#include <vector>

namespace xgboost {
namespace common {

struct Timer {
#ifdef __ENCLAVE__ // replace high_resolution_clock with system_clock for OE
  using ClockT = std::chrono::system_clock;
  using TimePointT = std::chrono::system_clock::time_point;
  using DurationT = std::chrono::system_clock::duration;
#else
  using ClockT = std::chrono::high_resolution_clock;
  using TimePointT = std::chrono::high_resolution_clock::time_point;
  using DurationT = std::chrono::high_resolution_clock::duration;
#endif // __ENCLAVE__
  using SecondsT = std::chrono::duration<double>;

  TimePointT start;
  DurationT elapsed;
  Timer() { Reset(); }
  void Reset() {
    elapsed = DurationT::zero();
    Start();
  }
  void Start() { start = ClockT::now(); }
  void Stop() { elapsed += ClockT::now() - start; }
  double ElapsedSeconds() const { return SecondsT(elapsed).count(); }
  void PrintElapsed(std::string label) {
    char buffer[255];
    snprintf(buffer, sizeof(buffer), "%s:\t %fs", label.c_str(),
             SecondsT(elapsed).count());
    LOG(CONSOLE) << buffer;
    Reset();
  }
  void PrintElapsed(std::string label, std::string logfile) {
    // Create an ofstream (output file stream) object
    std::fstream outfile;

    // Open the file in write mode
    outfile.open(logfile, std::ios::out|std::ios::app);
    // Check if the file was opened successfully
    if (!outfile) {
        std::cerr << "Error opening file" << std::endl;
        return;
    }
    outfile<<"algorithm: ";
#ifdef __ENCLAVE_OBLIVIOUS__
#ifdef __ENCLAVE_DPOBLIVIOUS__
    outfile<<"DO\n";
#else
    outfile<<" O\n";
#endif
#else
    outfile<<"NO\n";
#endif
    outfile << label << ElapsedSeconds() << std::endl;

    // Close the file
    outfile.close();

    Reset();
  }
};

/**
 * \struct  Monitor
 *
 * \brief Timing utility used to measure total method execution time over the
 * lifetime of the containing object.
 */
struct Monitor {
 private:
  struct Statistics {
    Timer timer;
    size_t count{0};
    uint64_t nvtx_id;
  };

  // from left to right, <name <count, elapsed>>
  using StatMap = std::map<std::string, std::pair<size_t, size_t>>;

  std::string label_ = "";
  std::map<std::string, Statistics> statistics_map_;
  Timer self_timer_;

  /*! \brief Collect time statistics across all workers. */
  std::vector<StatMap> CollectFromOtherRanks() const;
  void PrintStatistics(StatMap const& statistics) const;

 public:
  Monitor() { self_timer_.Start(); }
  /*\brief Print statistics info during destruction.
   *
   * Please note that this may not work, as with distributed frameworks like Dask, the
   * model is pickled to other workers, and the global parameters like `global_verbosity_`
   * are not included in the pickle.
   */
  ~Monitor() {
    this->Print();
    self_timer_.Stop();
  }

  /*! \brief Print all the statistics. */
  void Print() const;
  void PrintForce(std::string logfile) const;
  void StartForce(const std::string &name);
  void StopForce(const std::string &name);

  void Init(std::string label) { this->label_ = label; }
  void Start(const std::string &name);
  void Stop(const std::string &name);
  std::pair<size_t, size_t> GetCost(const std::string &name);
};
}  // namespace common
}  // namespace xgboost
