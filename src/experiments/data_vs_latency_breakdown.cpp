/**
 * data_vs_latency_breakdown
 *
 * This experiment is part of one of the report's final insights surrouding overall performance of the
 * structure relative to in-place. We want to understand the latency of inplace vs a tuned out of place
 * as the data size grows. On top of that, we want to understand
 * where this latency is coming from, i.e. what portion of it is coming from reads in the workload vs
 * writes in the workload.
 *
 * Input:
 * - eps
 * - eps_rec
 * - initial_ns
 * - seeds
 * - num_trials
 * - write_props
 * - configs
 *
 * Output:
 * - conf_name: name given to this configuration
 * - work_name: Name given to this workload
 * - seed: seed used during this run
 * - trial: trial # of this run
 * - n: the number of elements in the workload
 * - rtime: time (ms) spent doing reads in this workload
 * - wtime: time (ms) spend doing writes in this workload
 */

#include "experiments.h"

namespace
{
  // Input
  size_t eps = 64;
  size_t eps_rec = 16;
  std::vector<size_t> initial_ns = {
      (size_t)1e6,
      (size_t)3e6,
      (size_t)1e7,
      (size_t)3e6};
  std::vector<int> seeds = {1, 2};
  size_t num_trials = 1;
  std::vector<float> write_props = {0.2, 0.4};

  std::vector<Configuration>
  get_configs()
  {
    /* BASE CONFIGS */
    Configuration base_config;
    base_config.eps = eps;
    base_config.eps_rec = eps_rec;

    Configuration base_outplace = base_config;
    base_outplace.buffer_size = 64;
    base_outplace.fill_ratio = 1.0;
    base_outplace.fill_ratio_rec = 1.0;
    base_outplace.split_neighborhood = 2;

    /* TESTED OUTPLACE CONFIGS */
    Configuration outplace_n2_b48 = base_outplace;
    outplace_n2_b48.name = "outplace_n2_b48";
    outplace_n2_b48.split_neighborhood = 2;
    outplace_n2_b48.buffer_size = 48;

    Configuration outplace_n2_b64 = base_outplace;
    outplace_n2_b64.name = "outplace_n2_b64";
    outplace_n2_b64.split_neighborhood = 2;
    outplace_n2_b64.buffer_size = 64;

    Configuration outplace_n4_b48 = base_outplace;
    outplace_n4_b48.name = "outplace_n4_b48";
    outplace_n4_b48.split_neighborhood = 4;
    outplace_n4_b48.buffer_size = 48;

    return {
        outplace_n2_b48,
        outplace_n2_b64,
        outplace_n4_b48,
    };
  }
}

void run_data_vs_latency_breakdown(std::string filename)
{
  std::ofstream fout;
  fout.open(filename);
  fout << "conf_name,work_name,seed,trial,n,rtime,wtime" << std::endl;
  auto configs = get_configs();
  progressbar bar(seeds.size() * initial_ns.size() * write_props.size() * num_trials * (configs.size() + 1));
  for (auto &seed : seeds)
  {
    for (auto &initial_n : initial_ns)
    {
      for (auto &write_prop : write_props)
      {
        int read_perc = std::round((1 - write_prop) * 100);
        std::string work_name = "R" + std::to_string(read_perc) + "W" + std::to_string(100 - read_perc);
        auto workload = generate_workload(work_name, initial_n, write_prop, initial_n, seed);
        for (size_t trial = 0; trial < num_trials; ++trial)
        {
          for (auto &config : configs)
          {
            auto [rtime, wtime, memory, pgm] = lspecific_benchmark_workload_config(workload, config);
            fout << config.name << ","
                 << work_name << ","
                 << seed << ","
                 << trial << ","
                 << initial_n << ","
                 << rtime << ","
                 << wtime << std::endl;
            std::cout << "pgm mem " << memory << std::endl;
            bar.update();
          }
          auto [rtime, wtime, memory] = ALEX_lspecific_benchmark_workload_config(workload);
          fout << "ALEX"
               << ","
               << work_name << ","
               << seed << ","
               << trial << ","
               << initial_n << ","
               << rtime << ","
               << wtime << std::endl;
          std::cout << "ALEX mem " << memory << std::endl;
          bar.update();
        }
      }
    }
  }
}