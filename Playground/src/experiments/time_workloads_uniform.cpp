/**
 * time_workloads_uniform
 *
 * This experiment aims to give a deeper understanding of the trade-off space of this model.
 * On the x-axis, it will vary the proportion of reads in the workload, and on the y-axis
 * it will measure the time that it takes to perform that workload.
 *
 * Input:
 * - eps
 * - eps_rec
 * - initial_n
 * - num_ops
 * - seeds
 * - num_trials
 * - write_props
 * - configs
 *
 * Output:
 * - name: name given to this configuration
 * - seed: seed used during this run
 * - trial: trial # of this run
 * - write_prop: the proporion of writes in the workload
 * - time: time (ms) to complete this workload (NOTE: does _not_ include time to construct initial index)
 * - size: size (bytes) of the model at the end of this workload
 */

#include "experiments.h"

namespace
{
  // Input
  size_t eps = 128;
  size_t eps_rec = 16;
  size_t initial_n = 1e7;
  size_t num_ops = 1e7;
  std::vector<int> seeds = {1, 2, 3};
  size_t num_trials = 1;
  std::vector<float> write_props = {0.0, 0.1, 0.2, 0.3};

  std::vector<Configuration>
  get_configs()
  {
    /* BASE CONFIGS */
    Configuration base_config;
    base_config.eps = eps;
    base_config.eps_rec = eps_rec;

    Configuration base_inplace = base_config;
    base_inplace.buffer_size = 0;
    base_inplace.fill_ratio = 0.75;
    base_inplace.fill_ratio_rec = 0.75;

    Configuration base_outplace = base_config;
    base_outplace.buffer_size = 128;
    base_outplace.fill_ratio = 1.0;
    base_outplace.fill_ratio_rec = 1.0;

    /* TESTED INPLACE CONFIGS */
    Configuration inplace_n0 = base_inplace;
    inplace_n0.name = "inplace_n0";
    inplace_n0.split_neighborhood = 0;

    Configuration inplace_n4 = base_inplace;
    inplace_n4.name = "inplace_n4";
    inplace_n4.split_neighborhood = 4;

    /* TESTED OUTPLACE CONFIGS */
    Configuration outplace_n0 = base_outplace;
    outplace_n0.name = "outplace_n0";
    outplace_n0.split_neighborhood = 0;

    Configuration outplace_n4 = base_outplace;
    outplace_n4.name = "outplace_n4";
    outplace_n4.split_neighborhood = 4;

    return {
        inplace_n0,
        inplace_n4,
        outplace_n0,
        outplace_n4};
  }
}

void run_time_workloads_uniform(std::string filename)
{
  std::ofstream fout;
  fout.open(filename);
  fout << "name,seed,trial,write_prop,time,size" << std::endl;
  auto configs = get_configs();
  progressbar bar(num_trials * seeds.size() * write_props.size() * configs.size());
  for (size_t trial = 0; trial < num_trials; ++trial)
  {
    for (auto &seed : seeds)
    {
      for (auto &write_prop : write_props)
      {
        auto workload = generate_workload("workload", initial_n, write_prop, num_ops, seed);
        for (auto &config : configs)
        {
          auto [time, memory] = benchmark_workload_config(workload, config);
          fout << config.name << ","
               << seed << ","
               << trial << ","
               << write_prop << ","
               << time << ","
               << memory << std::endl;
          bar.update();
        }
      }
    }
  }
}