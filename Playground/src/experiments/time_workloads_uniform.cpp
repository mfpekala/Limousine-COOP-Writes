/**
 * benchmark_workloads_uniform
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
  size_t eps = 64;
  size_t eps_rec = 8;
  size_t initial_n = 5e5;
  size_t num_ops = 5e5;
  std::vector<int> seeds = {1, 2, 3};
  size_t num_trials = 2;
  std::vector<float> write_props = {0.0, 0.1, 0.2, 0.3, 0.4};

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
    base_outplace.split_neighborhood = 2;

    /* TESTED INPLACE CONFIGS */
    Configuration inplace_n0 = base_inplace;
    inplace_n0.name = "inplace_n0";
    inplace_n0.split_neighborhood = 0;

    Configuration inplace_n1 = base_inplace;
    inplace_n1.name = "inplace_n1";
    inplace_n1.split_neighborhood = 1;

    Configuration inplace_n4 = base_inplace;
    inplace_n4.name = "inplace_n4";
    inplace_n4.split_neighborhood = 4;

    Configuration inplace_n8 = base_inplace;
    inplace_n8.name = "inplace_n8";
    inplace_n8.split_neighborhood = 8;

    /* TESTED OUTPLACE CONFIGS */
    Configuration outplace_n0 = base_outplace;
    outplace_n0.name = "outplace_n0";
    outplace_n0.split_neighborhood = 0;

    Configuration outplace_n1 = base_outplace;
    outplace_n1.name = "outplace_n1";
    outplace_n1.split_neighborhood = 1;

    Configuration outplace_n2 = base_outplace;
    outplace_n2.name = "outplace_n2";
    outplace_n2.split_neighborhood = 2;

    Configuration outplace_n4 = base_outplace;
    outplace_n4.name = "outplace_n4";
    outplace_n4.split_neighborhood = 4;

    Configuration outplace_n8 = base_outplace;
    outplace_n8.name = "outplace_n8";
    outplace_n8.split_neighborhood = 8;

    Configuration outplace_b8 = base_outplace;
    outplace_b8.name = "outplace_b8";
    outplace_b8.buffer_size = 8;

    Configuration outplace_b16 = base_outplace;
    outplace_b16.name = "outplace_b16";
    outplace_b16.buffer_size = 16;

    Configuration outplace_b32 = base_outplace;
    outplace_b32.name = "outplace_b32";
    outplace_b32.buffer_size = 32;

    Configuration outplace_b64 = base_outplace;
    outplace_b64.name = "outplace_b64";
    outplace_b64.buffer_size = 64;

    Configuration outplace_b128 = base_outplace;
    outplace_b128.name = "outplace_b128";
    outplace_b128.buffer_size = 128;

    Configuration outplace_b256 = base_outplace;
    outplace_b256.name = "outplace_b256";
    outplace_b256.buffer_size = 256;

    return {
        outplace_b8,
        outplace_b16,
        outplace_b32,
        outplace_b64,
        outplace_b128,
        outplace_b256,
    };
  }
}

void run_benchmark_workloads_uniform(std::string filename)
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