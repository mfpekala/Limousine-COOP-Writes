/**
 * compare_workloads
 *
 * This experiment aims to explore how the model performs/degrades when it is used on
 * skewed data.
 *
 * Input:
 * - eps
 * - eps_rec
 * - initial_n
 * - num_ops
 * - num_trials
 * - write_prop
 * - skews
 * - configs
 *
 * Output:
 * - name: name given to this configuration
 * - trial: trial # of this run
 * - skew: the skew of the workload
 * - time: time (ms) to complete this workload (NOTE: does _not_ include time to construct initial index)
 * - size: size (bytes) of the model at the end of this workload
 * - index_power: the index power of leaf nodes (avg size) of the model at the end of this workload
 */

#include "experiments.h"

namespace
{
  // Input
  size_t eps = 64;
  size_t eps_rec = 8;
  size_t initial_n = 5e6;
  size_t num_ops = 5e6;
  size_t num_trials = 3;
  float write_prop = 0.3;
  std::vector<float> skews = {0.1, 0.3, 0.5, 0.7, 0.9};

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

    return {
        inplace_n0,
        outplace_n0,
        outplace_n1,
        outplace_n2,
        outplace_n4,
        outplace_n8,
    };
  }
}

void run_compare_workloads(std::string filename)
{
  std::ofstream fout;
  fout.open(filename);
  fout << "name,trial,skew,time,size,index_power" << std::endl;
  auto configs = get_configs();
  progressbar bar(num_trials * skews.size() * configs.size());
  for (size_t trial = 0; trial < num_trials; ++trial)
  {
    for (auto &skew : skews)
    {
      std::string name = "skew_" + std::to_string(skew);
      auto workload = generate_skewed_workload(name, initial_n, write_prop, skew, num_ops);
      for (auto &config : configs)
      {
        auto [time, memory, pgm] = benchmark_workload_config(workload, config);
        fout << config.name << ","
             << trial << ","
             << skew << ","
             << time << ","
             << memory << ","
             << get_avg_leaf_size(pgm) << ","
             << pgm.get_tree_shape().encode() << ","
             << pgm.read_profile.encode() << ","
             << pgm.split_history.encode() << std::endl;
        bar.update();
      }
    }
  }
}