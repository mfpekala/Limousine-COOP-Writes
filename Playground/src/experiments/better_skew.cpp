/**
 * better_skew
 *
 * This experiment aims to explore how the model performs/degrades when it is used on
 * skewed data. BUT, it does it better
 *
 * Input:
 * - eps
 * - eps_rec
 * - initial_n
 * - num_ops
 * - num_trials
 * - write_prop
 * - skew (how skewed to make the data on inserts)
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
  size_t eps = 32;
  size_t eps_rec = 8;
  size_t initial_n = 5e6;
  size_t num_ops = 5e6;
  size_t granularity = 5e5;
  size_t num_trials = 5;
  float write_prop = 0.5;
  float skew = 0.9;

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
    base_outplace.buffer_size = 48;
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
        // inplace_n0,
        // outplace_n0,
        // outplace_n1,
        outplace_n2,
        outplace_n4,
        // outplace_n8,
    };
  }
}

void run_better_skew(std::string filename)
{
  std::ofstream fout;
  fout.open(filename);
  fout << "name,trial,skew,n,time,size,index_power" << std::endl;
  auto configs = get_configs();
  progressbar bar(num_trials * configs.size());
  for (size_t trial = 0; trial < num_trials; ++trial)
  {
    // First generate the base data
    auto base_workload = generate_workload("baseline", initial_n, write_prop, num_ops, 6);
    // NOTE: We'll end up only using the below for it's inserts, not it's base data
    auto skewed_workload = base_workload;
    for (int ix = 0; ix < skewed_workload.ops.size(); ++ix)
    {
      auto op = skewed_workload.ops[ix];
      if (op.type == READ)
        continue;
      skewed_workload.ops[ix].key /= 10;
    }

    // Then measure what this looks like on normal, continuing random data
    for (auto &config : configs)
    {
      auto baseline_pgm = pgm::BufferedPGMIndex<uint32_t, uint32_t>(
          base_workload.initial_data.begin(),
          base_workload.initial_data.end(),
          eps, eps_rec,
          config.fill_ratio,
          config.fill_ratio_rec,
          config.buffer_size, config.split_neighborhood);
      auto skew_pgm = pgm::BufferedPGMIndex<uint32_t, uint32_t>(
          base_workload.initial_data.begin(),
          base_workload.initial_data.end(),
          eps, eps_rec,
          config.fill_ratio,
          config.fill_ratio_rec,
          config.buffer_size, config.split_neighborhood);

      for (int start_x = 0; start_x < num_ops; start_x += granularity)
      {
        auto base_data_start = base_workload.ops.begin() + start_x;
        auto base_data_end = base_data_start + granularity;
        auto base_data_this = std::vector<Op>(base_data_start, base_data_end);
        auto skewed_data_start = skewed_workload.ops.begin() + start_x;
        auto skewed_data_end = skewed_data_start + granularity;
        auto skewed_data_this = std::vector<Op>(skewed_data_start, skewed_data_end);

        auto time_start = std::chrono::high_resolution_clock::now();
        for (auto &op : base_data_this)
        {
          if (op.type == READ)
            baseline_pgm.find(op.key);
          else
            baseline_pgm.insert(op.key, op.val);
        }
        auto time_end = std::chrono::high_resolution_clock::now();
        auto base_time = std::chrono::duration_cast<std::chrono::microseconds>(time_end - time_start).count();
        fout << config.name << ","
             << trial << ","
             << 0 << ","
             << start_x << ","
             << base_time << ","
             << baseline_pgm.size_in_bytes() << ","
             << get_avg_leaf_size(baseline_pgm) << std::endl;

        time_start = std::chrono::high_resolution_clock::now();
        for (auto &op : skewed_data_this)
        {
          if (op.type == READ)
            skew_pgm.find(op.key);
          else
            skew_pgm.insert(op.key, op.val);
        }
        time_end = std::chrono::high_resolution_clock::now();
        auto skew_time = std::chrono::duration_cast<std::chrono::microseconds>(time_end - time_start).count();
        fout << config.name << ","
             << trial << ","
             << 1 << ","
             << start_x << ","
             << skew_time << ","
             << skew_pgm.size_in_bytes() << ","
             << get_avg_leaf_size(skew_pgm) << std::endl;
        bar.update();
      }
    }
    bar.update();
  }
}