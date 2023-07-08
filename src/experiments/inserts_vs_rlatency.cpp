/**
 * inserts_vs_rlatency.cpp
 *
 * This experiment measures read latency as a function of the number of inserts
 * under a variety of configurations. We also compare against the baseline time
 * to do the same reads on a model trained from scratch without inserts.
 *
 * Input:
 * - eps
 * - eps_rec
 * - initial_n = initial data size
 * - num_inserts = total number of inserts to perform
 * - granularity = how many inserts to perform at a time up to num_inserts
 * - num_reads = total number of reads to time
 * - seeds = list of seeds to run this experiment on
 * - Configs
 *
 * Output:
 * - name = name of the configuration generating this row, or "baseline"
 * - data_size = the size of the model before inserts
 * - seed = seed used for this run
 * - time = ms required to perform the reads
 */

#include "experiments.h"

namespace
{
  // Input
  size_t eps = 128;
  size_t eps_rec = 16;
  size_t initial_n = 1e5;
  size_t num_inserts = 1e5;
  size_t granularity = 2e4;
  size_t num_reads = 1e5;
  std::vector<int> seeds = {1, 2, 3};

  std::vector<Configuration> get_configs()
  {
    Configuration base_config;
    base_config.eps = eps;
    base_config.eps_rec = eps_rec;

    Configuration base_in_place = base_config;
    base_in_place.name = "in_place";
    base_in_place.fill_ratio = 0.75;
    base_in_place.fill_ratio_rec = 0.75;
    base_in_place.buffer_size = 0;
    base_in_place.split_neighborhood = 0;

    Configuration in_place_n0 = base_in_place;
    in_place_n0.name = "in_place_n0";
    in_place_n0.split_neighborhood = 0;

    Configuration base_out_place = base_config;
    base_out_place.name = "out_place";
    base_out_place.fill_ratio = 1.0;
    base_out_place.fill_ratio_rec = 1.0;
    base_out_place.buffer_size = 128;
    base_out_place.split_neighborhood = 0;

    Configuration out_place_n0 = base_out_place;
    out_place_n0.name = "out_place_n0";
    out_place_n0.split_neighborhood = 0;

    Configuration out_place_n2 = base_out_place;
    out_place_n2.name = "out_place_n2";
    out_place_n2.split_neighborhood = 2;

    Configuration out_place_n8 = base_out_place;
    out_place_n8.name = "out_place_n8";
    out_place_n8.split_neighborhood = 8;

    return {
        in_place_n0,
        out_place_n0,
        out_place_n2,
        out_place_n8};
  }
}

void run_inserts_vs_rlatency(std::string filename)
{
  std::vector<Configuration> configs = get_configs();
  // Parameters for the indices
  std::ofstream fout;
  fout.open(filename);
  fout << "name,data_size,seed,time" << std::endl;
  progressbar bar(seeds.size() + 1);
  bar.update();
  for (auto &seed : seeds)
  {
    // Get the data
    auto data = get_random_data(initial_n, seed);
    auto inserts = get_random_inserts(num_inserts, granularity);
    auto reads = get_random_reads(data, num_reads);

    for (auto &config : configs)
    {
      size_t fixed_epsilon = (size_t)((float)eps_rec / config.fill_ratio);
      size_t fixed_epsilon_recursive = (size_t)((float)eps_rec / config.fill_ratio_rec);
      auto pgm = pgm::OopPGMIndex<uint32_t, uint32_t>(
          data.begin(),
          data.end(),
          fixed_epsilon,
          fixed_epsilon_recursive,
          config.fill_ratio,
          config.fill_ratio_rec,
          config.buffer_size,
          config.split_neighborhood);
      size_t size = initial_n;
      for (auto &insert_data : inserts)
      {
        do_inserts(pgm, insert_data);
        size += insert_data.size();
        size_t result = time_reads(pgm, reads);
        fout << config.name << "," << size << "," << seed << "," << result << std::endl;
      }
    }

    // Baseline experiment
    size_t clean_size = initial_n;
    for (auto &insert_data : inserts)
    {
      clean_size += insert_data.size();
      data.insert(data.end(), insert_data.begin(), insert_data.end());
      auto baseline_pgm = pgm::OopPGMIndex<uint32_t, uint32_t>(data.begin(), data.end(), eps, eps_rec, 1.0, 1.0, 0, 0);
      size_t result = time_reads(baseline_pgm, reads);
      fout << "baseline," << clean_size << "," << seed << "," << result << std::endl;
    }
    bar.update();
  }
  fout.close();
}