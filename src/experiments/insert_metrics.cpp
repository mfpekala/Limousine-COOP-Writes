/**
 * insert_metrics
 *
 * The primary goal of this experiment is to look inside our model and see how it is
 * working. This can be used for many things, but the most important task is diagnosing
 * performance issues that are slowing down our model, especially for reads.
 *
 * Input:
 * - Epsilon
 * - Epsilon recursive
 * - initial_n = initial data size
 * - num_inserts = total number of inserts to perform
 * - granularity = how many inserts to perform at a time up to num_inserts
 * - num_reads = the number of reads to profile at every granularity
 * - seeds = list of seeds to run this experiment on
 * - num_trials = how many times to repeat each seed
 * - List of configurations
 *
 * Output:
 * - name = name of the configuration generating this row, or "baseline"
 * - data_size = total data size (n + #done so far), will range from n to n + num_inserts
 * - seed = seed used for this run
 * - avg_node_size = average leaf node size recorded
 */

#include "experiments.h"

namespace
{
  // Input
  size_t eps = 64;
  size_t eps_rec = 8;
  size_t initial_n = 5e5;
  size_t num_inserts = 1e5;
  size_t granularity = 2e4;
  size_t num_reads = 1e3;

  std::vector<int> seeds = {1, 2, 3, 4, 5};
  size_t num_trials = 1;

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

void run_insert_metrics(std::string filename)
{
  std::vector<Configuration> configs = get_configs();
  // Parameters for the indices
  std::ofstream fout;
  fout.open(filename);
  fout << "name,data_size,seed,trial,tree_shape,read_profile,split_history" << std::endl;
  // Repeat the experiment for different seeds
  progressbar bar(num_trials * seeds.size() + 1);
  bar.update();
  for (int trial = 0; trial < num_trials; ++trial)
  {
    for (auto &seed : seeds)
    {
      auto do_write = [&](std::string name, size_t data_size, auto pgm)
      {
        fout
            << name
            << ","
            << data_size << ","
            << seed << ","
            << trial << ","
            << pgm.get_tree_shape().encode() << ","
            << pgm.read_profile.encode() << ","
            << pgm.split_history.encode() << ","
            << std::endl;
      };
      // Get the data
      auto data = get_random_data(initial_n, seed);
      auto inserts = get_random_inserts(num_inserts, granularity);

      for (auto &config : configs)
      {
        std::vector<std::pair<uint32_t, uint32_t>> growing_data = data;
        // The fixed_ makes it so that the in-place gets a boost so they start at same indexability
        size_t fixed_epsilon = (size_t)((float)eps / config.fill_ratio);
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

        auto reads = get_random_reads(data, num_reads);
        time_reads(pgm, reads);

        do_write(config.name, initial_n, pgm);

        size_t clean_size = initial_n;
        for (auto &insert_data : inserts)
        {
          do_inserts(pgm, insert_data);
          growing_data.insert(growing_data.end(), insert_data.begin(), insert_data.end());
          auto reads = get_random_reads(growing_data, num_reads);
          time_reads(pgm, reads);
          clean_size += insert_data.size();
          do_write(config.name, clean_size, pgm);
        }
      }

      bar.update();
    }
  }
  fout.close();
}