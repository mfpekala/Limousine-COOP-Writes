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
 * - n = initial data size
 * - num_inserts = total number of inserts to perform
 * - granularity = how many inserts to perform at a time up to num_inserts
 * - num_reads = the number of reads to profile at every granularity
 * - seeds = list of seeds to run this experiment on
 * - repeats = how many times to repeat each seed
 * - List of configurations, where each configuration specifies:
 *    - name = column name in output
 *    - fill_ratio
 *    - fill_ratio_recursive
 *    - max_buffer_size
 *    - split_neighborhood
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
  size_t epsilon = 128;
  size_t epsilon_recursive = 16;
  size_t n = 1000000;
  size_t num_inserts = 1000000;
  size_t granularity = 100000;
  size_t num_reads = 1000;
  std::vector<int> seeds = {1, 2, 3};
  size_t repeats = 1;

  struct Configuration
  {
    std::string name;
    float fill_ratio;
    float fill_ratio_recursive;
    size_t max_buffer_size;
    size_t split_neighborhood;
  };

  std::vector<Configuration> get_configs()
  {
    Configuration base_in_place;
    base_in_place.name = "in_place";
    base_in_place.fill_ratio = 0.75;
    base_in_place.fill_ratio_recursive = 0.75;
    base_in_place.max_buffer_size = 0;
    base_in_place.split_neighborhood = 0;

    /*
    TOO SLOW TO TEST
    Configuration in_place_n0 = base_in_place;
    in_place_n0.name = "in_place_n0";
    in_place_n0.split_neighborhood = 0;

    Configuration in_place_n2 = base_in_place;
    in_place_n2.name = "in_place_n2";
    in_place_n2.split_neighborhood = 2;

    Configuration out_place_n8 = base_in_place;
    out_place_n8.name = "in_place_n8";
    out_place_n8.split_neighborhood = 8;
    */

    Configuration base_out_place;
    base_out_place.name = "out_place";
    base_out_place.fill_ratio = 1.0;
    base_out_place.fill_ratio_recursive = 1.0;
    base_out_place.max_buffer_size = 128;
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
        /*
        TOO SLOW TO TEST
        in_place_n0,
        in_place_n2,
        in_place_n8,
        */
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
  fout << "name,data_size,seed,run,tree_shape,read_profile,split_history" << std::endl;
  // Repeat the experiment for different seeds
  progressbar bar(repeats * seeds.size() + 1);
  bar.update();
  for (int run = 0; run < repeats; ++run)
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
            << run << ","
            << pgm.get_tree_shape().encode() << ","
            << pgm.read_profile.encode() << ","
            << pgm.split_history.encode() << ","
            << std::endl;
      };
      // Get the data
      auto data = get_random_data(n, seed);
      auto inserts = get_random_inserts(num_inserts, granularity);

      for (auto &config : configs)
      {
        std::vector<std::pair<uint32_t, uint32_t>> growing_data = data;
        size_t fixed_epsilon = (size_t)((float)epsilon / config.fill_ratio);
        size_t fixed_epsilon_recursive = (size_t)((float)epsilon_recursive / config.fill_ratio_recursive);
        auto pgm = pgm::BufferedPGMIndex<uint32_t, uint32_t>(
            data.begin(),
            data.end(),
            fixed_epsilon,
            fixed_epsilon_recursive,
            config.fill_ratio,
            config.fill_ratio_recursive,
            config.max_buffer_size,
            config.split_neighborhood);

        auto reads = get_random_reads(data, num_reads);
        time_reads(pgm, reads);

        do_write(config.name, n, pgm);
        // pgm.reset_metrics(); // TODO: Design choice, reset per batch now or go back and do the calculation later

        size_t clean_size = n;
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

      /*
      TODO: What does this mean in the baseline case?
      // Baseline experiment
      auto first_pgm = pgm::BufferedPGMIndex<uint32_t, uint32_t>(data.begin(), data.end(), epsilon, epsilon_recursive, 1.0, 1.0, 0, 0);
      fout << "baseline," << n << "," << seed << "," << get_avg_leaf_size(first_pgm) << std::endl;
      size_t clean_size = n;
      for (auto &insert_data : inserts)
      {
        clean_size += insert_data.size();
        data.insert(data.end(), insert_data.begin(), insert_data.end());
        auto baseline_pgm = pgm::BufferedPGMIndex<uint32_t, uint32_t>(data.begin(), data.end(), epsilon, epsilon_recursive, 1.0, 1.0, 0, 0);
        fout << "baseline," << clean_size << "," << seed << "," << get_avg_leaf_size(baseline_pgm) << std::endl;
      }
      */

      bar.update();
    }
  }
  fout.close();
}