/**
 * inserts_vs_index_power
 *
 * This experiment studies how the structure of the model changes as more
 * and more inserts are received. Specifically, we are interested in looking
 * at the average size of a leaf node as a measure of how powerful the structure
 * is. We run this experiment for a variety of configurations with a specified
 * initial size and number of inserts. We also compare the indexing power to a
 * naive model which is trained on the initial data + inserts to see how our
 * model degrades compared to the baseline.
 *
 * Input:
 * - Epsilon
 * - Epsilon recursive
 * - n = initial data size
 * - num_inserts = total number of inserts to perform
 * - granularity = how many inserts to perform at a time up to num_inserts
 * - seeds = list of seeds to run this experiment on
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
  size_t granularity = 10000;
  std::vector<int> seeds = {1, 2, 3};

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
    base_out_place.max_buffer_size = 512;
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

void run_inserts_vs_index_power(std::string filename)
{
  std::vector<Configuration> configs = get_configs();
  // Parameters for the indices
  std::ofstream fout;
  fout.open(filename);
  fout << "name,data_size,seed,avg_node_size" << std::endl;
  // Repeat the experiment for different seeds
  progressbar bar(seeds.size() + 1);
  bar.update();
  for (auto &seed : seeds)
  {
    // Get the data
    auto data = get_random_data(n, seed);
    auto inserts = get_random_inserts(num_inserts, granularity);

    for (auto &config : configs)
    {
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
      fout << config.name << "," << n << "," << seed << "," << get_avg_leaf_size(pgm) << std::endl;
      size_t size = n;
      for (auto &insert_data : inserts)
      {
        do_inserts(pgm, insert_data);
        size += insert_data.size();
        fout << config.name << "," << size << "," << seed << "," << get_avg_leaf_size(pgm) << std::endl;
      }
    }

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
    bar.update();
  }
  fout.close();
}