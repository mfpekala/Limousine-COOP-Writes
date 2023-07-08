/**
 * inserts_vs_wlatency.cpp
 *
 * This experiment measures insert latency as a function of the number of inserts
 * under a variety of configurations. We run this experiment for a variety of
 * configurations with a specified initial size and number of inserts. We also
 * compare the time required for inserts to the time required to construct a new model.
 *
 * Input:
 * - eps
 * - eps_rec
 * - initial_ns = list of the initial data sizes that we want to test on
 * - num_inserts = total number of inserts to perform for each initial n
 * - seeds = list of seeds to run this experiment on
 * - Configs
 *
 * Output:
 * - name = name of the configuration generating this row, or "baseline"
 * - data_size = the size of the model before inserts
 * - seed = seed used for this run
 * - time = ms required to perform the inserts
 */

#include "experiments.h"

namespace
{
  // Input
  size_t eps = 64;
  size_t eps_rec = 8;
  std::vector<size_t> initial_ns = {
      300000,
      1000000,
      3000000,
  };
  size_t num_inserts = 100000;
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

void run_inserts_vs_wlatency(std::string filename)
{
  std::vector<Configuration> configs = get_configs();
  // Parameters for the indices
  std::ofstream fout;
  fout.open(filename);
  fout << "name,data_size,seed,time" << std::endl;
  // Repeat the experiment for different seeds
  progressbar bar(seeds.size() * initial_ns.size() + 1);
  bar.update();
  for (auto &initial_n : initial_ns)
  {
    for (auto &seed : seeds)
    {
      // Get the data
      auto data = get_random_data(initial_n, seed);
      auto inserts = get_random_inserts(num_inserts, num_inserts)[0];

      for (auto &config : configs)
      {
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
        size_t result = time_inserts(pgm, inserts);
        fout << config.name << "," << initial_n << "," << seed << "," << result << std::endl;
      }

      // Baseline experiment
      auto start = std::chrono::high_resolution_clock::now();
      auto new_data = get_random_data(initial_n, seed);
      auto new_inserts = get_random_inserts(num_inserts, num_inserts)[0];
      new_data.insert(new_data.end(), new_inserts.begin(), new_inserts.end());
      auto baseline_pgm = pgm::OopPGMIndex<uint32_t, uint32_t>(new_data.begin(), new_data.end(), eps, eps_rec, 1.0, 1.0, 0, 0);
      auto end = std::chrono::high_resolution_clock::now();
      size_t result = std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count();
      fout << "baseline," << initial_n << "," << seed << "," << result << std::endl;
      bar.update();
    }
  }
  fout.close();
}