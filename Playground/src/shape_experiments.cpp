#include <vector>
#include <cstdlib>
#include <fstream>
#include <iostream>
#include <algorithm>
#include <string>
#include <chrono>
#include <thread>
#include "buffered/pgm_index_buffered.hpp"
#include "debug/progressbar.hpp"

std::vector<std::pair<uint32_t, uint32_t>> get_random_data(size_t n, int seed)
{
  // Generate some random key-value pairs to bulk-load the Dynamic PGM-index
  std::vector<std::pair<uint32_t, uint32_t>> data_raw(n);
  std::srand(seed);
  std::generate(data_raw.begin(), data_raw.end(), []
                { return std::make_pair(std::rand(), std::rand()); });
  std::sort(data_raw.begin(), data_raw.end());
  std::vector<std::pair<uint32_t, uint32_t>> data;
  for (auto &p : data_raw)
  {
    if (data.size() && data.back().first == p.first)
    {
      continue;
    }
    data.push_back(p);
  }
  return data;
}

/*
Given data and a number of inserts, return the number of segments before and after the inserts.
Returns a pair of (before, after).
*/
std::pair<size_t, size_t> get_number_of_segments(
    std::vector<std::pair<uint32_t, uint32_t>> data,
    size_t num_inserts,
    size_t epsilon,
    size_t epsilon_recursive,
    float fill_factor)
{
  // Generate some random key-value pairs to bulk-load the Dynamic PGM-index
  pgm::BufferedPGMIndex<uint32_t, uint32_t>
      buffered_pgm(data.begin(), data.end(), epsilon, epsilon_recursive, fill_factor, fill_factor, 0);

  size_t before_num = buffered_pgm.full_segments_count();

  // Do a bunch of inserts of random numbers
  for (int i = 0; i < num_inserts; i++)
  {
    auto q = std::rand();
    auto v = std::rand();
    buffered_pgm.insert(q, v);
  }

  size_t after_num = buffered_pgm.full_segments_count();

  return std::pair<size_t, size_t>(before_num, after_num);
}

// Helper function to get the average segment size at the leaf level
size_t get_avg_seg_size(pgm::BufferedPGMIndex<uint32_t, uint32_t> &buffered_pgm)
{
  size_t sum = 0;
  for (auto &seg : buffered_pgm.levels[0])
  {
    sum += seg.data.size();
  }
  return sum / buffered_pgm.segments_count();
}

/**
 * Helper function to get a output-friendly histogram of leaf segment sizes
 * @param buffered_pgm - The index to get the histogram for
 * @param n_bins - How many bins should the histogram have
 * @returns a pair (max_val, key_vals) where max_val is the maximum value in the histogram
 * and key_vals[i] is the number of values less than i/key_vals.size() * max_val
 */
std::pair<size_t, std::vector<size_t>> get_leaf_seg_size_histogram(pgm::BufferedPGMIndex<uint32_t, uint32_t> &buffered_pgm, size_t n_bins = 20, size_t hist_max = 30000)
{
  std::vector<size_t> key_vals(n_bins + 1, 0);
  for (auto &seg : buffered_pgm.levels[0])
  {
    size_t bin = (((double)seg.data.size() * n_bins) / (double)hist_max);
    if (bin > n_bins)
    {
      bin = n_bins;
    }
    key_vals[bin]++;
  }
  return std::pair<size_t, std::vector<size_t>>(hist_max, key_vals);
}

/*
An experiment that aims to test how the fill factor parameter affects the number of segments.
*/
void run_ff_count_experiment()
{
  std::vector<float> of_interest = {0.1,
                                    0.2,
                                    0.3,
                                    0.4,
                                    0.5,
                                    0.6,
                                    0.7,
                                    0.8,
                                    0.9};
  std::vector<int> seeds = {1, 2, 3, 4, 5};
  size_t n = 100000;
  size_t num_inserts = 50000;
  size_t epsilon = 64;
  size_t epsilon_recursive = 4;

  std::ofstream fout;
  fout.open("fill_factor.csv");
  fout << "ff,before,after" << std::endl;
  for (auto ff : of_interest)
  {
    size_t before_sum = 0;
    size_t after_sum = 0;
    for (auto seed : seeds)
    {
      auto data = get_random_data(n, seed);
      auto [before, after] = get_number_of_segments(data, num_inserts, epsilon, epsilon_recursive, ff);
      before_sum += before;
      after_sum += after;
    }
    before_sum /= seeds.size();
    after_sum /= seeds.size();
    std::cout
        << "ff: " << ff << ", num segs: " << before_sum << "," << after_sum << std::endl;
    fout << ff << "," << before_sum << "," << after_sum << std::endl;
  }
  fout.close();
}

// Helper function to hide the logic of actually doing the inserts for testing purposes
void do_inserts(pgm::BufferedPGMIndex<uint32_t, uint32_t> &buffered_pgm, size_t num_inserts)
{
  for (int i = 0; i < num_inserts; i++)
  {
    auto q = std::rand();
    auto v = std::rand();
    buffered_pgm.insert(q, v);
  }
}

/**
 * Helper function to do a run of inserts vs size for a given index
 * @param n - How big should the index be before we start inserting
 * @param num_inserts - How many inserts should we do
 * @param granularity - How many inserts should we do at a time
 */
std::vector<std::pair<size_t, size_t>> do_inserts_vs_size_run(
    pgm::BufferedPGMIndex<uint32_t, uint32_t> &buffered_pgm,
    size_t num_inserts,
    size_t granularity)
{
  progressbar bar(num_inserts / granularity);
  std::vector<std::pair<size_t, size_t>> results;
  for (int inserts = 0; inserts < num_inserts; inserts += granularity)
  {
    do_inserts(buffered_pgm, granularity);
    results.push_back(std::pair<size_t, size_t>(inserts + granularity, get_avg_seg_size(buffered_pgm)));
    bar.update();
  }
  return results;
}

/**
 * We want to be able to compare the numbers from our models with the numbers
 * from a model trained straight from the data.
 */
std::vector<std::pair<size_t, size_t>> do_baseline_inserts_vs_size_run(
    std::vector<std::pair<uint32_t, uint32_t>> &base_data,
    size_t epsilon,
    size_t epsilon_recursive,
    float fill_factor,
    size_t max_buffer_size,
    size_t num_inserts,
    size_t granularity)
{
  progressbar bar(num_inserts / granularity);
  std::vector<std::pair<size_t, size_t>> results;
  for (int inserts = 0; inserts < num_inserts; inserts += granularity)
  {
    for (int i = 0; i < granularity; i++)
    {
      base_data.push_back(std::pair<uint32_t, uint32_t>(std::rand(), std::rand()));
    }
    auto baseline_pgm = pgm::BufferedPGMIndex<uint32_t, uint32_t>(
        base_data.begin(), base_data.end(), epsilon, epsilon_recursive, 1.0, 1.0, max_buffer_size);
    results.push_back(std::pair<size_t, size_t>(inserts + granularity, get_avg_seg_size(baseline_pgm)));
    bar.update();
  }
  return results;
}

/**
 * Helper function to do a run of inserts vs size for a given index for histogram data
 * @param n - How big should the index be before we start inserting
 * @param num_inserts - How many inserts should we do
 * @param granularity - How many inserts should we do at a time
 */
std::vector<std::pair<size_t, std::pair<size_t, std::vector<size_t>>>> do_hist_inserts_vs_size_run(
    pgm::BufferedPGMIndex<uint32_t, uint32_t> &buffered_pgm,
    size_t num_inserts,
    size_t granularity,
    size_t n_bins = 20,
    size_t hist_max = 30000)
{
  progressbar bar(num_inserts / granularity);
  std::vector<std::pair<size_t, std::pair<size_t, std::vector<size_t>>>> results;
  for (int inserts = 0; inserts < num_inserts; inserts += granularity)
  {
    do_inserts(buffered_pgm, granularity);
    std::pair<size_t, std::vector<size_t>> hist = get_leaf_seg_size_histogram(buffered_pgm, n_bins, hist_max);
    results.push_back(std::pair<size_t, std::pair<size_t, std::vector<size_t>>>(
        inserts + granularity, hist));
    bar.update();
  }
  return results;
}

/**
 * An experiment that aims to test how the number of inserts affects the size of the segments.
 * The idea is to compare in-place to out-of-place. We want to show that pure in-place
 * devolves to b-trees must faster than pure-out-of place.
 */
void run_inserts_vs_size_experiment(std::string filename)
{
  // Parameters for the indices
  size_t start_epsilon = 64;
  size_t start_epsilon_recursive = 8;
  float fill_factor = 0.5;
  size_t max_buffer_size = 512;
  // Experiment meta-knobs
  std::vector<int> seeds = {1, 2, 3};
  size_t n = 500000;
  size_t n_inserts = 100000;
  size_t granularity = 500;
  std::ofstream fout;
  fout.open(filename);
  fout << "type,seed,num_inserts,seg_size" << std::endl;
  // Repeat the experiment for different seeds
  // progressbar bar(seeds.size() * 2);
  for (auto &seed : seeds)
  {
    std::cout << "Seed " << seed << std::endl;
    // Get the data
    auto data = get_random_data(n, seed);
    auto in_place_pgm = pgm::BufferedPGMIndex<uint32_t, uint32_t>(
        data.begin(), data.end(), start_epsilon * 2, start_epsilon_recursive * 2, fill_factor, fill_factor, 0);
    auto out_of_place_pgm = pgm::BufferedPGMIndex<uint32_t, uint32_t>(
        data.begin(), data.end(), start_epsilon, start_epsilon_recursive, 1.0, 1.0, max_buffer_size);
    // bar.update();
    // In-place experiment
    fout << "in_place," << seed << "," << 0 << "," << get_avg_seg_size(in_place_pgm) << std::endl;
    auto in_place_results = do_inserts_vs_size_run(in_place_pgm, n_inserts, granularity);
    for (auto &result : in_place_results)
    {
      fout << "in_place," << seed << "," << result.first << "," << result.second << std::endl;
    }
    // Out-of-place experiment
    fout << "out_of_place," << seed << "," << 0 << "," << get_avg_seg_size(out_of_place_pgm) << std::endl;
    auto out_of_place_results = do_inserts_vs_size_run(out_of_place_pgm, n_inserts, granularity);
    for (auto &result : out_of_place_results)
    {
      fout << "out_of_place," << seed << "," << result.first << "," << result.second << std::endl;
    }
    // Baseline experiment
    fout << "baseline," << seed << "," << 0 << "," << get_avg_seg_size(out_of_place_pgm) << std::endl;
    auto baseline_results = do_baseline_inserts_vs_size_run(data, start_epsilon, start_epsilon_recursive, 1.0, max_buffer_size, n_inserts, granularity);
    for (auto &result : baseline_results)
    {
      fout << "baseline," << seed << "," << result.first << "," << result.second << std::endl;
    }
    std::cout << std::endl;
    // bar.update();
  }
  fout.close();
}

/**
 * An experiment that aims to test how the number of inserts affects the size of nodes.
 * Similar to the above experiment but the output is the full distribution of node sizes.
 * Hopefully will capture more information and gain deeper insights into how the data
 * structure is adaptively responding to inputs.
 */
void run_histogram_inserts_vs_size_experiment(std::string outfile)
{
  // Parameters for the indices
  size_t start_epsilon = 32;
  size_t start_epsilon_recursive = 8;
  float fill_factor = 0.5;
  size_t max_buffer_size = 128;
  // Experiment meta-knobs
  std::vector<int> seeds = {6};
  size_t n = 1000000;
  size_t n_inserts = 100000;
  size_t granularity = 1000;
  size_t n_bins = 50;
  size_t hist_max = 30000; // Manually adjust
  std::ofstream fout;
  fout.open(outfile);
  fout << "type,seed,num_inserts,max_val,";
  for (int i = 0; i < n_bins; i++)
  {
    fout << "bin_" << i;
    if (i < n_bins - 1)
    {
      fout << ",";
    }
    else
    {
      fout << std::endl;
    }
  }

  auto write_results = [&](std::string type, int seed, size_t num_inserts, std::pair<size_t, std::vector<size_t>> &results)
  {
    fout << type << "," << seed << "," << num_inserts << "," << results.first << ",";
    for (int i = 0; i <= n_bins; i++)
    {
      fout << results.second[i];
      if (i < n_bins)
      {
        fout << ",";
      }
      else
      {
        fout << std::endl;
      }
    }
  };

  // Repeat the experiment for different seeds
  // progressbar bar(seeds.size() * 2);
  for (auto &seed : seeds)
  {
    std::cout << "Seed " << seed << std::endl;
    // Get the data
    auto data = get_random_data(n, seed);
    auto in_place_pgm = pgm::BufferedPGMIndex<uint32_t, uint32_t>(
        data.begin(), data.end(), start_epsilon * 2, start_epsilon_recursive * 2, fill_factor, fill_factor, 0);
    auto out_of_place_pgm = pgm::BufferedPGMIndex<uint32_t, uint32_t>(
        data.begin(), data.end(), start_epsilon, start_epsilon_recursive, 1.0, 1.0, max_buffer_size);
    // bar.update();
    // In-place experiment
    std::pair<size_t, std::vector<size_t>> in_place_results = get_leaf_seg_size_histogram(in_place_pgm, n_bins, hist_max);
    write_results("in_place", seed, 0, in_place_results);
    auto many_in_place_results = do_hist_inserts_vs_size_run(in_place_pgm, n_inserts, granularity, n_bins);
    for (auto &result : many_in_place_results)
    {
      write_results("in_place", seed, result.first, result.second);
    }
    // Out-of-place experiment
    std::pair<size_t, std::vector<size_t>> out_of_place_results = get_leaf_seg_size_histogram(out_of_place_pgm, n_bins, hist_max);
    write_results("out_of_place", seed, 0, out_of_place_results);
    auto many_out_of_place_results = do_hist_inserts_vs_size_run(out_of_place_pgm, n_inserts, granularity, n_bins);
    for (auto &result : many_out_of_place_results)
    {
      write_results("out_of_place", seed, result.first, result.second);
    }
  }
  fout.close();
}

size_t time_inserts(pgm::BufferedPGMIndex<uint32_t, uint32_t> &pgm, size_t n_inserts)
{
  auto start = std::chrono::high_resolution_clock::now();
  for (int i = 0; i < n_inserts; i++)
  {
    auto q = std::rand();
    auto v = std::rand();
    pgm.insert(q, v);
  }
  auto end = std::chrono::high_resolution_clock::now();
  return std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count();
}

void run_insert_time_experiment(std::string outfile)
{
  // Parameters for the indices
  size_t start_epsilon = 64;
  size_t start_epsilon_recursive = 16;
  float fill_factor = 0.5;
  size_t max_buffer_size = 512;
  // Experiment meta-knobs
  std::vector<int> seeds = {1, 2, 3, 4, 5};
  size_t n = 1000000;
  size_t n_inserts = 100000;
  size_t granularity = 1000;
  std::ofstream fout;
  fout.open(outfile);
  fout << "type,seed,num_inserts,time" << std::endl;
  for (auto &seed : seeds)
  {
    std::cout << "Seed " << seed << std::endl;
    // Get the data
    auto data = get_random_data(n, seed);
    auto in_place_pgm = pgm::BufferedPGMIndex<uint32_t, uint32_t>(
        data.begin(), data.end(), start_epsilon * 2, start_epsilon_recursive * 2, fill_factor, fill_factor, 0);
    auto out_of_place_pgm = pgm::BufferedPGMIndex<uint32_t, uint32_t>(
        data.begin(), data.end(), start_epsilon, start_epsilon_recursive, 1.0, 1.0, max_buffer_size);
    // In-place experiment
    fout << "in_place," << seed << "," << 0 << "," << time_inserts(in_place_pgm, 0) << std::endl;
    for (int inserts = granularity; inserts <= n_inserts; inserts += granularity)
    {
      fout << "in_place," << seed << "," << inserts << "," << time_inserts(in_place_pgm, granularity) << std::endl;
    }
    // Out-of-place experiment
    fout << "out_of_place," << seed << "," << 0 << "," << time_inserts(out_of_place_pgm, 0) << std::endl;
    for (int inserts = granularity; inserts <= n_inserts; inserts += granularity)
    {
      fout << "out_of_place," << seed << "," << inserts << "," << time_inserts(out_of_place_pgm, granularity) << std::endl;
    }
  }
}

void run_insert_time_magnitude_experiment(std::string outfile)
{
  // Parameters for the indices
  size_t start_epsilon = 64;
  size_t start_epsilon_recursive = 16;
  float fill_factor = 0.5;
  size_t max_buffer_size = 512;
  // Experiment meta-knobs
  std::vector<int> seeds = {1, 2, 3};
  std::vector<int> ns = {1000, 10000, 100000, 300000, 1000000, 3000000, 10000000, 30000000, 100000000};
  size_t n_inserts = 10000;
  std::ofstream fout;
  fout.open(outfile);
  fout << "type,seed,data_size,time" << std::endl;
  for (auto &seed : seeds)
  {
    std::cout << "Seed " << seed << std::endl;
    for (auto &n : ns)
    {
      std::cout << "n = " << n << std::endl;
      // Get the data
      auto data = get_random_data(n, seed);
      auto in_place_pgm = pgm::BufferedPGMIndex<uint32_t, uint32_t>(
          data.begin(), data.end(), start_epsilon * 2, start_epsilon_recursive * 2, fill_factor, fill_factor, 0);
      auto out_of_place_pgm = pgm::BufferedPGMIndex<uint32_t, uint32_t>(
          data.begin(), data.end(), start_epsilon, start_epsilon_recursive, 1.0, 1.0, max_buffer_size);
      // In-place experiment
      auto time = time_inserts(in_place_pgm, n_inserts);
      fout << "in_place," << seed << "," << n << "," << time << std::endl;
      // Out-of-place experiment
      time = time_inserts(out_of_place_pgm, n_inserts);
      fout << "out_of_place," << seed << "," << n << "," << time << std::endl;
    }
  }
}

void run_height_vs_inserts_experiment()
{
  size_t n = 100000;
  size_t num_inserts = 1000000;
  size_t granularity = 10000;
  size_t epsilon = 64;
  size_t epsilon_recursive = 4;
  float ff = 0.5;

  auto data = get_random_data(n, 1);
  pgm::BufferedPGMIndex<uint32_t, uint32_t>
      buffered_pgm(data.begin(), data.end(), epsilon, epsilon_recursive, ff);

  std::ofstream fout;
  fout.open("height_vs_inserts.csv");
  fout << "num_inserts,height" << std::endl;
  for (int inserts = 0; inserts < num_inserts; inserts += granularity)
  {
    for (int i = 0; i < granularity; i++)
    {
      auto q = std::rand();
      auto v = std::rand();
      buffered_pgm.insert(q, v);
    }
    size_t height = buffered_pgm.height();
    std::cout
        << "num_inserts: " << inserts << ", height: " << height << std::endl;
    fout << inserts << "," << height << std::endl;
  }
  fout.close();
}

void pareto_experiment()
{
  size_t n = 10000;
  size_t num_inserts = 10000;
  size_t granulatiry = 1000;
  size_t epsilon = 64;
  size_t epsilon_recursive = 8;
  std::vector<float> ffs = {
      0.25,
      0.5,
      0.75,
  };
  std::vector<size_t> buf_sizes = {
      16,
      64,
      128,
      256};
  /*
  for (auto ff : ffs)
  {
    for (auto buf_size : buf_sizes)
    {
      auto data = get_random_data(n, 1);
      auto index = pgm::BufferedPGMIndex<uint32_t, uint32_t>();
      index.
      for (int inserts = 0; inserts < num_inserts; inserts += granularity)
      {
        for (int i = 0; i < granularity; i++)
        {
          auto q = std::rand();
          auto v = std::rand();
          buffered_pgm.insert(q, v);
        }
        size_t avg_seg_size = get_avg_seg_size(data);
      }
    }
  }
  */
}

int main()
{
  run_inserts_vs_size_experiment("inserts_vs_baseline.csv");
}
