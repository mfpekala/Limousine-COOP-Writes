#include <vector>
#include <cstdlib>
#include <fstream>
#include <iostream>
#include <algorithm>
#include <string>
#include "buffered/pgm_index_buffered.hpp"

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

// returns (before_num, after_num)
std::pair<size_t, size_t> get_number_of_segments(
    std::vector<std::pair<uint32_t, uint32_t>> data,
    size_t num_inserts,
    size_t epsilon,
    size_t epsilon_recursive,
    float fill_factor)
{
  // Generate some random key-value pairs to bulk-load the Dynamic PGM-index
  pgm::BufferedPGMIndex<uint32_t, uint32_t>
      buffered_pgm(epsilon, epsilon_recursive, fill_factor, data.begin(), data.end());

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

int main()
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
  size_t num_inserts = 5000;
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
