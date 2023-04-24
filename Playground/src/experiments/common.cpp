#include "experiments.h"

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

std::vector<std::vector<std::pair<uint32_t, uint32_t>>> get_random_inserts(size_t n, size_t granularity)
{
  if (n % granularity != 0)
  {
    throw std::runtime_error("n must be a multiple of granularity");
  }
  std::vector<std::vector<std::pair<uint32_t, uint32_t>>> result;
  for (size_t i = 0; i < n; i += granularity)
  {
    std::vector<std::pair<uint32_t, uint32_t>> data(granularity);
    std::generate(data.begin(), data.end(), []
                  { return std::make_pair(std::rand(), std::rand()); });
    result.push_back(data);
  }
  return result;
}

// Helper function to get the average segment size at the leaf level
size_t get_avg_leaf_size(pgm::BufferedPGMIndex<uint32_t, uint32_t> &buffered_pgm)
{
  size_t sum = 0;
  for (auto &seg : buffered_pgm.levels[0])
  {
    sum += seg.data.size();
  }
  // std::cout << "sum: " << sum << ", count: " << buffered_pgm.segments_count() << std::endl;
  return sum / buffered_pgm.segments_count();
}

std::pair<size_t, std::vector<size_t>> get_leaf_seg_size_histogram(pgm::BufferedPGMIndex<uint32_t, uint32_t> &buffered_pgm, size_t n_bins, size_t hist_max)
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

void do_inserts(pgm::BufferedPGMIndex<uint32_t, uint32_t> &buffered_pgm, std::vector<std::pair<uint32_t, uint32_t>> &insert_data)
{
  for (auto &p : insert_data)
  {
    buffered_pgm.insert(p.first, p.second);
  }
}

size_t time_inserts(pgm::BufferedPGMIndex<uint32_t, uint32_t> &buffered_pgm, std::vector<std::pair<uint32_t, uint32_t>> &insert_data)
{
  auto start = std::chrono::high_resolution_clock::now();
  do_inserts(buffered_pgm, insert_data);
  auto end = std::chrono::high_resolution_clock::now();
  return std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count();
}
