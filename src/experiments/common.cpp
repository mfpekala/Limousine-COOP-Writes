/**
 * Definitions for common helper functions for experiments.
 */

#include <assert.h>

#include "experiments.h"
#include "zipfian.hpp"

std::vector<std::pair<uint32_t, uint32_t>> get_random_data(size_t n, int seed) {
  // Generate some random key-value pairs to bulk-load the Dynamic PGM-index
  std::vector<std::pair<uint32_t, uint32_t>> data_raw(n);
  std::srand(seed);
  std::generate(data_raw.begin(), data_raw.end(),
                [] { return std::make_pair(std::rand(), std::rand()); });
  std::sort(data_raw.begin(), data_raw.end());
  std::vector<std::pair<uint32_t, uint32_t>> data;
  for (auto &p : data_raw) {
    if (data.size() && data.back().first == p.first) {
      continue;
    }
    data.push_back(p);
  }
  return data;
}

std::vector<std::vector<std::pair<uint32_t, uint32_t>>> get_random_inserts(
    size_t n, size_t granularity) {
  if (n % granularity != 0) {
    throw std::runtime_error("n must be a multiple of granularity");
  }
  std::vector<std::vector<std::pair<uint32_t, uint32_t>>> result;
  for (size_t i = 0; i < n; i += granularity) {
    std::vector<std::pair<uint32_t, uint32_t>> data(granularity);
    std::generate(data.begin(), data.end(),
                  [] { return std::make_pair(std::rand(), std::rand()); });
    result.push_back(data);
  }
  return result;
}

std::vector<uint32_t> get_random_reads(
    std::vector<std::pair<uint32_t, uint32_t>> data, size_t num_reads) {
  // Pick num_reads random elements from data
  std::vector<uint32_t> result;
  for (size_t i = 0; i < num_reads; i++) {
    size_t ix = std::rand() % data.size();
    auto val = data[ix].first;
    result.push_back(val);
  }
  return result;
}

// Helper function to get the average segment size at the leaf level
size_t get_avg_leaf_size(pgm::CoopPGMIndex<uint32_t, uint32_t> &buffered_pgm) {
  size_t sum = 0;
  for (auto &model : buffered_pgm.model_tree[0]) {
    sum += model.n;
  }
  return sum / buffered_pgm.model_tree[0].size();
}

std::pair<size_t, std::vector<size_t>> get_leaf_seg_size_histogram(
    pgm::CoopPGMIndex<uint32_t, uint32_t> &buffered_pgm, size_t n_bins,
    size_t hist_max) {
  std::vector<size_t> key_vals(n_bins + 1, 0);
  for (auto &model : buffered_pgm.model_tree[0]) {
    size_t bin = (((double)model.n * n_bins) / (double)hist_max);
    if (bin > n_bins) {
      bin = n_bins;
    }
    key_vals[bin]++;
  }
  return std::pair<size_t, std::vector<size_t>>(hist_max, key_vals);
}

void do_inserts(pgm::CoopPGMIndex<uint32_t, uint32_t> &buffered_pgm,
                std::vector<std::pair<uint32_t, uint32_t>> &insert_data) {
  for (auto &p : insert_data) {
    buffered_pgm.insert(p.first, p.second);
  }
}

size_t time_inserts(pgm::CoopPGMIndex<uint32_t, uint32_t> &buffered_pgm,
                    std::vector<std::pair<uint32_t, uint32_t>> &insert_data) {
  auto start = std::chrono::high_resolution_clock::now();
  do_inserts(buffered_pgm, insert_data);
  auto end = std::chrono::high_resolution_clock::now();
  return std::chrono::duration_cast<std::chrono::milliseconds>(end - start)
      .count();
}

size_t time_reads(pgm::CoopPGMIndex<uint32_t, uint32_t> &buffered_pgm,
                  std::vector<uint32_t> &keys) {
  auto start = std::chrono::high_resolution_clock::now();
  for (auto &key : keys) {
    buffered_pgm.find(key);
  }
  auto end = std::chrono::high_resolution_clock::now();
  return std::chrono::duration_cast<std::chrono::milliseconds>(end - start)
      .count();
}

Workload generate_workload(std::string name, size_t initial_n,
                           float prop_writes, size_t num_ops, int seed) {
  // Setup randomness for prop_writes
  std::random_device rd;
  std::mt19937 gen(rd());
  std::uniform_real_distribution<float> dis(0.0, 1.0);

  Workload result;
  result.name = name;
  result.initial_data = get_random_data(initial_n, seed);
  result.ops = std::vector<Op>(num_ops);
  // auto valid_reads = initial_data;
  for (int ix = 0; ix < num_ops; ++ix) {
    bool is_write = dis(gen) < prop_writes;
    if (is_write) {
      Op new_op;
      new_op.type = WRITE;
      new_op.key = std::rand();
      new_op.val = std::rand();
      result.ops.push_back(new_op);
    } else {
      Op new_op;
      new_op.type = READ;
      new_op.key =
          result.initial_data[std::rand() % result.initial_data.size()].first;
      new_op.val = 0;  // Arbitrary
      result.ops.push_back(new_op);
    }
  }
  return result;
}

// From
// https://stackoverflow.com/questions/9983239/how-to-generate-zipf-distributed-numbers-efficiently
static bool first = true;  // Static first time flag
uint32_t zipf(double alpha, uint32_t N) {
  static double c = 0.0;         // Normalization constant
  static double *sum_probs;      // Pre-calculated sum of probabilities
  double z;                      // Uniform random number (0 < z < 1)
  uint32_t zipf_value;           // Computed exponential value to be returned
  uint32_t i;                    // Loop counter
  uint_fast32_t low, high, mid;  // Binary-search bounds

  // Setup randomness
  std::random_device rd;
  std::mt19937 gen(rd());
  std::uniform_real_distribution<float> dis(0.0, 1.0);

  // Compute normalization constant on first call only
  if (first == true) {
    for (i = 1; i <= N; i++) {
      c = c + (1.0 / pow((double)i, alpha));
    }
    c = 1.0 / c;

    sum_probs = (double *)malloc((N + 1) * sizeof(*sum_probs));
    sum_probs[0] = 0;
    for (i = 1; i <= N; i++) {
      sum_probs[i] = sum_probs[i - 1] + c / pow((double)i, alpha);
    }
    first = false;
  }

  // Pull a uniform random number (0 < z < 1)
  do {
    z = dis(gen);
  } while ((z == 0) || (z == 1));

  // Map z to the value
  low = 1, high = N, mid;
  do {
    mid = floor((low + high) / 2);
    if (sum_probs[mid] >= z && sum_probs[mid - 1] < z) {
      zipf_value = mid;
      break;
    } else if (sum_probs[mid] >= z) {
      high = mid - 1;
    } else {
      low = mid + 1;
    }
  } while (low <= high);

  // Assert that zipf_value is between 1 and N
  assert((zipf_value >= 1) && (zipf_value <= N));

  return (zipf_value);
}

std::vector<std::pair<uint32_t, uint32_t>> get_random_zipfian_data(
    size_t n, uint32_t N, float alpha) {
  // Generate some random key-value pairs to bulk-load the Dynamic PGM-index
  std::vector<std::pair<uint32_t, uint32_t>> data_raw(n);
  std::generate(data_raw.begin(), data_raw.end(), [N, alpha] {
    return std::make_pair(zipf(alpha, N), std::rand());
  });
  std::sort(data_raw.begin(), data_raw.end());
  std::vector<std::pair<uint32_t, uint32_t>> data;
  for (auto &p : data_raw) {
    // Get rid of duplicates for efficiency
    if (data.size() && data.back().first == p.first) {
      continue;
    }
    data.push_back(p);
  }
  return data;
}

std::vector<std::pair<uint32_t, uint32_t>> get_random_zipfian_data2(
    size_t n, uint32_t N, float alpha) {
  std::vector<std::pair<uint32_t, uint32_t>> data_raw(n);
  std::default_random_engine generator;
  zipfian_int_distribution<uint32_t>::param_type p(1, 1e9, 0.9, 27.000);
  zipfian_int_distribution<uint32_t> distribution(p);
  // Generate some random key-value pairs to bulk-load the Dynamic PGM-index
  for (int ix = 0; ix < n; ++ix) {
    uint32_t i = distribution(generator);
    data_raw[ix] = std::make_pair(i, std::rand());
  }
  std::sort(data_raw.begin(), data_raw.end());
  std::vector<std::pair<uint32_t, uint32_t>> data;
  for (auto &p : data_raw) {
    // Get rid of duplicates for efficiency
    if (data.size() && data.back().first == p.first) {
      continue;
    }
    data.push_back(p);
  }
  return data;
}

Workload generate_zipfian_workload(std::string name, size_t initial_n,
                                   float prop_writes, size_t num_ops,
                                   float alpha) {
  // Setup randomness for prop_writes
  std::random_device rd;
  std::mt19937 gen(rd());
  std::uniform_real_distribution<float> dis(0.0, 1.0);

  Workload result;
  result.name = name;
  const uint32_t N = 1e9;  // One hundred million
  // result.initial_data = get_random_zipfian_data(initial_n, N, alpha);
  result.initial_data = get_random_zipfian_data(initial_n, N, alpha);
  result.ops = std::vector<Op>(num_ops);
  for (int ix = 0; ix < num_ops; ++ix) {
    bool is_write = dis(gen) < prop_writes;
    if (is_write) {
      Op new_op;
      new_op.type = WRITE;
      new_op.key = zipf(alpha, N);  // Has a chance to insert new data or access
                                    // old data, depending on alpha
      new_op.val = std::rand();
      result.ops.push_back(new_op);
    } else {
      Op new_op;
      new_op.type = READ;
      new_op.key =  // Still correct to index randomly, as we want any initial
                    // element to have equal chance of being read
          result.initial_data[std::rand() % result.initial_data.size()].first;
      new_op.val = 0;  // Arbitrary
      result.ops.push_back(new_op);
    }
  }
  return result;
}

std::tuple<size_t, size_t, pgm::CoopPGMIndex<uint32_t, uint32_t>>
benchmark_workload_config(Workload &workload, Configuration &config) {
  auto pgm = pgm::CoopPGMIndex<uint32_t, uint32_t>(
      workload.initial_data.begin(), workload.initial_data.end(), config.eps,
      config.eps_rec, config.fill_ratio, config.fill_ratio_rec,
      config.buffer_size, config.split_neighborhood);

  auto start = std::chrono::high_resolution_clock::now();
  for (auto &op : workload.ops) {
    if (op.type == WRITE) {
      pgm.insert(op.key, op.val);
    } else {
      auto test = pgm.find(op.key);
    }
  }
  auto end = std::chrono::high_resolution_clock::now();

  size_t time =
      std::chrono::duration_cast<std::chrono::milliseconds>(end - start)
          .count();
  size_t mem = pgm.size_in_bytes();
  return std::make_tuple(time, mem, pgm);
}

std::tuple<size_t, size_t, size_t, pgm::CoopPGMIndex<uint32_t, uint32_t>>
lspecific_benchmark_workload_config(Workload &workload, Configuration &config) {
  auto pgm = pgm::CoopPGMIndex<uint32_t, uint32_t>(
      workload.initial_data.begin(), workload.initial_data.end(), config.eps,
      config.eps_rec, config.fill_ratio, config.fill_ratio_rec,
      config.buffer_size, config.split_neighborhood);

  double rtime = 0;
  double wtime = 0;
  // In order to geed needed precision, nanoseconds are needed below. However,
  // this would overflow. So, we'll instead add to an accumulator, and every
  // 10000 operations will divide this by 1e6 to get milliseconds and add to
  // sum.
  size_t acc = 0;
  size_t ACC_FREQ = 100000;
  size_t rtime_acc = 0;
  size_t wtime_acc = 0;
  auto start = std::chrono::high_resolution_clock::now();
  auto end = std::chrono::high_resolution_clock::now();
  for (auto &op : workload.ops) {
    acc++;
    if (op.type == WRITE) {
      start = std::chrono::high_resolution_clock::now();
      pgm.insert(op.key, op.val);
      end = std::chrono::high_resolution_clock::now();
      wtime_acc +=
          std::chrono::duration_cast<std::chrono::nanoseconds>(end - start)
              .count();
    } else {
      start = std::chrono::high_resolution_clock::now();
      auto test = pgm.find(op.key);
      end = std::chrono::high_resolution_clock::now();
      rtime_acc +=
          std::chrono::duration_cast<std::chrono::nanoseconds>(end - start)
              .count();
    }
    if (acc % ACC_FREQ == 0) {
      rtime += rtime_acc / 1e6;
      wtime += wtime_acc / 1e6;
      rtime_acc = 0;
      wtime_acc = 0;
    }
  }
  rtime += rtime_acc / 1e6;
  wtime += wtime_acc / 1e6;
  size_t mem = pgm.size_in_bytes();
  return std::make_tuple(rtime, wtime, mem, pgm);
}
