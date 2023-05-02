#include <vector>
#include <cstdlib>
#include <fstream>
#include <iostream>
#include <algorithm>
#include <string>
#include <chrono>
#include <thread>
#include <random>
#include "buffered/pgm_index_buffered.hpp"
#include "debug/progressbar.hpp"

/* HELPFUL DATA STRUCTURES */

struct Configuration
{
  std::string name;
  float fill_ratio;
  float fill_ratio_recursive;
  size_t max_buffer_size;
  size_t split_neighborhood;
};

enum OpType
{
  READ = 0,
  WRITE = 1
};

struct Op
{
  OpType type;
  uint32_t key;
  uint32_t val;
};

struct Workload
{
  std::string name;
  std::vector<std::pair<uint32_t, uint32_t>> initial_data;
  std::vector<Op> ops;
};

/* HELPER FUNCTIONS */

/**
 * Returns uniformly random data of size n with given seed
 * @param n - Number of entries
 * @param seed - Random seed to use
 */
std::vector<std::pair<uint32_t, uint32_t>> get_random_data(size_t n, int seed);

/**
 * Returns a list of lists of uniformly random inserts of size granularity,
 * such that the total number of elements is n
 * @param n - The total number of inserts
 * @param granularity - The number of inserts per list
 */
std::vector<std::vector<std::pair<uint32_t, uint32_t>>> get_random_inserts(size_t n, size_t granularity);

/**
 * Given data returns a random set of keys to read to measure performance
 * @param data - The base data
 * @param num_reads - The number of reads to perform
 */
std::vector<uint32_t> get_random_reads(std::vector<std::pair<uint32_t, uint32_t>> data, size_t num_reads);

/**
 * Gets the average segment size of leaf segments
 * @param buffered_pgm - A pgm model to analyze
 */
size_t get_avg_leaf_size(pgm::BufferedPGMIndex<uint32_t, uint32_t> &buffered_pgm);

/**
 * Helper function to get a output-friendly histogram of leaf segment sizes
 * @param buffered_pgm - The index to get the histogram for
 * @param n_bins - How many bins should the histogram have
 * @returns a pair (max_val, key_vals) where max_val is the maximum value in the histogram
 * and key_vals[i] is the number of values less than i/key_vals.size() * max_val
 */
std::pair<size_t, std::vector<size_t>> get_leaf_seg_size_histogram(pgm::BufferedPGMIndex<uint32_t, uint32_t> &buffered_pgm, size_t n_bins = 20, size_t hist_max = 30000);

/**
 * A helper function to actually do a certain number of inserts on a model
 * @param buffered_pgm - The model to insert into
 * @param data - The inserts to do
 */
void do_inserts(pgm::BufferedPGMIndex<uint32_t, uint32_t> &buffered_pgm, std::vector<std::pair<uint32_t, uint32_t>> &insert_data);

/**
 * A helper function to do a certain number of inserts and return the time it takes
 * @param buffered_pgm - The model to insert into
 * @param num_inserts - How many inserts to time
 */
size_t time_inserts(pgm::BufferedPGMIndex<uint32_t, uint32_t> &buffered_pgm, std::vector<std::pair<uint32_t, uint32_t>> &insert_data);

/**
 * A helper function to time a certain number of reads on a model
 * @param buffered_pgm - The model to read from
 * @param keys - The keys to lookup
 */
size_t time_reads(pgm::BufferedPGMIndex<uint32_t, uint32_t> &buffered_pgm, std::vector<uint32_t> &keys);

/**
 * A helper function to generate a workload
 * @param name - Name of the workload (for output / legibility)
 * @param initial_n - How much data should the workload start with
 * @param prop_writes - Proportion of writes in the workload
 * @param num_ops - The number of operations in the workload
 * @param seed - Random seed to generate data
 */
Workload generate_workload(std::string name, size_t initial_n, float prop_writes, size_t num_ops, int seed);

/**
 * A function that runs a workload using a given configuration and returns the time and memory footprint
 * @param workload - The workload to run
 * @param config - The configuration to use for this workload
 */
std::pair<size_t, size_t> benchmark_workload_config(Workload workload, Configuration config);

/* EXPERIMENTS */

void run_inserts_vs_index_power(std::string filename);

void run_inserts_vs_wlatency(std::string filename);

void run_inserts_vs_rlatency(std::string filename);

void run_insert_metrics(std::string filename);

void read_inserted(std::string filename);