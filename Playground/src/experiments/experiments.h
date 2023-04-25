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

/* EXPERIMENTS */

void run_inserts_vs_index_power(std::string filename);

void run_inserts_vs_wlatency(std::string filename);

void run_inserts_vs_rlatency(std::string filename);
