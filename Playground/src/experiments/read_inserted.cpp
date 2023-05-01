/**
 * Experiment name
 *
 * High-level description. What does this experiment change? What does it leave constant?
 *
 * Input:
 * - (list of variables that you can set)
 *
 * Output:
 * - (list of named columns in order and what they represent)
 */

#include "experiments.h"

namespace
{
  // Input
  // < Define all the Inputs listed above>
}

void read_inserted(std::string filename)
{
  size_t n = 100000;
  size_t epsilon = 32;
  size_t epsilon_recursive = 16;
  size_t max_buffer_size = 256;
  size_t split_neighborhood = 4;
  auto data = get_random_data(n, 6);
  auto pgm = pgm::BufferedPGMIndex<uint32_t, uint32_t>(
      data.begin(),
      data.end(),
      epsilon,
      epsilon_recursive,
      1.0, 1.0,
      max_buffer_size,
      split_neighborhood);
  ;

  auto inserts = get_random_data(n, 8);
  do_inserts(pgm, inserts);

  std::vector<uint32_t> keys;
  for (auto &p : inserts)
  {
    keys.push_back(p.first);
  }

  time_reads(pgm, keys);
}