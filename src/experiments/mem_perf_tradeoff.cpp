/**
 * mem_perf_tradeoff
 *
 * The index developed exhibits a clear trade-off between memory usage and
 * performance on workloads with a high(er) proportion of writes. This
 * experiment quantifies this tradeoff by fixing a limited number of workloads
 * showing how memory and performance vary with neighborhood size and buffer
 * size (independently).
 *
 * Input:
 * - eps
 * - eps_rec
 * - initial_n
 * - num_ops
 * - seeds
 * - num_trials
 * - configs
 * - workloads
 * - neighs = the different neighborhood sizes to test
 * - bufs = the different buffer sizes to test
 * - fixed_neigh = the fixed neighborhood size to use when testing buffer sizes
 * - fixed_buf = the fixed buffer size to use when testing neighborhood sizes
 *
 * Output:
 * - conf_name: name given to this configuration
 * - work_name: name given to this workload
 * - seed: seed used during this run
 * - trial: trial # of this run
 * - neigh: the neighborhood size used in this run
 * - buf: the buffer size used in this run
 * - time: time (ms) to complete this workload (NOTE: does _not_ include time to
 * construct initial index)
 * - size: size (bytes) of the model at the end of this workload
 * - read_profile: the read profile of the model at the end of this workload
 * - tree_shape: the shape of the tree at the end of this workload
 */

#include "experiments.h"

namespace {
// Input
size_t eps = 64;
size_t eps_rec = 8;
size_t initial_n = 3e6;
size_t num_ops = 1e6;
std::vector<int> seeds = {1};
size_t num_trials = 2;
std::vector<size_t> neighs = {0, 1, 4, 8, 16};
std::vector<size_t> bufs = {4, 8, 16, 32, 64};
size_t fixed_neigh = 2;
size_t fixed_buf = 48;

std::vector<Configuration> get_configs() {
  std::vector<Configuration> result;

  /* BASE CONFIGS */
  Configuration base_config;
  base_config.eps = eps;
  base_config.eps_rec = eps_rec;

  Configuration base_inplace = base_config;
  base_inplace.buffer_size = 0;
  base_inplace.fill_ratio = 0.75;
  base_inplace.fill_ratio_rec = 0.75;

  Configuration base_outplace = base_config;
  base_outplace.buffer_size = 128;
  base_outplace.fill_ratio = 1.0;
  base_outplace.fill_ratio_rec = 1.0;
  base_outplace.split_neighborhood = 2;

  /* TESTED INPLACE CONFIGS */
  Configuration inplace_n0 = base_inplace;
  inplace_n0.name = "inplace_n0";
  inplace_n0.split_neighborhood = 0;
  result.push_back(inplace_n0);

  /* TESTED OUTPLACE CONFIGS */
  Configuration outplace_n0 = base_outplace;
  outplace_n0.name = "outplace_n0";
  outplace_n0.split_neighborhood = 0;

  for (auto &neigh : neighs) {
    Configuration outplace = outplace_n0;
    outplace.name =
        "outplace_n" + std::to_string(neigh) + "_b" + std::to_string(fixed_buf);
    outplace.split_neighborhood = neigh;
    outplace.buffer_size = fixed_buf;
    result.push_back(outplace);
  }
  for (auto &buf : bufs) {
    Configuration outplace = outplace_n0;
    outplace.name =
        "outplace_n" + std::to_string(fixed_neigh) + "_b" + std::to_string(buf);
    outplace.split_neighborhood = fixed_neigh;
    outplace.buffer_size = buf;
    result.push_back(outplace);
  }
  return result;
}

std::vector<Workload> get_workloads(int seed, bool zipfian) {
  Workload wp2, wp4;
  const float alpha = 1.1;  // Experimentally seems representative given no-dupe
  if (!zipfian) {
    wp2 = generate_workload("W20:R80", initial_n, 0.2, num_ops, seed);
    wp4 = generate_workload("W40:R60", initial_n, 0.4, num_ops, seed);
  } else {
    wp2 = generate_zipfian_workload("W20:R80", initial_n, 0.2, num_ops, alpha);
    wp4 = generate_zipfian_workload("W40:R60", initial_n, 0.4, num_ops, alpha);
  }
  return {wp2, wp4};
}
}  // namespace

// Just generates zipf data to make sure it has the distribution we think it
// does
void sanity_check_zipf(std::string filename) {
  std::ofstream fout;
  fout.open(filename);
  float alpha = 1.05;
  uint32_t num_trials = 100000;
  uint32_t N = 100000000;
  for (int trial = 0; trial < num_trials; ++trial) {
    uint32_t val = zipf(alpha, N);
    fout << val << std::endl;
  }
}

// Double check that the zipf workloads were generating are what we think they
// are
void sanity_check_zipf_workload(std::string filename) {
  std::ofstream fout;
  fout.open(filename);
  float alpha = 1.05;
  uint32_t num_elements = 1e5;
  uint32_t num_ops = 1e5;
  Workload w =
      generate_zipfian_workload("W20:R80", num_elements, 0.2, num_ops, alpha);
  for (auto &el : w.ops) {
    fout << el.key << std::endl;
  }
}

// Write the workloads analyzed to a file so we can verify zipfian and quantify
// skew
void write_workload(Workload w, int num) {
  std::ofstream fout_initial;
  fout_initial.open("workload_initial_" + std::to_string(num) + ".csv");
  for (auto &el : w.initial_data) {
    fout_initial << el.first << "," << el.second << std::endl;
  }
  std::ofstream fout_ops;
  fout_ops.open("workload_ops_" + std::to_string(num) + ".csv");
  for (auto &op : w.ops) {
    fout_ops << op.key << "," << op.val << "," << std::endl;
  }
}

void run_mem_perf_tradeoff(std::string filename) {
  std::ofstream fout;
  fout.open(filename);
  fout << "conf_name,work_name,seed,trial,neigh,buf,time,size,tree_shape,read_"
          "profile,split_history"
       << std::endl;

  auto configs = get_configs();

  progressbar bar(num_trials * seeds.size() * 2 * configs.size());
  for (size_t trial = 0; trial < num_trials; ++trial) {
    for (auto &seed : seeds) {
      // Second bool is is_zipfian
      auto workloads = get_workloads(seed, true);
      int wx = 0;
      for (auto &workload : workloads) {
        // Write the workload to a file so we can quantify skew of exactly what
        // is used in experiment
        write_workload(workload, wx++);
        for (auto &config : configs) {
          auto [time, memory, pgm] =
              benchmark_workload_config(workload, config);
          fout << config.name << "," << workload.name << "," << seed << ","
               << trial << "," << config.split_neighborhood << ","
               << config.buffer_size << "," << time << "," << memory << ","
               << pgm.get_tree_shape().encode() << ","
               << pgm.read_profile.encode() << "," << pgm.split_history.encode()
               << std::endl;
          bar.update();
        }
      }
    }
  }
}