#include <algorithm>
#include <cstdlib>
#include <fstream>
#include <iostream>
#include <string>
#include <vector>

#include "coop/pgm_index_buffered.hpp"
#include "experiments.h"

int main(int argc, char **argv) {
  run_mem_perf_tradeoff("skew_workload.csv");
  return 0;
}