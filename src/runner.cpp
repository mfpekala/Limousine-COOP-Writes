#include <vector>
#include <cstdlib>
#include <fstream>
#include <iostream>
#include <algorithm>
#include <string>
#include "buffered/pgm_index_buffered.hpp"
#include "experiments.h"

int main(int argc, char **argv)
{
    run_better_skew("better_skew.csv");
    // debug();
    // run_compare_workloads("compare_workloads.csv");
    return 0;
}