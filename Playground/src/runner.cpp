#include <vector>
#include <cstdlib>
#include <fstream>
#include <iostream>
#include <algorithm>
#include <string>
#include "buffered/pgm_index_buffered.hpp"
#include "experiments.h"

void debug()
{
    auto data = get_skewed_data(1000000, 0.5);

    for (auto &d : data)
    {
        std::cout << d.first << " " << d.second << std::endl;
    }

    Configuration config;
    config.eps = 64;
    config.eps_rec = 8;
    config.fill_ratio = 1.0;
    config.fill_ratio_rec = 1.0;
    config.buffer_size = 64;
    config.split_neighborhood = 2;

    auto pgm = pgm::BufferedPGMIndex<uint32_t, uint32_t>(
        data.begin(),
        data.end(),
        config.eps,
        config.eps_rec,
        config.fill_ratio,
        config.fill_ratio_rec,
        config.buffer_size,
        config.split_neighborhood);

    auto inserts = get_random_inserts(1000000, 1000000);
    for (auto &insert_data : inserts)
    {
        do_inserts(pgm, insert_data);
    }
    auto reads = get_random_reads(data, 100000);
    time_reads(pgm, reads);

    std::cout << pgm.get_tree_shape().encode() << std::endl;
    std::cout << pgm.read_profile.encode() << std::endl;
    std::cout << pgm.split_history.encode() << std::endl;
}

int main(int argc, char **argv)
{
    run_better_skew("better_skew.csv");
    // debug();
    // run_compare_workloads("compare_workloads.csv");
    return 0;
}