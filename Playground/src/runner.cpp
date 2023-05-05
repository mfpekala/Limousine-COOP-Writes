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
    auto data = get_random_data(100000, 1);

    Configuration config;
    config.eps = 64;
    config.eps_rec = 8;
    config.fill_ratio = 0.75;
    config.fill_ratio_rec = 0.75;
    config.buffer_size = 0;
    config.split_neighborhood = 0;

    auto pgm = pgm::BufferedPGMIndex<uint32_t, uint32_t>(
        data.begin(),
        data.end(),
        config.eps,
        config.eps_rec,
        config.fill_ratio,
        config.fill_ratio_rec,
        config.buffer_size,
        config.split_neighborhood);

    auto inserts = get_random_inserts(1e4, 1000);
    for (auto &insert_data : inserts)
    {
        do_inserts(pgm, insert_data);
    }

    std::cout << pgm.split_history.encode() << std::endl;
}

int main(int argc, char **argv)
{
    run_inserts_vs_wlatency("inserts_vs_wlatency.csv");
    // debug();
    return 0;
}