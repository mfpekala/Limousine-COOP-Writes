#include <vector>
#include <cstdlib>
#include <fstream>
#include <iostream>
#include <algorithm>
#include <string>
#include "buffered/pgm_index_buffered.hpp"
#include "experiments.h"

void simple_fast()
{
    auto data = get_random_data(1000000, 1);
    auto pgm = pgm::BufferedPGMIndex<uint32_t, uint32_t>(data.begin(), data.end(), 64, 8);

    pgm.print_tree(1);
    auto inserts = get_random_inserts(1000000, 1000000)[0];
    for (auto &i : inserts)
    {
        pgm.insert(i.first, i.second);
    }
    pgm.print_tree(1);

    size_t num_errors = 0;
    auto looking = 2125817040;
    for (auto &entry : data)
    {
        volatile auto k = 0;
        if (entry.first == looking)
        {
            k += 1;
        }
        auto val = pgm.find(entry.first);
        if (val != entry.second)
        {
            // std::cout << entry.first << std::endl;
            num_errors++;
        }
    }
    std::cout << "num_errors: " << num_errors << std::endl;
}

int main(int argc, char **argv)
{
    run_time_workloads_uniform("workloads_uniform.csv");
    return 0;
}