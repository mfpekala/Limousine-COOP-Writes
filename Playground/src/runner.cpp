#include <vector>
#include <cstdlib>
#include <fstream>
#include <iostream>
#include <algorithm>
#include <string>
#include "buffered/pgm_index_buffered.hpp"

void simple_buffered()
{
    // Generate some random key-value pairs to bulk-load the Dynamic PGM-index
    std::vector<std::pair<uint32_t, uint32_t>> data_raw(1000000);
    std::srand(1);
    std::generate(data_raw.begin(), data_raw.end(), []
                  { return std::make_pair(std::rand(), std::rand()); });

    // Set the first value to 6: 6 for testing purposes
    data_raw[0].first = 6;
    data_raw[0].second = 6;
    // Make sure that there are no entries with key 2 for testing purposes
    for (auto &d : data_raw)
    {
        if (d.first == 2)
        {
            d.first = 3;
        }
    }

    std::sort(data_raw.begin(), data_raw.end());

    std::vector<std::pair<uint32_t, uint32_t>> data;
    data.reserve(data_raw.size());
    for (auto &p : data_raw)
    {
        if (data.size() && data.back().first == p.first)
        {
            continue;
        }
        data.push_back(p);
    }

    // Construct and bulk-load the Dynamic PGM-index
    const size_t epsilon = 64; // space-time trade-off parameter
    const size_t epsilon_recursive = 4;
    pgm::BufferedPGMIndex<uint32_t, uint32_t>
        buffered_pgm(data.begin(), data.end(), epsilon, epsilon_recursive);

    buffered_pgm.print_tree(1);

    // Do a bunch of inserts of random numbers
    for (int i = 0; i < 500000; i++)
    {
        auto q = std::rand() * std::rand();
        auto v = std::rand();
        buffered_pgm.insert(q, v);
    }

    buffered_pgm.print_tree(1);

    // Make sure that all the keys from the data made it into the index with the right value
    size_t num_errors = 0;
    for (auto &entry : data)
    {
        auto q = entry.first;
        auto v = entry.second;
        auto v2 = buffered_pgm.find(q);
        if (v != v2)
        {
            num_errors++;
            // std::cout << "Error: " << q << " " << v << " " << v2 << std::endl;
        }
    }
    std::cout << "NUM_ERRORS: " << num_errors << std::endl;
}

int generate_padding_stats()
{
    // Generate some random key-value pairs to bulk-load the Dynamic PGM-index
    std::vector<std::pair<uint32_t, uint32_t>> data(100);
    std::generate(data.begin(), data.end(), []
                  { return std::make_pair(std::rand(), std::rand()); });
    std::sort(data.begin(), data.end());

    // Construct and bulk-load the Dynamic PGM-index
    const size_t epsilon = 32;
    const size_t epsilon_recursive = 2;
    pgm::BufferedPGMIndex<uint32_t, uint32_t> buffered_pgm(data.begin(), data.end(), epsilon, epsilon_recursive);

    size_t NUM_SEGMENTS = 1;
    size_t num_sampled = 0;

    std::ofstream paddingTopFile;
    std::string run = "A";
    paddingTopFile.open("../paddings/" + std::to_string(epsilon) + "_" + run + "_paddingTops.txt");

    paddingTopFile.close();
}

/*
int main(int argc, char **argv)
{
    simple_buffered();

    return 0;
}
 */