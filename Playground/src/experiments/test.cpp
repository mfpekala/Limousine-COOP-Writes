#include <iostream>
#include "experiments.h"


using namespace std;

int test_func()
{
    std::cout << "\nHello world!";

    size_t n = 1000000;
    int seed = 2;
    size_t num_inserts = 100000;
    size_t granularity = num_inserts;


    // get data and workload
    auto data = get_random_data(n, seed);
    auto inserts = get_random_inserts(num_inserts, granularity);


    // // set up for in-place writes
    // size_t epsilon = 128;
    // size_t epsilon_recursive = 16;
    // float fill_ratio = 0.75;
    // float fill_ratio_recursive = 0.75;
    // size_t max_buffer_size = 0;
    // size_t split_neighborhood = 8;
    // size_t fixed_epsilon = (size_t)((float)epsilon / fill_ratio);
    // size_t fixed_epsilon_recursive = (size_t)((float)epsilon_recursive / fill_ratio_recursive);

    // set up for out-of-place writes
    size_t epsilon = 128;
    size_t epsilon_recursive = 16;
    float fill_ratio = 1.0;
    float fill_ratio_recursive = 1.0;
    size_t max_buffer_size = 512;
    size_t split_neighborhood = 2;
    size_t fixed_epsilon = (size_t)((float)epsilon / fill_ratio);
    size_t fixed_epsilon_recursive = (size_t)((float)epsilon_recursive / fill_ratio_recursive);

    // create pgm index
    auto pgm = pgm::BufferedPGMIndex<uint32_t, uint32_t>(
          data.begin(),
          data.end(),
          fixed_epsilon,
          fixed_epsilon_recursive,
          fill_ratio,
          fill_ratio_recursive,
          max_buffer_size,
          split_neighborhood);
    std::cout << "\nPGM created!";
    
    // do the inserts
    for (auto &insert_data : inserts)
    {
        do_inserts(pgm, insert_data);
    }
    std::cout << "\nInserts finished!";

    // get metrics
    pgm.read_profile.encode();
    pgm.split_history.encode();
    std::cout << "\n";

    return 0;
}