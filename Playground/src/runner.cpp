/*
 * This example shows how to use pgm::DynamicPGMIndex, a std::map-like container supporting inserts and deletes.
 * Compile with:
 *   g++ updates.cpp -std=c++17 -I../include -o updates
 * Run with:
 *   ./updates
 */

#include <vector>
#include <cstdlib>
#include <fstream>
#include <iostream>
#include <algorithm>
#include <string>
#include "pgm/pgm_index.hpp"
#include "pgm/pgm_index_dynamic.hpp"
#include "debug/debug.hpp"

void updates() {
    // Generate some random key-value pairs to bulk-load the Dynamic PGM-index
    std::vector<std::pair<uint32_t, uint32_t>> data(1000000);
    std::generate(data.begin(), data.end(), [] { return std::make_pair(std::rand(), std::rand()); });
    std::sort(data.begin(), data.end());

    // Construct and bulk-load the Dynamic PGM-index
    pgm::DynamicPGMIndex<uint32_t, uint32_t> dynamic_pgm(data.begin(), data.end(), 2, 2, 2);

    // Insert some data
    dynamic_pgm.insert_or_assign(2, 4);
    dynamic_pgm.insert_or_assign(4, 8);
    dynamic_pgm.insert_or_assign(8, 16);

    // Delete data
    dynamic_pgm.erase(4);
    
    // Print the structure
    debug::print_dynamic_structure(dynamic_pgm);
}

void simple() {
    // Generate some random data
    std::vector<int> data(10000);
    std::generate(data.begin(), data.end(), std::rand);
    data.push_back(42);
    std::sort(data.begin(), data.end());

    // Construct the PGM-index
    const int epsilon = 128; // space-time trade-off parameter
    pgm::PGMIndex<int, epsilon> index(data);

    // Query the PGM-index
    auto q = 42;
    auto range = index.search(q);
    auto lo = data.begin() + range.lo;
    auto hi = data.begin() + range.hi;
    std::cout << *std::lower_bound(lo, hi, q);
}

void simple_buffered() {
    // Generate some random key-value pairs to bulk-load the Dynamic PGM-index
    std::vector<std::pair<uint32_t, uint32_t>> data_raw(10000);
    std::srand(1);
    std::generate(data_raw.begin(), data_raw.end(), [] { return std::make_pair(std::rand(), std::rand()); });

    // Set the first value to 6: 6 for testing purposes
    data_raw[0].first = 6;
    data_raw[0].second = 6;
    // Make sure that there are no entries with key 2 for testing purposes
    for (auto &d : data_raw) {
        if (d.first == 2) {
            d.first = 3;
        }
    }

    std::sort(data_raw.begin(), data_raw.end());

    std::vector<std::pair<uint32_t, uint32_t>> data;
    data.reserve(data_raw.size());
    for (auto &p : data_raw) {
        if (data.size() && data.back().first == p.first) {
            continue;
        }
        data.push_back(p);
    }

    // Construct and bulk-load the Dynamic PGM-index
    const int epsilon = 8; // space-time trade-off parameter
    pgm::BufferedPGMIndex<uint32_t, uint32_t, epsilon> buffered_pgm(data);

    size_t looking_for = 847549551;
    size_t seg_ix = buffered_pgm.by_level_segment_ix_for_key(looking_for, 0);
    std::cout << "Looking for: " << looking_for << ", got: " << seg_ix << std::endl;

    /*
    // Print some facts about the model
    std::cout << "Height: " << buffered_pgm.height() << std::endl;
    std::cout << "Segment Count: " << buffered_pgm.segments_count() << std::endl;

    // Make sure we can search for the key 6
    auto q = 6;
    auto range = buffered_pgm.search(q);
    
    // Make sure we can find the value for key 6
    auto v = buffered_pgm.find(q);
    std::cout << "Value for key 6: " << v << std::endl;

    // Test insert
    q = 2;
    v = buffered_pgm.find(q);
    std::cout << "Value for key 2: " << v << std::endl;
    buffered_pgm.insert(2, 4);
    v = buffered_pgm.find(q);
    std::cout << "Value for key 2 (after insert): " << v << std::endl;
    */

    buffered_pgm.print_tree();

    /*
    // Do a bunch of inserts of random numbers
    for (int i = 0; i < 10000; i++) {
        auto q = std::rand();
        auto v = std::rand();
        buffered_pgm.insert(q, v);
    }
    */

    
    // Make sure that all the keys from the data made it into the index with the right value
    for (auto &entry : data) {
        auto q = entry.first;
        auto v = entry.second;
        auto v2 = buffered_pgm.find(q);
        if (v != v2) {
            std::cout << "Error: " << q << " " << v << " " << v2 << std::endl;
        }
    }
    
}

int generate_padding_stats() {
    // Generate some random key-value pairs to bulk-load the Dynamic PGM-index
    std::vector<std::pair<uint32_t, uint32_t>> data(100);
    std::generate(data.begin(), data.end(), [] { return std::make_pair(std::rand(), std::rand()); });
    std::sort(data.begin(), data.end());

    // Construct and bulk-load the Dynamic PGM-index
    const size_t epsilon = 32;
    pgm::BufferedPGMIndex<uint32_t, uint32_t, epsilon> buffered_pgm(data);

    size_t NUM_SEGMENTS = 1;
    size_t num_sampled = 0;

    std::ofstream paddingTopFile;
    std::string run = "A";
    paddingTopFile.open("../paddings/" + std::to_string(epsilon) + "_" + run + "_paddingTops.txt");
    for (auto &segment : buffered_pgm.segments) {
        auto[paddingTop, paddingBottom] = segment.get_padding();
        for (auto &p : paddingTop) {
            paddingTopFile << p << std::endl;
        }
        if (num_sampled >= NUM_SEGMENTS) {
            break;
        }
    }
    paddingTopFile.close();
}

int main(int argc, char **argv) {
    simple_buffered();

    return 0;
}