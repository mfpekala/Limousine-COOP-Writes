#include <vector>
#include <cstdlib>
#include <fstream>
#include <iostream>
#include <algorithm>
#include <string>
#include "buffered/fast_index.hpp"
#include "experiments.h"

void simple_fast()
{
    auto data = get_random_data(1000000, 6);
    auto pgm = pgm::FastPGMIndex<uint32_t, uint32_t>(data.begin(), data.end(), 64, 8);

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
    simple_fast();
    return 0;
}

/*
Height: 2
Level: 1, num_segs: 1
(fk:38037, n:5),
Level: 0, num_segs: 5
(fk:38037, n:47687), (fk:1015550455, n:2), (fk:1015566872, n:27533), (fk:1612080553, n:2), (fk:1612104862, n:24775)
*/