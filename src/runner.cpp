#include <vector>
#include <cstdlib>
#include <fstream>
#include <iostream>
#include <algorithm>
#include <string>
#include "coop/pgm_index_buffered.hpp"
#include "experiments.h"

int main(int argc, char **argv)
{
    run_inserts_vs_index_power("test.csv");
    return 0;
}