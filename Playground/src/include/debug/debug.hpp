#include <iostream>
#include <cstddef>
#include <cstdint>
#include <iterator>
#include "buffered/pgm_index_buffered.hpp"
#include "pgm/pgm_index.hpp"
#include "pgm/pgm_index_dynamic.hpp"

#pragma once

namespace debug {
void print_structure(pgm::PGMIndex<int> &pgm) {

}

void print_dynamic_structure(pgm::DynamicPGMIndex<uint32_t, uint32_t> &dynamic_pgm) {
  std::cout << "In print_structure" << std::endl;

  for (uint8_t ix = 0; ix < dynamic_pgm.used_levels; ++ix) {
    std::cout << "LEVEL " << unsigned(ix) << std::endl;

    // If it has values print out the range
    if (dynamic_pgm.level(ix).empty()) {
      std::cout << "❌ empty" << std::endl;
    } else {
      // NOTE: Inefficient, just to play around with the structure
      auto first = dynamic_pgm.level(ix).begin();
      auto last = std::prev(dynamic_pgm.level(ix).end());
      std::cout << "Range [" << first->first << ", " << last->first << "]" << std::endl; 
    }

    // High level information if it has a PGM
    if (dynamic_pgm.has_pgm(ix)) {
      std::cout << "✅ pgm" << std::endl;
      auto pgm = dynamic_pgm.pgm(ix);
      std::cout << "Epsilon: " << pgm.epsilon_value << std::endl;
    } else {
      std::cout << "❌ no pgm" << std::endl;
    }
  }

  dynamic_pgm.find(6);
  return;
}
}