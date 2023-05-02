// This file is part of work for CS 265 at Harvard University.
// The code is based on the original work by the authors of the paper:
// https://pgm.di.unipi.it/
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

#pragma once

#include "pgm/piecewise_linear_model.hpp"
#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <iterator>
#include <limits>
#include <stdexcept>
#include <utility>
#include <vector>
#include <chrono>
// #include "debug/metrics.hpp"

namespace pgm
{
  /**
   * @tparam K - the type of keys in the data structure
   * @tparam V - the type of values in the data structure
   */
  template <typename K, typename V>
  class FastPGMIndex
  {
  public:
    // Types and handy shortnames
    struct Model;

    using Entry = std::pair<K, V>;
    using EntryVector = std::vector<Entry>;
    using DataLevel = std::vector<EntryVector>;
    using BufferLevel = std::vector<EntryVector>;
    using ModelLevel = std::vector<Model>;
    using ModelTree = std::vector<ModelLevel>;

    // Tree structure
    ModelTree model_tree;
    DataLevel leaf_data;
    BufferLevel buffer_data;

    // Parameters
    size_t eps = 128;
    size_t reduced_eps;
    size_t eps_rec = 16;
    size_t reduced_eps_rec;
    float fill_ratio = 1.0;
    float fill_ratio_rec = 1.0;
    size_t buffer_size = 128;
    size_t split_neighborhood = 4;

    // Helper function to reset these reduced values when passing in something into the constructor
    void reset_reduced_values()
    {
      reduced_eps = (size_t)((float)eps * fill_ratio);
      reduced_eps_rec = (size_t)((float)eps_rec * fill_ratio);
    }

    template <typename RandomIt>
    void build(RandomIt first, RandomIt last, size_t eps, size_t rec_eps)
    {
      auto n = std::distance(first, last);
      if (n <= 0)
        return;

      // NOTE: Potentially not the most efficient, but logically easiest to work with as we're
      // building the index
      std::vector<K> keys;
      std::vector<V> values;
      keys.reserve(n);
      values.reserve(n);
      // Could be more efficient by using std::move
      for (auto it = first; it != last; ++it)
      {
        keys.push_back(it->first);
        values.push_back(it->second);
      }

      // Allocate initial guesses for the sizes of what we'll need for the index
      size_t size_guess = n / (eps * eps);
      DataLevel base_data;
      base_data.reserve(size_guess);
      BufferLevel base_buffers;
      base_buffers.resert(size_guess);
      ModelLevel base_models;
      base_models.reserve(size_guess);

      // For keeping track of nodes as we build them for the base level
      size_t cur_node_size = 0;
      RandomIt first_node_data = first;
      RandomIt last_node_data = last;

      auto in_fun = [&](auto i)
      {
        K key = keys[i];
        if (cur_node_size == 0)
        {
          first_node_data = std::next(first, i);
        }
        cur_node_size++;
        last_node_data = std::next(first, i + 1);
        return std::pair<K, size_t>(key, cur_node_size);
      };

      auto out_fun = [&](auto can_seg)
      {
        EntryVector data(first_node_data, last_node_data);
        EntryVector buffer;
        buffer.reserve(buffer_size);
        auto model = Model(this, cur_node_size, can_seg);

        base_data.push_back(data);
        base_buffers.push_back(buffer);
        base_models.push_back(model);

        cur_node_size = 0;
      };

      auto build_level = [&](auto in_fun, auto out_fun)
      {
        auto n_segments = internal::make_segmentation_par(n, eps, in_fun, out_fun);
        return n_segments;
      };

      size_t last_n = build_level(in_fun, out_fun);

      // The above code successfully builds the base level of the model
      // Now it's time to recursively construct the internal levels
      while (last_n > 1)
      {
        ModelLevel last_level = model_tree.back();
        ModelLevel next_level;
        size_t cur_model_size = 0;

        auto rec_in_fun = [&](auto i)
        {
          K key = last_level[i].first_key;
          size_t val = cur_model_size++;
          return std::pair<K, size_t>(key, val);
        };

        auto rec_out_fun = [&](auto can_seg)
        {
          if (cur_model_size <= 0)
            return;
          Model model = Model(this, cur_model_size, can_seg);
          next_level.push_back(model);
          cur_model_size = 0;
        };

        last_n = build_level(rec_in_fun, rec_out_fun);
        model_tree.push_back(next_level);
      }
    }

    // Each model will index it's elements assuming it has the first one (index 0)
    // This helper function counts the number of elements before this segment and
    // returns the proper offset
    size_t get_offset_into_level(size_t model_ix, size_t level)
    {
      size_t sum = 0;
      for (size_t ix = 0; ix < model_ix && ix < model_tree[level].size(); ++ix)
      {
        sum += model_tree[level][ix].n;
      }
      return sum;
    }

    // A helper function to return the index you will find a certain key in a level
    size_t get_ix_into_level(K key, size_t goal_level)
    {
      size_t level = model_tree.size() - 1;
      Model cur_model = model_tree[level][0];
      size_t new_ix = 0;
      while (level < goal_level)
      {
        // Predict the window to look
        size_t pred_ix = cur_model.predict_pos(key);
        size_t offset = get_offset_into_level(new_ix, level);
        pred_ix += offset;
        size_t WINDOW = level > 0 ? eps + 2 : eps_rec + 2;
        size_t lowest_ix = WINDOW < pred_ix ? pred_ix - WINDOW : 0;
        size_t highest_ix = pred_ix + WINDOW + 1 < model_tree[level].size() ? pred_ix + WINDOW + 1 : model_tree[level].size();
        // Find the first model where the _next_ model is too high
        new_ix = lowest_ix;
        while (new_ix + 1 < highest_ix && model_tree[level - 1][new_ix + 1].first_key < key)
        {
          ++new_ix;
        }
        cur_model = model_tree[level - 1][new_ix];
        level--;
      }
      return new_ix;
    }

    // Returns the index of the leaf model that would contain this key
    auto leaf_model_for_key(const K &key)
    {
      return model_tree[0].begin() + get_ix_into_level(key, 0);
    }
    // The same as above but auto will type it to return a mutable model
    auto mutable_leaf_model_for_key(const K &key)
    {
      return model_tree[0].begin() + get_ix_into_level(key, 0);
    }
  };

#pragma push pack 1

  template <typename K, typename V>
  struct FastPGMIndex<K, V>::Model
  {
    const FastPGMIndex<K, V> *super;
    size_t n;
    K first_key;
    float slope;
    int32_t intercept;
    size_t num_inplaces;

    // Constructor one: specify a model manually (usually used to make dummy models)
    Model(
        const FastPGMIndex<K, V> *super,
        size_t n,
        K first_key,
        float slope, int32_t intercept) : super(super), n(n),
                                          first_key(first_key),
                                          slope(slope),
                                          intercept(intercept) {}

    // Constructor two (the useful one): specify a model using a "canonical segment" (from original index)
    Model(
        const FastPGMIndex<K, V> *super,
        size_t n,
        const typename internal::OptimalPiecewiseLinearModel<K, size_t>::CanonicalSegment &can_seg) : super(super), n(n)
    {
      first_key = can_seg.get_first_x();
      auto [cs_slope, cs_intercept] = can_seg.get_floating_point_segment(first_key);
      slope = cs_slope;
      intercept = cs_intercept;
    }

    size_t predict_pos(K key)
    {
      auto pos = int64_t(slope * (key - first_key)) + intercept;
      pos = pos > 0 ? size_t(pos) : 0ull;
      pos = pos < n ? pos : n - 1;
      return pos;
    }
  };
}