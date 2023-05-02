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
    using LeafPos = std::pair<size_t, size_t>;

    // Tree structure
    ModelTree model_tree;
    DataLevel leaf_data;
    BufferLevel buffer_data;

    // Parameters
    size_t eps;
    size_t reduced_eps;
    size_t eps_rec;
    size_t reduced_eps_rec;
    float fill_ratio;
    float fill_ratio_rec;
    size_t buffer_size;
    size_t split_neighborhood;

    // Helper function to reset these reduced values when passing in something into the constructor
    void reset_reduced_values()
    {
      reduced_eps = (size_t)((float)eps * fill_ratio);
      reduced_eps_rec = (size_t)((float)eps_rec * fill_ratio);
    }

    // Constructor to build the index and set params
    template <typename RandomIt>
    FastPGMIndex(
        RandomIt first,
        RandomIt last,
        size_t eps = 128,
        size_t eps_rec = 16,
        float fill_ratio = 1.0,
        float fill_ratio_rec = 1.0,
        size_t buffer_size = 128,
        size_t split_neighborhood = 4) : eps(eps),
                                         eps_rec(eps_rec),
                                         fill_ratio(fill_ratio),
                                         fill_ratio_rec(fill_ratio_rec),
                                         buffer_size(buffer_size),
                                         split_neighborhood(split_neighborhood)
    {
      reset_reduced_values();
      build(first, last, eps, eps_rec);
    }

    // The function that does the actual building
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
      base_buffers.reserve(size_guess);
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

      auto build_level = [&](size_t n_els, auto in_fun, auto out_fun)
      {
        auto n_segments = internal::make_segmentation_par(n_els, eps, in_fun, out_fun);
        return n_segments;
      };

      size_t last_n = build_level(n, in_fun, out_fun);
      leaf_data = base_data;
      buffer_data = base_buffers;
      model_tree.push_back(base_models);

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

        last_n = build_level(last_n, rec_in_fun, rec_out_fun);
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
      while (goal_level < level)
      {
        // Predict the window to look
        size_t pred_ix = cur_model.predict_pos(key);
        size_t offset = get_offset_into_level(new_ix, level);
        pred_ix += offset;
        size_t WINDOW = level > 0 ? eps + 2 : eps_rec + 2;
        size_t lowest_ix = WINDOW < pred_ix ? pred_ix - WINDOW : 0;
        size_t highest_ix = pred_ix + WINDOW + 1 < model_tree[level - 1].size() ? pred_ix + WINDOW + 1 : model_tree[level - 1].size();
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

    // A helper function to get the model_ix, data_ix pair that is k elements before
    LeafPos go_back_by(size_t k, size_t model_ix, size_t data_ix)
    {
      size_t new_model_ix = model_ix;
      size_t new_data_ix = data_ix;
      while (0 <= new_model_ix && 0 < k)
      {
        if (k <= new_data_ix)
        {
          new_data_ix -= k;
          k = 0;
        }
        else
        {
          k -= (new_data_ix + 1);
          if (new_model_ix == 0)
          {
            return LeafPos(0, 0);
          }
          new_model_ix--;
          new_data_ix = model_tree[0][new_model_ix].n - 1;
        }
      }
      return LeafPos(new_model_ix, new_data_ix);
    }

    // A helper function to get the model_ix, data_ix pair that is k elements after
    LeafPos go_forward_by(size_t k, size_t model_ix, size_t data_ix)
    {
      size_t new_model_ix = model_ix;
      size_t new_data_ix = data_ix;
      while (new_model_ix < model_tree[0].size() && 0 < k)
      {
        if (new_data_ix + k < model_tree[0][new_model_ix].n)
        {
          new_data_ix += k;
          k = 0;
        }
        else
        {
          size_t diff = model_tree[0][new_model_ix].n - new_data_ix;
          k -= diff;
          if (new_model_ix >= model_tree[0].size() - 1)
          {
            return LeafPos(model_tree[0].size() - 1, model_tree[0].back().n - 1);
          }
          new_model_ix++;
          new_data_ix = 0;
        }
      }
      return LeafPos(new_model_ix, new_data_ix);
    }

    // A helper function to return the position of a key
    // NOTE: Will return the position that this key _should_ occupy
    // It's up to the caller to double check if that pos has the key or not
    // This is useful to not repeat work
    LeafPos find_pos(const K &key)
    {
      // Get the bounds
      size_t model_ix = get_ix_into_level(key, 0);
      size_t data_ix = model_tree[0][model_ix].predict_pos(key);
      size_t WINDOW = eps + 2;
      auto [start_model_ix, start_data_ix] = go_back_by(WINDOW, model_ix, data_ix);
      auto [end_model_ix, end_data_ix] = go_forward_by(WINDOW, model_ix, data_ix);

      size_t check_model_ix = start_model_ix;
      size_t check_data_ix = start_data_ix;
      while (check_model_ix <= end_model_ix)
      {
        if (check_model_ix == end_model_ix && end_data_ix < check_data_ix)
        {
          // We've gone past the end
          return LeafPos(end_model_ix, end_data_ix);
        }
        EntryVector &data = leaf_data[check_model_ix];
        if (data.size() <= check_data_ix)
        {
          check_model_ix++;
          check_data_ix = 0;
          continue;
        }
        if (data[check_data_ix].first == key)
        {
          return LeafPos(check_model_ix, check_data_ix);
        }
        if (key < data[check_data_ix].first)
        {
          return LeafPos(end_model_ix, end_data_ix);
          ;
        }
        check_data_ix++;
      }
      return LeafPos(end_model_ix, end_data_ix);
    }

    // Finds the value corresponding to a key
    // TODO: Implement this in an iterator-like way that supports range queries with
    // better memory performance
    V find(const K &key)
    {
      // Get the bounds
      auto p = find_pos(key);
      if (leaf_data[p.first][p.second].first != key)
      {
        return std::numeric_limits<V>::max();
      }
      return leaf_data[p.first][p.second].second;
    }

    inline void can_absorb_inplace_insert(size_t mx, size_t level, EntryVector &entries)
    {
      Model model = model_tree[level][mx];
      return reduced_eps + model.num_inplaces + entries.size() < eps;
    }

    // A helper function to handle an in-place insert into the tree
    void handle_inplace_insert(size_t mx, size_t level, EntryVector &entries)
    {
      // It's assumed this mx, level pair satisfies the can_absorb above
      // Need to handle leaf and internal separately
      if (level == 0)
      {
        // Leaf model
        EntryVector &data = model_tree[0][mx];
        auto insert_iter = std::lower_bound(data.begin(), data.end(), entries[0]);
        data.insert(insert_iter, entries);
      }
    }

    void insert(K key, V value)
    {
      // First make sure the key doesn't already exist in the leaf data
      LeafPos exist_pos = find_pos(key);
      if (leaf_data[0][exist_pos.first][exist_pos].first == key)
      {
        leaf_data[0][exist_pos.first][exist_pos.second].second = value;
        return;
      }
      // Then make sure the key doesn't already exist in the buffer
      for (size_t ix = 0; ix < buffer_data[exist_pos.first].size(); ++ix)
      {
        if (buffer_data[exist_pos.first][ix].first == key)
        {
          buffer_data[exist_pos.first][ix].second = value;
          return;
        }
      }

      // Now we can be sure that the entry doesn't already exist in the index
      size_t mx = get_ix_into_level(key, 0);
      EntryVector e = {std::make_pair(key, value)};
      if (can_absorb_inplace_insert(mx, 0, e))
      {
        handle_inplace_insert(mx, 0, e);
      }
    }

    void print_tree(size_t smallest_level)
    {
      std::cout << "Height: " << model_tree.size() << std::endl;
      int level = model_tree.size() - 1;
      while (smallest_level <= level && 0 <= level)
      {
        std::cout << "Level: " << level << ", num_segs: " << model_tree[level].size() << std::endl;
        for (size_t mx = 0; mx < model_tree[level].size(); ++mx)
        {
          std::cout << "(fk:" << model_tree[level][mx].first_key
                    << ", n:" << model_tree[level][mx].n
                    << "), ";
        }
        std::cout << std::endl;
        level--;
      }
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
    size_t num_inplaces = 0;

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