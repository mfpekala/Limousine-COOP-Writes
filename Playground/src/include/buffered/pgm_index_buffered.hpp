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
#include "debug/metrics.hpp"

namespace pgm
{
    /**
     * @tparam K - the type of keys in the data structure
     * @tparam V - the type of values in the data structure
     */
    template <typename K, typename V>
    class BufferedPGMIndex
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

        // Metrics
        inline TreeShape get_tree_shape() const
        {
            TreeShape result;
            for (const auto &level : model_tree)
                result.level_sizes.push_back(level.size());
            return result;
        }
        ReadProfile read_profile;
        SplitHistory split_history;
        inline void log_split(size_t level, size_t data_mvmt)
        {
            while (split_history.splits_by_level.size() < level + 1)
            {
                split_history.splits_by_level.push_back(0);
                split_history.data_movement_by_level.push_back(0);
            }
            split_history.splits_by_level[level]++;
            split_history.data_movement_by_level[level] += data_mvmt;
        }

        // Helper function to reset these reduced values when passing in something into the constructor
        void reset_reduced_values()
        {
            reduced_eps = (size_t)((float)eps * fill_ratio);
            reduced_eps_rec = (size_t)((float)eps_rec * fill_ratio);
        }

        // Constructor to build the index and set params
        template <typename RandomIt>
        BufferedPGMIndex(
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
        // TODO: Investigate the BIG small BIG pattern
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
            size_t WINDOW = eps + 3;
            auto [start_model_ix, start_data_ix] = go_back_by(WINDOW, model_ix, data_ix);

            size_t check_model_ix = start_model_ix;
            size_t check_data_ix = start_data_ix;
            size_t num_explored = 0;
            while (check_model_ix < model_tree[0].size() && num_explored < WINDOW * 2)
            {
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
                    return LeafPos(check_model_ix, check_data_ix);
                }
                check_data_ix++;
                num_explored++;
            }
            return LeafPos(model_tree[0].size() - 1, leaf_data[model_tree[0].size() - 1].size() - 1);
        }

        // Finds the value corresponding to a key
        // TODO: Implement this in an iterator-like way that supports range queries with
        // better memory performance
        V find(const K &key)
        {
            // Get the bounds
            auto p = find_pos(key);
            if (leaf_data[p.first][p.second].first == key)
            {
                // It's where it should be
                read_profile.num_data++;
                return leaf_data[p.first][p.second].second;
            }

            /*
            // SORTED VERSION
            // It's not where it should be
            Entry e(key, std::numeric_limits<V>::max());
            auto iter = std::lower_bound(buffer_data[p.first].begin(), buffer_data[p.first].end(), e);
            if (iter != buffer_data[p.first].end() && iter->first == key)
            {
                read_profile.num_buffer++;
                return iter->second;
            }
            */

            /*
            // UNSORTED VERSION
            */
            // It's not where it should be
            for (auto &el : buffer_data[p.first])
            {
                if (el.first == key)
                {
                    read_profile.num_buffer++;
                    return el.second;
                }
            }
            read_profile.num_ne++;
            return std::numeric_limits<V>::max();
        }

        inline bool can_absorb_inplace_inserts(size_t mx, size_t level, size_t num_inserts)
        {
            Model &model = model_tree[level][mx];
            return reduced_eps + model.num_inplaces + num_inserts < eps;
        }

        // A helper function to handle an in-place insert into the tree
        void handle_inplace_inserts(size_t mx, size_t level, Entry e)
        {
            if (level == 0)
            {
                // At the leaf level we need to go in and actually change the data properly
                EntryVector &data = leaf_data[mx];
                auto insert_iter = std::lower_bound(data.begin(), data.end(), e);
                data.insert(insert_iter, e);
            }
            // At all other levels we just need to update the model's num_inplaces
            model_tree[level][mx].num_inplaces++;
        }

        void insert(K key, V value)
        {
            // First make sure the key doesn't already exist in the leaf data
            LeafPos exist_pos = find_pos(key);
            if (leaf_data[exist_pos.first][exist_pos.second].first == key)
            {
                leaf_data[exist_pos.first][exist_pos.second].second = value;
                return;
            }

            Entry e = std::pair<K, V>(key, value);

            /*
            // SORTED VERSION
            auto iter = std::lower_bound(
                buffer_data[exist_pos.first].begin(),
                buffer_data[exist_pos.first].end(),
                e);
            if (iter != buffer_data[exist_pos.first].end() && iter->first == key)
            {
                iter->second = value;
                return;
            }
            */

            /*
            // UNSORTED VERSION
            */
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
            if (can_absorb_inplace_inserts(exist_pos.first, 0, 1))
            {
                // Can handle as an in place insert
                handle_inplace_inserts(exist_pos.first, 0, e);
                return;
            }

            /*
            // SORTED VERSION
            // Iter already defined above
            buffer_data[exist_pos.first].insert(iter, e);
             */

            /*
            // UNSORTED VERSION
            */
            buffer_data[exist_pos.first].push_back(e);

            if (buffer_data[exist_pos.first].size() > buffer_size)
            {
                leaf_split(exist_pos.first);
            }
        }

        // Returns the indices (into the level) of the models participating in this merge
        // Note that the first value is inclusive, and the second is exclusive
        inline std::pair<size_t, size_t> get_split_window(size_t split_mx, size_t level)
        {
            size_t use_neighborhood = split_neighborhood;
            size_t parent_ix = get_ix_into_level(model_tree[level][split_mx].first_key, level + 1);
            size_t low_mx = split_mx;
            size_t moved = 0;
            while (0 < low_mx && moved < use_neighborhood)
            {
                size_t prev_parent_ix = get_ix_into_level(model_tree[level][low_mx - 1].first_key, level + 1);
                if (prev_parent_ix != parent_ix)
                    break;
                moved++;
                low_mx--;
            }
            size_t high_mx = split_mx + 1;
            moved = 0;
            while (high_mx < model_tree[level].size() && moved < use_neighborhood)
            {
                size_t next_parent_ix = get_ix_into_level(model_tree[level][high_mx].first_key, level + 1);
                if (next_parent_ix != parent_ix)
                    break;
                moved++;
                high_mx++;
            }

            return std::pair<size_t, size_t>(low_mx, high_mx);
        }

        // A helper function intoduced to help debug splitting at the leaf level
        void verify_leaves()
        {
            K last_key = model_tree[0][0].first_key;
            size_t ix = 1;
            while (ix < model_tree[0].size())
            {
                K cur_key = model_tree[0][ix].first_key;
                if (last_key >= cur_key)
                {
                    throw std::invalid_argument("Leaves bad (trad case)");
                }
                // Now check if last el of previous buffer is bigger
                if (buffer_data[ix - 1].size() > 0 && buffer_data[ix - 1].back().first >= cur_key)
                {
                    throw std::invalid_argument("Leaves bad (buffer edge case)");
                }
                last_key = cur_key;
                ix++;
            }
            // Now check if the buffers are sorted
            size_t bx = 0;
            while (bx < buffer_data.size())
            {
                if (!std::is_sorted(buffer_data[bx].begin(), buffer_data[bx].end()))
                {
                    throw std::invalid_argument("Leaves bad (buffer unsorted)");
                }
                ++bx;
            }
        }

        // A helper function intoduced to help debug splitting at internal levels
        void verify_internal_level(size_t level)
        {
            K last_key = model_tree[level][0].first_key;
            size_t ix = 1;
            while (ix < model_tree[level].size())
            {
                K cur_key = model_tree[level][ix].first_key;
                if (last_key >= cur_key)
                {
                    throw std::invalid_argument("Internal bad");
                }
                ++ix;
            }
        }

        // Splits the model at index mx in the leaf level
        void leaf_split(size_t mx)
        {
            // Data movement calculation
            // 1. Total number of elements in data + buffer of all models in window
            // 2. Total size of models that get added to the tree
            size_t data_mvmt = 0;

            auto [low_mx, high_mx] = get_split_window(mx, 0);
            size_t total_els = 0;
            for (size_t mx = low_mx; mx < high_mx; ++mx)
            {
                total_els += model_tree[0][mx].n + buffer_data[mx].size();
            }
            data_mvmt += total_els * sizeof(Entry);

            LeafPos proper_pos = LeafPos(low_mx, 0);
            LeafPos buff_pos = LeafPos(low_mx, 0);

            EntryVector next_node;
            DataLevel new_nodes_data;
            BufferLevel new_buffers_data;
            ModelLevel new_models_data;

            size_t feed_ix = low_mx;
            size_t proper_ix = 0;
            size_t buffer_ix = 0;

            auto in_fun = [&](size_t i)
            {
                bool proper_exhausted = leaf_data[feed_ix].size() <= proper_ix;
                bool buffer_exhaused = buffer_data[feed_ix].size() <= buffer_ix;
                if (proper_exhausted && buffer_exhaused)
                {
                    // Move on to next model
                    feed_ix++;
                    proper_ix = 0;
                    buffer_ix = 0;
                    proper_exhausted = leaf_data[feed_ix].size() <= proper_ix;
                    buffer_exhaused = buffer_data[feed_ix].size() <= buffer_ix;
                }
                Entry next_e;
                if (proper_exhausted)
                {
                    next_e = buffer_data[feed_ix][buffer_ix++];
                }
                else if (buffer_exhaused)
                {
                    next_e = leaf_data[feed_ix][proper_ix++];
                }
                else
                {
                    K proper_key = leaf_data[feed_ix][proper_ix].first;
                    K buffer_key = buffer_data[feed_ix][buffer_ix].first;
                    next_e = proper_key < buffer_key
                                 ? leaf_data[feed_ix][proper_ix++]
                                 : buffer_data[feed_ix][buffer_ix++];
                }
                next_node.push_back(next_e);
                return std::pair<K, size_t>(next_e.first, next_node.size() - 1);
            };

            auto out_fun = [&](auto can_seg)
            {
                if (next_node.size() <= 0)
                    return;
                EntryVector buf;
                buf.reserve(buffer_size);
                auto model = Model(this, next_node.size(), can_seg);
                new_nodes_data.push_back(EntryVector(next_node));
                new_buffers_data.push_back(buf);
                new_models_data.push_back(model);
                next_node.clear();
            };
            internal::make_segmentation_par(total_els, reduced_eps, in_fun, out_fun);

            // Now we erase the old buffer and data? (idk if we should do data, didn't before) and replace
            leaf_data.erase(leaf_data.begin() + low_mx, leaf_data.begin() + high_mx);
            leaf_data.insert(leaf_data.begin() + low_mx, new_nodes_data.begin(), new_nodes_data.end());
            buffer_data.erase(buffer_data.begin() + low_mx, buffer_data.begin() + high_mx);
            buffer_data.insert(buffer_data.begin() + low_mx, new_buffers_data.begin(), new_buffers_data.end());
            model_tree[0].erase(model_tree[0].begin() + low_mx, model_tree[0].begin() + high_mx);
            model_tree[0].insert(model_tree[0].begin() + low_mx, new_models_data.begin(), new_models_data.end());
            data_mvmt += new_models_data.size() * sizeof(Model);

            log_split(0, data_mvmt);

            size_t parent_ix = get_ix_into_level(new_models_data[0].first_key, 1);
            int change_in_size = new_models_data.size() - (high_mx - low_mx);
            model_tree[1][parent_ix].n += change_in_size;
            if (change_in_size > 0)
                internal_split(parent_ix, 1);
        }

        // Split a model higher up in the tree
        void internal_split(size_t split_mx, size_t level)
        {
            // Data movement calculation
            // Unlike in the leaf case, this split doesn't involve any movement of the base
            // data OR the buffer data. So, data movement is just
            // 1. Total size of models that get added to the tree
            size_t data_mvmt = 0;

            auto [low_mx, high_mx] = get_split_window(split_mx, level);
            size_t total_els = 0;
            for (size_t mx = low_mx; mx < high_mx; ++mx)
            {
                total_els += model_tree[level][mx].n;
            }
            size_t offset = get_offset_into_level(low_mx, level);
            ModelLevel new_models;
            size_t cur_model_size = 0;

            auto in_fun = [&](size_t i)
            {
                return std::pair<K, size_t>(model_tree[level - 1][offset + i].first_key, cur_model_size++);
            };

            auto out_fun = [&](auto can_seg)
            {
                if (cur_model_size <= 0)
                    return;
                Model model = Model(this, cur_model_size, can_seg);
                cur_model_size = 0;
                new_models.push_back(model);
            };

            internal::make_segmentation_par(total_els, reduced_eps_rec, in_fun, out_fun);

            model_tree[level].erase(model_tree[level].begin() + low_mx, model_tree[level].begin() + high_mx);
            model_tree[level].insert(model_tree[level].begin() + low_mx, new_models.begin(), new_models.end());
            data_mvmt += new_models.size() * sizeof(Model);

            log_split(level, data_mvmt);

            int change_in_size = new_models.size() - (high_mx - low_mx);
            if (model_tree.size() - 1 <= level)
            {
                make_new_root();
            }
            else
            {
                size_t parent_ix = get_ix_into_level(new_models[0].first_key, level + 1);
                model_tree[level + 1][parent_ix].n += change_in_size;
                if (change_in_size > 0)
                    internal_split(parent_ix, level + 1);
            }
        }

        void make_new_root()
        {
            while (model_tree.back().size() > 1)
            {
                // Each time we do this, it really should be considered a split at level model_tree.size()
                size_t level = model_tree.size();
                // Data movement calculation
                // Same as internal_split, data_mvmt is just the size of the new models
                size_t data_mvmt = 0;

                size_t total_els = model_tree.back().size();
                ModelLevel new_models;
                size_t cur_model_size = 0;

                auto in_fun = [&](size_t i)
                {
                    return std::pair<K, size_t>(model_tree.back()[i].first_key, cur_model_size++);
                };

                auto out_fun = [&](auto can_seg)
                {
                    if (cur_model_size <= 0)
                        return;
                    Model model = Model(this, cur_model_size, can_seg);
                    cur_model_size = 0;
                    new_models.push_back(model);
                };

                internal::make_segmentation_par(total_els, reduced_eps_rec, in_fun, out_fun);

                log_split(level, data_mvmt);

                model_tree.push_back(new_models);
            }
        }

        // Returns the size of the index
        // NOTE: Does not include size of the data, just size of the models + buffers
        size_t size_in_bytes()
        {
            size_t result = 0;

            for (auto &level : model_tree)
            {
                result += level.size() * sizeof(Model);
            }

            for (auto &buf : buffer_data)
            {
                result += buf.capacity() * sizeof(Entry);
            }

            return result;
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
                    std::cout << "(ix: " << mx
                              << ", fk: " << model_tree[level][mx].first_key
                              << ", n: " << model_tree[level][mx].n
                              << "), ";
                }
                std::cout << std::endl;
                level--;
            }
        }
    };

#pragma push pack 1

    template <typename K, typename V>
    struct BufferedPGMIndex<K, V>::Model
    {
        const BufferedPGMIndex<K, V> *super;
        size_t n;
        K first_key;
        float slope;
        int32_t intercept;
        size_t num_inplaces = 0;

        // Constructor one: specify a model manually (usually used to make dummy models)
        Model(
            const BufferedPGMIndex<K, V> *super,
            size_t n,
            K first_key,
            float slope, int32_t intercept) : super(super), n(n),
                                              first_key(first_key),
                                              slope(slope),
                                              intercept(intercept) {}

        // Constructor two (the useful one): specify a model using a "canonical segment" (from original index)
        Model(
            const BufferedPGMIndex<K, V> *super,
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