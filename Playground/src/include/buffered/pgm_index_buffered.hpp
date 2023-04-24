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

namespace pgm
{

    /**
     * A space-efficient index that enables fast search operations on a sorted sequence of @c n numbers.
     *
     * A search returns a struct @ref ApproxPos containing an approximate position of the sought key in the sequence and
     * the bounds of a range of size 2*Epsilon+1 where the sought key is guaranteed to be found if present.
     * If the key is not present, the range is guaranteed to contain a key that is not less than (i.e. greater or equal to)
     * the sought key, or @c n if no such key is found.
     * In the case of repeated keys, the index finds the position of the first occurrence of a key.
     *
     * The @p Epsilon template parameter should be set according to the desired space-time trade-off. A smaller value
     * makes the estimation more precise and the range smaller but at the cost of increased space usage.
     *
     * Internally the index uses a succinct piecewise linear mapping from keys to their position in the sorted order.
     * This mapping is represented as a sequence of linear models (segments) which, if @p EpsilonRecursive is not zero, are
     * themselves recursively indexed by other piecewise linear mappings.
     *
     * Inserts are handled by keeping a buffer on top of every segment of keys that "want" to live in this segment
     * but can't because their inclusion would destroy the Epsilon bounds of the segment. More logic concerning
     * the process of merging and deleting is coming soon.
     *
     * @tparam K the type of the indexed keys
     * @tparam V the type of the indexed values
     * @tparam Epsilon controls the size of the returned search range
     * @tparam EpsilonRecursive controls the size of the search range in the internal structure
     * @tparam The floating-point type to use for slopes
     */
    template <typename K, typename V>
    class BufferedPGMIndex
    {
    public:
        struct Segment;
        struct Iterator;
        using Level = std::vector<Segment>;
        // Convenient shorthand for type of data and buffer
        using Entry = std::pair<K, V>;
        using EntryVector = std::vector<Entry>;
        using segment_iterator = typename std::vector<Segment>::const_iterator;
        using data_iterator = typename EntryVector::const_iterator;

        /**
         * A struct that stores the result of a query to a @ref PGMIndex, that is, a range [@ref lo, @ref hi)
         * centered around an approximate position @ref pos of the sought key.
         * NOTE: Similar to ApproxPos of the classic index, but uses iterators instead
         */
        struct BufferApproxPos
        {
            Iterator pos; ///< The approximate position of the key.
            Iterator lo;  ///< The lower bound of the range.
            Iterator hi;  ///< The upper bound of the range.
        };

        size_t n; ///< The number of elements this index was built on.
        std::vector<Level> levels;

        /**
         * Constructs the model given data.
         * @param first the beginning of the data (iterator)
         * @param last the end of the data (iterator)
         * @param epsilon the epsilon value to use for searching for data
         * @param epsilon_recursive the epsilon value to use for searching for segments higher up the tree
         * @param segments the vector of segments to build the index on
         * @param levels_offsets the vector of offsets to build the index on
         */
        template <typename RandomIt>
        void build(RandomIt first, RandomIt last, size_t epsilon, size_t epsilon_recursive)
        {
            auto n = (size_t)std::distance(first, last);
            if (n == 0)
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

            Level base_level;
            base_level.reserve(n / epsilon_value * epsilon_value);

            // Ignores the last element if the key is the max (sentinel) value
            auto ignore_last = std::prev(last)->first == std::numeric_limits<K>::max(); // max() is the sentinel value
            auto last_n = n - ignore_last;
            last -= ignore_last;

            auto build_level = [&](auto epsilon, auto in_fun, auto out_fun)
            {
                auto n_segments = internal::make_segmentation_par(last_n, epsilon, in_fun, out_fun);
                return n_segments;
            };

            // Variables for keeping track of the actual data that will float into
            // the leaf segments
            size_t cur_seg_size = 0;
            RandomIt seg_first = first;
            RandomIt seg_last = first;

            auto in_fun = [&](auto i)
            {
                auto x = keys[i];
                if (cur_seg_size == 0)
                {
                    seg_first = std::next(first, i);
                }
                cur_seg_size++;
                seg_last = std::next(first, i + 1);
                return std::pair<K, size_t>(x, cur_seg_size);
            };

            auto out_fun = [&](auto cs)
            {
                cur_seg_size = 0;
                base_level.emplace_back(this, cs, seg_first, seg_last);
            };
            last_n = build_level(epsilon, in_fun, out_fun);

            // Build upper levels
            levels.push_back(base_level);
            Level last_level = base_level;
            Level next_level;
            while (epsilon_recursive && last_n > 1)
            {
                next_level.clear();
                std::vector<size_t> new_sizes = {0};
                auto in_fun_rec = [&](auto i)
                {
                    return std::pair<K, size_t>(last_level[i].first_x, new_sizes.back()++);
                };
                auto out_fun_rec = [&](auto cs)
                {
                    if (new_sizes.back() > 0)
                    {
                        Segment new_segment = Segment(this, cs, new_sizes.back());
                        new_sizes.push_back(0);
                        next_level.push_back(new_segment);
                    }
                };
                last_n = build_level(epsilon_recursive, in_fun_rec, out_fun_rec);
                levels.push_back(next_level);
                last_level = next_level;
            }
        }

        /**
         * Helper function that returns the index of the segment in a given level that should be used
         * to index a key.
         */
        size_t by_level_segment_ix_for_key(const K &key, size_t goal_level)
        {
            size_t level = height() - 1;
            Segment cur_seg = levels[level][0];
            size_t new_ix = 0;
            while (goal_level < level)
            {
                size_t pred_ix = cur_seg.predict_pos(key);
                pred_ix += get_internal_offset_into_level(level, new_ix);
                size_t lowest_ix = pred_ix > epsilon_recursive_value + 2 ? pred_ix - epsilon_recursive_value - 2 : 0;
                size_t highest_ix = pred_ix + epsilon_recursive_value + 3 < levels[level - 1].size() ? pred_ix + epsilon_recursive_value + 3 : levels[level - 1].size();
                // TODO: Make this binary search to go faster
                // Honestly doesn't really matter unless EpsilonRecursive is big, which it usually isn't.
                lowest_ix = lowest_ix >= highest_ix ? highest_ix - 1 : lowest_ix;
                new_ix = lowest_ix;
                size_t check_ix = lowest_ix + 1;
                while (check_ix < highest_ix && key > levels[level - 1][check_ix].get_first_proper_key())
                {
                    // Go until the first segment where _THE NEXT_ segment's first key is greater than the key
                    // we're looking for. Setup in such a way that it will never return the sentinel segment,
                    // only the last one with data.
                    new_ix++;
                    check_ix++;
                }
                cur_seg = levels[level - 1][new_ix];
                level--;
            }
            return new_ix;
        }

        /**
         * Returns the segment responsible for a given key, that is, the rightmost segment having key <= the sought key.
         * @param key the value of the element to search for
         * @return an iterator to the segment responsible for the given key
         */
        auto segment_for_key(const K &key)
        {
            return levels[0].begin() + by_level_segment_ix_for_key(key, 0);
        }

        /**
         * Returns the segment responsible for a given key, that is, the rightmost segment having key <= the sought key.
         * NOTE: Needed because on inserts we need a mutable non-const type
         * @param key the value of the element to search for
         * @return an iterator to the segment responsible for the given key
         */
        auto mutable_segment_for_key(const K &key)
        {
            return levels[0].begin() + by_level_segment_ix_for_key(key, 0);
        }

    public:
        size_t epsilon_value = 128;
        size_t epsilon_recursive_value = 16;
        float fill_ratio = 0.75;
        float fill_ratio_recursive = 0.75;
        size_t reduced_epsilon_value = (size_t)((float)epsilon_value * fill_ratio);
        size_t reduced_epsilon_recursive_value = (size_t)((float)epsilon_recursive_value * fill_ratio);
        size_t max_buffer_size = 512;
        size_t split_neighborhood = 3;

        // Helper function to reset these reduced values when passing in something into the constructor
        void reset_reduced_values()
        {
            reduced_epsilon_value = (size_t)((float)epsilon_value * fill_ratio);
            reduced_epsilon_recursive_value = (size_t)((float)epsilon_recursive_value * fill_ratio);
        }

        /**
         * Constructs the index on the sorted keys in the range [first, last).
         * NOTE: Constructs the index obeying reduced epsilon, not the original epsilon.
         * This allows us to be confident that each segment will be able to absorb at least a certain
         * number of inserts.
         * @param first, last the range containing the sorted keys to be indexed
         */
        template <typename RandomIt>
        BufferedPGMIndex(
            RandomIt first,
            RandomIt last,
            size_t epsilon = 128,
            size_t epsilon_recursive = 16,
            float fill_ratio = 0.75,
            float fill_ratio_recursive = 0.75,
            size_t max_buffer_size = 512, size_t split_neighborhood = 4) : epsilon_value(epsilon),
                                                                           epsilon_recursive_value(epsilon_recursive),
                                                                           fill_ratio(fill_ratio),
                                                                           fill_ratio_recursive(fill_ratio_recursive),
                                                                           max_buffer_size(max_buffer_size),
                                                                           split_neighborhood(split_neighborhood),
                                                                           n(std::distance(first, last))
        {
            reset_reduced_values();
            build(first, last, reduced_epsilon_value, reduced_epsilon_recursive_value);
        }

        /**
         * Helper function to get internal offsets
         */
        size_t get_internal_offset_into_level(size_t level, size_t seg_ix)
        {
            size_t sum = 0;
            for (size_t ix = 0; ix < seg_ix && ix < levels[level].size(); ++ix)
            {
                sum += levels[level][ix].n;
            }
            return sum;
        }

        /**
         * Returns the approximate position and the range where @p key can be found.
         * @param key the value of the element to search for
         * @return a struct with the approximate position and bounds of the range
         */
        BufferApproxPos search(const K &key)
        {
            auto seg_iter = segment_for_key(key);
            auto ix = seg_iter->predict_pos(key);
            auto data_iter = seg_iter->ix_to_data_iterator(ix);
            Iterator pos = Iterator(this, seg_iter, data_iter);
            Iterator lo = Iterator(pos);
            Iterator hi = Iterator(pos);
            lo.go_back_by(epsilon_value + 2);
            hi.advance_by(epsilon_value + 3);
            return {pos, lo, hi};
        }

        /**
         * Finds the entry of an element with key equivalent to @p key. Returns an iterator pointing to that
         * entry.
         * NOTE: Does not search in the insert/delete buffers!
         * @param key the value of the element to search for
         * @return an iterator to the entry with key equivalent to @p key, or end() if no such element is found
         */
        Iterator findIterator(const K &key)
        {
            BufferApproxPos range = search(key);
            Iterator it = range.lo;
            while (it != range.hi)
            {
                Entry p = *it;
                if (p.first == key)
                {
                    return it;
                }
                it++;
            }
            return end();
        }

        /**
         * Finds the entry of an element with key equivalent to @p key.
         * @param key the value of the element to search for
         * @return the entry of the element with key equivalent to @p key, or the maximum value of @p K V if no such element is found
         */
        Entry findEntry(const K &key)
        {
            Iterator it = findIterator(key);
            if (it != end())
            {
                return *it;
            }
            auto seg = *segment_for_key(key);
            for (auto &p : seg.buffer)
            {
                if (p.first == key)
                {
                    return p;
                }
            }
            return std::make_pair(std::numeric_limits<K>::max(), std::numeric_limits<V>::max());
        }

        V find(const K &key)
        {
            return findEntry(key).second;
        }

        /**
         * Inserts a key-value pair into the index.
         * @param key the key of the element to insert
         * @param value the value of the element to insert
         */
        void insert(K key, V value)
        {
            size_t seg_ix = by_level_segment_ix_for_key(key, 0);
            Entry e = std::make_pair(key, value);
            bool needs_split = levels[0][seg_ix].insert(e);
            if (needs_split)
            {
                split(0, seg_ix);
            }
        }

        /**
         * A helper function for handling inserts at non-base levels (level > 0).
         * Note that during split, we can produce an arbitrary number of new segments. Internal
         * segments have no insert buffers, but are trained at a lower epsilon bound, meaning they
         * can absorb some but not all inserts.
         * @param split_level - The level that the split happened at
         * @param low_split_ix - The index of the lowest segment that participated in the retrain
         * @param high_split_ix - The index of the highest segment that participated in the retrain
         * @param new_segs - The new segments generated by retraining this segment on data + buffer
         */
        void internal_insert(size_t split_level, size_t low_split_ix, size_t high_split_ix, std::vector<Segment> new_segs)
        {
            // NOTE: For now we are forcing that the window not include segments with different parents
            // With more sophisticated code this may be relaxed in the future
            // A representative element from the original segment (used to find indexes)
            K first_key = levels[split_level][low_split_ix].get_first_proper_key();
            if (split_level >= height() - 1)
            {
                // Root node
                // First just update the level
                levels[split_level].erase(levels[split_level].begin() + low_split_ix, levels[split_level].begin() + high_split_ix);
                levels[split_level].insert(levels[split_level].begin() + low_split_ix, new_segs.begin(), new_segs.end());
                size_t last_size = new_segs.size();
                while (last_size > 1)
                {
                    std::vector<size_t> new_sizes = {0};
                    std::vector<Segment> rec_new_segs;
                    auto in_fun = [&](auto i)
                    {
                        new_sizes.back()++;
                        return std::pair<K, size_t>(new_segs[i].first_x, new_sizes.back());
                    };
                    auto out_fun = [&](auto cs)
                    {
                        if (new_sizes.back() > 0)
                        {
                            Segment new_seg = Segment(this, cs, new_sizes.back());
                            rec_new_segs.push_back(new_seg);
                            new_sizes.push_back(0);
                        }
                    };
                    auto build_segments = [&](auto in_fun, auto out_fun)
                    {
                        auto n_segments = internal::make_segmentation_par(last_size, reduced_epsilon_recursive_value, in_fun, out_fun);
                        return n_segments;
                    };
                    last_size = build_segments(in_fun, out_fun);
                    levels.push_back(rec_new_segs);
                    new_segs = rec_new_segs;
                }
            }
            else
            {
                // If it's not the root, update the parent and the level
                // The index of the parent of this segment in the level above
                size_t parent_ix = by_level_segment_ix_for_key(first_key, split_level + 1);
                Segment *parent_seg = &levels[split_level + 1][parent_ix];

                // Previously, the parent indexed high_split_ix - low_split_ix
                // Now, it indexes new_segs.size()
                parent_seg->n += new_segs.size() - (high_split_ix - low_split_ix);

                // The first question we must answer is whether or not this internal segment
                // can absorb these new segments.
                bool can_absorb = reduced_epsilon_recursive_value + parent_seg->num_inplaces + new_segs.size() - 1 < epsilon_recursive_value;
                if (can_absorb)
                {
                    parent_seg->num_inplaces += new_segs.size() - 1;
                    n += new_segs.size() - 1;
                }
                // Now do the same thing but for the actual level undergoing the split
                // TODO: Also make this cleaner using std stuff
                std::vector<Segment> new_split_level;
                for (size_t ix = 0; ix < low_split_ix; ++ix)
                {
                    new_split_level.push_back(levels[split_level][ix]);
                }
                for (auto s : new_segs)
                {
                    new_split_level.push_back(s);
                }
                for (size_t ix = high_split_ix; ix < levels[split_level].size(); ++ix)
                {
                    new_split_level.push_back(levels[split_level][ix]);
                }
                levels[split_level] = new_split_level;
                if (!can_absorb)
                {
                    // We need to split this node as well.
                    split(split_level + 1, parent_ix);
                }
            }
        }

        std::pair<size_t, size_t> get_split_window(size_t split_level, size_t splitting_ix)
        {
            K og_first_key = levels[split_level][splitting_ix].get_first_proper_key();
            size_t og_parent_ix = by_level_segment_ix_for_key(og_first_key, split_level + 1);
            // First find the low ix
            size_t low_seg_ix = splitting_ix;
            size_t low_check_ix = low_seg_ix;
            K low_first_key = levels[split_level][low_check_ix].get_first_proper_key();
            size_t low_parent_ix = by_level_segment_ix_for_key(low_first_key, split_level + 1);
            while (low_check_ix > 0 && splitting_ix - low_seg_ix < split_neighborhood)
            {
                low_check_ix--;
                low_first_key = levels[split_level][low_check_ix].get_first_proper_key();
                low_parent_ix = by_level_segment_ix_for_key(low_first_key, split_level + 1);
                if (low_parent_ix == og_parent_ix)
                {
                    low_seg_ix = low_check_ix;
                }
                else
                {
                    break;
                }
            }
            size_t high_seg_ix = splitting_ix + 1;
            if (splitting_ix >= levels[split_level].size() - 1)
            {
                high_seg_ix = levels[split_level].size();
            }
            else
            {
                K high_first_key = levels[split_level][high_seg_ix].get_first_proper_key();
                size_t high_parent_ix = by_level_segment_ix_for_key(high_first_key, split_level + 1);
                while (high_seg_ix < levels[split_level].size() && high_seg_ix - splitting_ix < split_neighborhood)
                {
                    high_seg_ix++;
                    if (high_seg_ix == levels[split_level].size())
                    {
                        break;
                    }
                    high_first_key = levels[split_level][high_seg_ix].get_first_proper_key();
                    high_parent_ix = by_level_segment_ix_for_key(high_first_key, split_level + 1);
                    if (high_parent_ix != og_parent_ix)
                    {
                        break;
                    }
                }
            }
            return std::make_pair(low_seg_ix, high_seg_ix);
        }

        /**
         * This function helps answer the question: given an internal segment, I want an
         * iterator that points to it's first child in the level below it.
         */
        segment_iterator get_first_child(size_t level, size_t seg_ix)
        {
            if (level <= 0)
            {
                throw std::invalid_argument("Must be an internal segment to have children");
            }
            size_t offset = 0;
            for (size_t ix = 0; ix < seg_ix; ++ix)
            {
                offset += levels[level][ix].n;
            }
            return levels[level - 1].begin() + offset;
        }

        /**
         * A helper function to split a segment (meaning retrain it on the data + buffer + neighbors)
         * and recursively trigger necessary inserts up the tree
         * @param splitting_ix - The index of the segment that is being retrained / split
         */
        void split(size_t split_level, size_t splitting_ix)
        {
            // First determine the window that will group up to participate in the merge
            auto [low_seg_ix, high_seg_ix] = get_split_window(split_level, splitting_ix);
            bool is_internal = split_level > 0;
            // How many total elements are going to be in this newly made thing
            size_t n = 0;
            for (size_t seg_ix = low_seg_ix; seg_ix < high_seg_ix; ++seg_ix)
            {
                n += levels[split_level][seg_ix].n + levels[split_level][seg_ix].buffer.size();
            }
            // Stuff used for splitting at the leaf level
            segment_iterator seg_iter = levels[split_level].begin() + low_seg_ix;
            size_t proper_ix = 0; // Index into proper data
            size_t buffer_ix = 0; // Index into buffer data
            std::vector<Entry> temporary_fill;
            // Stuff used for splitting at internal segments
            segment_iterator child_iter = is_internal ? get_first_child(split_level, low_seg_ix) : seg_iter;
            std::vector<size_t> new_sizes = {0};
            // Common stuff
            std::vector<Segment> new_segs;
            auto in_fun = [&](auto i)
            {
                // What will get passed down to the piecewise linear model
                std::pair<K, size_t> inserting;

                if (!is_internal)
                {
                    // If it's not internal, it's a LEAF insert, in which case we need to look inside
                    // the data and the buffers and be careful we are getting the right element
                    if (proper_ix >= seg_iter->n && buffer_ix >= seg_iter->buffer.size())
                    {
                        // We have ran out of data to read from this segment, move on to next
                        proper_ix = 0;
                        buffer_ix = 0;
                        seg_iter++;
                    }
                    // Is the next insert coming from proper data or buffer
                    bool is_proper;
                    if (proper_ix >= seg_iter->n)
                    {
                        is_proper = false;
                    }
                    else if (buffer_ix >= seg_iter->buffer.size())
                    {
                        is_proper = true;
                    }
                    else
                    {
                        K next_proper = seg_iter->data[proper_ix].first;
                        K next_buffer = seg_iter->buffer[buffer_ix].first;
                        is_proper = next_proper < next_buffer;
                    }
                    // Get the entry, put it in the temp array, return the key with rank
                    Entry e = is_proper ? seg_iter->data[proper_ix++] : seg_iter->buffer[buffer_ix++];
                    temporary_fill.push_back(e);
                    inserting.first = e.first;
                    inserting.second = temporary_fill.size();
                }
                else
                {
                    inserting.first = (child_iter++)->first_x;
                    inserting.second = new_sizes.back()++;
                }

                return inserting;
            };
            auto out_fun = [&](auto cs)
            {
                if (!is_internal)
                {
                    if (temporary_fill.size() > 0)
                    {
                        Segment new_seg = Segment(this, cs, temporary_fill.begin(), temporary_fill.end());
                        new_segs.push_back(new_seg);
                    }
                    temporary_fill.clear();
                }
                else
                {
                    if (new_sizes.back() > 0)
                    {
                        Segment new_seg = Segment(this, cs, new_sizes.back());
                        new_segs.push_back(new_seg);
                        new_sizes.push_back(0);
                    }
                }
            };
            auto build_segments = [&](auto in_fun, auto out_fun)
            {
                auto n_segments = internal::make_segmentation_par(n, is_internal ? reduced_epsilon_recursive_value : reduced_epsilon_value, in_fun, out_fun);
                return n_segments;
            };
            size_t n_segments = build_segments(in_fun, out_fun);
            internal_insert(split_level, low_seg_ix, high_seg_ix, new_segs);
        }

        /**
         * Returns the number of segments in the last level of the index.
         * @return the number of segments
         */
        size_t segments_count() const
        {
            return levels.empty() ? 0 : levels[0].size();
        }

        /**
         * Returns the number of segments in the full index, including internal
         */
        size_t full_segments_count() const
        {
            size_t sum = 0;
            for (auto &l : levels)
            {
                sum += l.size();
            }
            return sum;
        }

        /**
         * Returns the number of levels of the index.
         * @return the number of levels of the index
         */
        size_t height() const
        {
            return levels.size();
        }

        /**
         * Returns the size of the index in bytes.
         * @return the size of the index in bytes
         */
        size_t size_in_bytes() const
        {
            // TODO: Fix
            return 0;
        }

        Iterator begin() const
        {
            auto first_seg = levels[0].begin();
            return Iterator(this, first_seg, first_seg->data.begin());
        }
        Iterator end() const
        {
            auto last_seg = levels[0].end();
            return Iterator(this, last_seg, last_seg->data.end());
        }

        void print_tree(int lowest_level = 0)
        {
            std::cout << "Tree size: " << height() << " levels, " << full_segments_count() << " segments" << std::endl;
            int level = height() - 1;
            for (level; level >= lowest_level; --level)
            {
                std::cout << "Level: " << level << std::endl;
                for (size_t cur_ix = 0; cur_ix < levels[level].size(); ++cur_ix)
                {
                    std::cout << "(ix:" << cur_ix << ", " << levels[level][cur_ix].get_first_proper_key() << ", sz:" << levels[level][cur_ix].n << ")"
                              << " - ";
                }
                std::cout << std::endl;
            }
        }
    };

#pragma pack(push, 1)

    template <typename K, typename V>
    struct BufferedPGMIndex<K, V>::Segment
    {
        friend class BufferedPGMIndex;
        using buffered_pgm_type = BufferedPGMIndex<K, V>;

        const buffered_pgm_type *super; ///< The index this segment belongs to
        size_t n;                       ///< How many non-buffer entries are indexed by this segment
        K first_x;                      ///< The first x value (key) in this segment
        float slope;                    ///< The slope of the segment.
        int32_t intercept;              ///< The intercept of the segment.

        bool is_internal; ///< Is this an internal segment?
        // The below `data` and `buffer` only exist on leaf segments
        EntryVector data;   ///< The data stored in this segment in sorted order by key
        EntryVector buffer; ///< A buffer of inserts waiting to come to this segment

        // Information about splits
        size_t num_inplaces = 0; ///< The number of in-place inserts that have been performed on this segment

        Segment(
            const buffered_pgm_type *super,
            size_t n,
            K first_x,
            float slope,
            int32_t intercept,
            bool is_internal) : super(super),
                                n(n),
                                first_x(first_x),
                                slope(slope),
                                intercept(intercept),
                                is_internal(is_internal){};

        // Constructor for a LEAF segment that takes a canonical segment
        template <typename RandomIt>
        explicit Segment(
            const buffered_pgm_type *super,
            const typename internal::OptimalPiecewiseLinearModel<K, size_t>::CanonicalSegment &cs,
            RandomIt first, RandomIt last)
            : super(super)
        {
            is_internal = false;
            first_x = cs.get_first_x();
            auto [cs_slope, cs_intercept] = cs.get_floating_point_segment(first_x);
            if (cs_intercept > std::numeric_limits<decltype(intercept)>::max())
                throw std::overflow_error("Change the type of Segment::intercept to int64");
            slope = cs_slope;
            intercept = cs_intercept;
            n = std::distance(first, last);
            RandomIt cur = first;
            std::copy(first, last, std::back_inserter(data));
            buffer.reserve(super->max_buffer_size);
        }

        // Constructor for an INTERNAL segment that takes a canonical segment
        explicit Segment(
            const buffered_pgm_type *super,
            const typename internal::OptimalPiecewiseLinearModel<K, size_t>::CanonicalSegment &cs,
            size_t n)
            : super(super),
              n(n)
        {
            is_internal = true;
            first_x = cs.get_first_x();
            auto [cs_slope, cs_intercept] = cs.get_floating_point_segment(first_x);
            if (cs_intercept > std::numeric_limits<decltype(intercept)>::max())
                throw std::overflow_error("Change the type of Segment::intercept to int64");
            slope = cs_slope;
            intercept = cs_intercept;
        }

        friend inline bool operator<(const Segment &s, const K &k) { return s.first_x < k; }
        friend inline bool operator<(const K &k, const Segment &s) { return k < s.first_x; }
        friend inline bool operator<(const Segment &s, const Segment &t) { return s.first_x < t.first_x; }

        operator K() { return first_x; };

        /**
         * Returns the approximate position of the specified key.
         * @param k the key whose position must be approximated
         * @return the approximate position of the specified key
         */
        inline size_t operator()(const K &k) const
        {
            auto pos = int64_t(slope * (k - first_x)) + intercept;
            pos = pos > 0 ? size_t(pos) : 0ull;
            pos = pos < n ? pos : n - 1;
            return pos;
        }

        /**
         * Does the exact same as the above function but is more readable
         */
        inline size_t predict_pos(const K &k) const
        {
            return this->operator()(k);
        }

        /**
         * Checks if the buffer can absorb another out of place write
         * @return true if the buffer can absorb another out of place write
         */
        inline bool buffer_has_space() const
        {
            return buffer.size() >= (size_t)((float)super->max_buffer_size);
        }

        inline data_iterator ix_to_data_iterator(size_t offset) const
        {
            if (offset < 0)
            {
                return data.cbegin();
            }
            if (offset >= n)
            {
                return std::prev(data.cend());
            }
            return data.cbegin() + offset;
        }

        /**
         * A helper function to return the first key in the segment
         * NOTE: Does not include buffer!
         */
        K get_first_proper_key() const
        {
            return first_x;
        }

        /**
         * A function to perform an insert into a LEAF segment. Returns a boolean indicating whether the
         * buffer is full _after_ the insert (in which case the parent should trigger
         * a retrain).
         * NOTE: If the key already exists it updates the value
         * @return true (parent needs to retrain) or false
         */
        bool insert(const Entry &e)
        {
            if (is_internal)
            {
                throw std::invalid_argument("Attempted to segment insert into an internal segment");
            }
            // First check if the key already exists in the buffer
            auto existing = std::lower_bound(buffer.begin(), buffer.end(), e);
            if (existing != buffer.end() && existing->first == e.first)
            {
                existing->second = e.second;
                return false;
            }
            // Then see if it already exists "properly" in the segment
            existing = std::lower_bound(data.begin(), data.end(), e);
            if (existing != data.end() && existing->first == e.first)
            {
                existing->second = e.second;
                return false;
            }

            // It does not exist in the segment!
            // If we can't take any more in place updates send it to the buffer
            bool is_buffered_insert = super->fill_ratio < 1;
            if (!is_buffered_insert)
            {
                // Note that we care most about the case when we support no in place inserts
                // This if statement just reduces the amount of silly checking we must do
                // when we know that we are just going to end up shoving it to the buffer anyway
                if (super->reduced_epsilon_value + num_inplaces >= super->epsilon_value)
                {
                    // If there isn't space for an in-place insert, make it buffered
                    is_buffered_insert = true;
                }
                // OR if it would get inserted at the beginning make it buffered
                /*
                TODO: I have a hunch this code can be deleted, leaving as a comment in case but
                if all goes correctly it should be impossible to ever reach this code
                if (existing == data.begin())
                {
                    // If the data would end up at the beginning of a segment, make it buffered
                    is_buffered_insert = true;
                }
                */

                size_t predicted_pos = predict_pos(e.first);
                size_t actual_pos = std::distance(data.begin(), existing);
                size_t difference = predicted_pos > actual_pos ? predicted_pos - actual_pos : actual_pos - predicted_pos;
                if (difference > super->epsilon_value)
                {
                    // If the predicted position is too far away from the actual position, make it buffered
                    // NOTE: THIS is the true "out-of-place" insert
                    is_buffered_insert = true;
                }
            }

            if (is_buffered_insert)
            {
                // Insert into sorted position in buffer
                auto location = std::lower_bound(buffer.begin(), buffer.end(), e);
                buffer.insert(location, e);
            }
            else
            {
                // Insert into sorted position in data
                data.insert(existing, e);
                num_inplaces++;
                n++;
            }
            return buffer_has_space();
        }
    };

    template <typename K, typename V>
    struct BufferedPGMIndex<K, V>::Iterator
    {
        friend class BufferedPGMIndex;
        using buffered_pgm_type = BufferedPGMIndex<K, V>;

        struct Cursor
        {
            segment_iterator seg_iter;
            data_iterator data_iter;
            Cursor() = default;
            Cursor(
                const segment_iterator seg_iter) : segment_iterator(seg_iter){};
        };

        const buffered_pgm_type *super; //< Pointer to the buffered_pgm that is being iterated
        Cursor current;                 //< The current cursor

        void advance()
        {
            auto end = super->end();
            if (current.seg_iter == end.current.seg_iter && current.data_iter == end.current.data_iter)
            {
                return;
            }
            current.data_iter++;
            if (current.data_iter == current.seg_iter->data.end())
            {
                current.seg_iter++;
                current.data_iter = current.seg_iter->data.begin();
            }
        }

        void advance_by(size_t n)
        {
            while (n--)
                advance();
        }

        void go_back()
        {
            auto begin = super->begin();
            if (current.seg_iter == begin.current.seg_iter && current.data_iter == begin.current.data_iter)
            {
                return;
            }
            if (current.data_iter != current.seg_iter->data.begin())
            {
                current.data_iter--;
            }
            else
            {
                current.seg_iter--;
                current.data_iter = current.seg_iter->data.end();
                current.data_iter--;
            }
        }

        void go_back_by(size_t n)
        {
            while (n--)
                go_back();
        }

        Iterator &operator++()
        {
            advance();
            return *this;
        }

        Iterator operator++(int)
        {
            Iterator tmp = *this;
            advance();
            return tmp;
        }

        Iterator &operator+=(size_t n)
        {
            advance_by(n);
            return *this;
        }

        Iterator &operator--()
        {
            go_back();
            return *this;
        }

        Iterator operator--(int)
        {
            Iterator tmp = *this;
            go_back();
            return tmp;
        }

        Iterator &operator-=(size_t n)
        {
            go_back_by(n);
            return *this;
        }

        bool operator==(const Iterator &other) const
        {
            return current.seg_iter == other.current.seg_iter && current.data_iter == other.current.data_iter;
        }

        bool operator!=(const Iterator &other) const
        {
            return !(*this == other);
        }

        const Entry &operator*() const
        {
            return *current.data_iter;
        }

        const Entry *operator->() const
        {
            return &(*current.data_iter);
        }

        Iterator() = default;
        // Defining an iterator by an iterator into the segment list and an iterator into the data list for that segment
        Iterator(
            const buffered_pgm_type *super,
            const segment_iterator seg_iter,
            const data_iterator data_iter) : super(super)
        {
            current = Cursor();
            current.seg_iter = seg_iter;
            current.data_iter = data_iter;
        };
        // For copying an iterator
        Iterator(const Iterator &other) : super(other.super), current(other.current){};
    };

#pragma pack(pop)

}