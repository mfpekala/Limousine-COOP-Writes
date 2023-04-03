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

namespace pgm {

#define PGM_SUB_EPS(x, epsilon) ((x) <= (epsilon) ? 0 : ((x) - (epsilon)))
#define PGM_ADD_EPS(x, epsilon, size) ((x) + (epsilon) + 2 >= (size) ? (size) : (x) + (epsilon) + 2)

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
template<typename K, typename V, size_t Epsilon = 64, size_t EpsilonRecursive = 8, typename Floating = float>
class BufferedPGMIndex {
public:
    static_assert(Epsilon > 0);
    struct Segment;
    struct Iterator;
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
    struct BufferApproxPos {
        Iterator pos; ///< The approximate position of the key.
        Iterator lo;  ///< The lower bound of the range.
        Iterator hi;  ///< The upper bound of the range.
    };

    size_t n;                           ///< The number of elements this index was built on.
    std::vector<Segment> segments;      ///< The segments composing the index.
    std::vector<size_t> levels_offsets; ///< The starting position of each level in segments[], in reverse order.
    float fill_ratio = 0.5;             ///< The fill ratio of the buffer, i.e. the ratio between the
                                        ///< bound the model is trained on vs expected to give.
    float fill_ratio_recursive = 0.5;   ///< The fill ratio of the buffer, i.e. the ratio between the
                                        ///< bound the model is trained on vs expected to give.

    /**
     * Constructs the model given data.
     * @param first the beginning of the data (iterator)
     * @param last the end of the data (iterator)
     * @param epsilon the epsilon value to use for searching for data
     * @param epsilon_recursive the epsilon value to use for searching for segments higher up the tree
     * @param segments the vector of segments to build the index on
     * @param levels_offsets the vector of offsets to build the index on
     */
    template<typename RandomIt>
    void build(RandomIt first, RandomIt last,
                      size_t epsilon, size_t epsilon_recursive,
                      std::vector<Segment> &segments,
                      std::vector<size_t> &levels_offsets) {
        auto n = (size_t) std::distance(first, last);
        if (n == 0)
            return;
        
        // NOTE: Potentially not the most efficient, but logically easiest to work with as we're
        // building the index
        std::vector<K> keys;
        std::vector<V> values;
        keys.reserve(n);
        values.reserve(n);
        // Could be more efficient by using std::move
        for (auto it = first; it != last; ++it) {
            keys.push_back(it->first);
            values.push_back(it->second);
        }

        levels_offsets.push_back(0);
        segments.reserve(n / (epsilon * epsilon));

        // Ignores the last element if the key is the max (sentinel) value
        auto ignore_last = std::prev(last)->first == std::numeric_limits<K>::max(); // max() is the sentinel value
        auto last_n = n - ignore_last;
        last -= ignore_last;

        auto build_level = [&](auto epsilon, auto in_fun, auto out_fun) {
            auto n_segments = internal::make_segmentation_par(last_n, epsilon, in_fun, out_fun);
            if (last_n > 1 && segments.back().slope == 0) {
                // Here we need to ensure that keys > *(last-1) are approximated to a position == prev_level_size
                // This is done by adding a segment with slope 0 and intercept last_n-1
                segments.emplace_back(this, std::prev(last)->first + 1, 0, 0);
                ++n_segments;
            }
            segments.emplace_back(this, last_n); // Add the sentinel segment
            return n_segments;
        };

        // Variables for keeping track of the actual data that will float into
        // the leaf segments
        size_t cur_seg_size = 0;
        RandomIt seg_first = first;
        RandomIt seg_last = first;

        auto in_fun = [&](auto i) {
            auto x = keys[i];
            if (cur_seg_size == 0) {
                seg_first = std::next(first, i);
            }
            cur_seg_size++;
            seg_last = std::next(first, i + 1);
            // Here there is an adjustment for inputs with duplicate keys: at the end of a run of duplicate keys equal
            // to x=first[i] such that x+1!=first[i+1], we map the values x+1,...,first[i+1]-1 to their correct rank i
            auto flag = i > 0 && i + 1u < n && x == keys[i - 1] && x != keys[i + 1] && x + 1 != keys[i + 1];
            return std::pair<K, size_t>(x + flag, cur_seg_size);
        };
        
        auto out_fun = [&](auto cs) {
            cur_seg_size = 0;
            segments.emplace_back(this, cs, seg_first, seg_last);
        };
        last_n = build_level(epsilon, in_fun, out_fun);
        levels_offsets.push_back(levels_offsets.back() + last_n + 1);

        // Recursive entries is used for populating the recursive levels of the tree
        // with the necessary pairs/data to get same segment behaviour with the correct structure
        EntryVector rec_entries;

        // Build upper levels
        while (epsilon_recursive && last_n > 1) {
            // - 2 because of the sentinel segment that exists at every level
            auto offset = levels_offsets[levels_offsets.size() - 2];
            auto in_fun_rec = [&](auto i) {
                Entry e = segments[offset + i].data[0];
                rec_entries.push_back(e);
                return std::pair<K, size_t>(e.first, rec_entries.size() - 1); 
            };
            auto out_fun_rec = [&](auto cs) {
                segments.emplace_back(this, cs, rec_entries.begin(), rec_entries.end());
                rec_entries.clear();
            };
            last_n = build_level(epsilon_recursive, in_fun_rec, out_fun_rec);
            levels_offsets.push_back(levels_offsets.back() + last_n + 1);
        }
    }

    /**
     * Helper function that returns what level a segment is in
    */
    size_t level_for_segment_ix(size_t seg_ix) {
        size_t result = 0;
        while (result < height - 1) {
            if (seg_ix < levels_offsets[result]) break;
            ++result;
        }
        return result;
    }

    /**
     * Helper function that returns the index of the segment in a given level that should be used
     * to index a key.
    */
    size_t by_level_segment_ix_for_key(const K &key, size_t goal_level) const {
        size_t level = height() - 1;
        size_t cur_seg_ix = first_segment_ix_for_level(level);
        while (goal_level < level) {
            size_t pred_ix = segments[cur_seg_ix].predict_pos(key);
            size_t floor_ix = first_segment_ix_for_level(level - 1);
            size_t ceiling_ix = last_segment_ix_for_level(level - 1);
            size_t lowest_ix = floor_ix + (EpsilonRecursive < pred_ix ? pred_ix - EpsilonRecursive : 0);
            size_t highest_ix = (floor_ix + pred_ix + EpsilonRecursive >= ceiling_ix) ? ceiling_ix : floor_ix + pred_ix + EpsilonRecursive;
            // TODO: Make this binary search to go faster
            // Honestly doesn't really matter unless EpsilonRecursive is big, which it usually isn't.
            size_t new_ix = lowest_ix;
            size_t check_ix = lowest_ix + 1;
            while (check_ix < ceiling_ix && key > segments[check_ix].get_first_proper_key()) {
                // Go until the first segment where _THE NEXT_ segment's first key is greater than the key
                // we're looking for. Setup in such a way that it will never return the sentinel segment,
                // only the last one with data.
                new_ix++;
                check_ix++;
            }
            cur_seg_ix = new_ix;
            level--;
        }
        return cur_seg_ix;
    }

    /**
    * A useful function to have during splits, will give you the index of the segment one level above the segment specified
    * by the index that points into the segment.
    * NOTE: Returns -1 if it is the root
    */
    ssize_t parent_seg_ix(size_t seg_ix) const {
        size_t level = level_for_segment_ix(seg_ix);
        if (level >= height() - 1) {
            // Root segment
            return -1;
        }
        return by_level_segment_ix_for_key(segments[seg_ix].get_first_proper_key(), level+1);
    }

    /**
     * Returns the segment responsible for a given key, that is, the rightmost segment having key <= the sought key.
     * @param key the value of the element to search for
     * @return an iterator to the segment responsible for the given key
     */
    auto segment_for_key(const K &key) const {
        if constexpr (EpsilonRecursive == 0) {
            return std::prev(std::upper_bound(segments.begin(), segments.begin() + segments_count(), key));
        }

        return segments.begin() + by_level_segment_ix_for_key(key, 0);
    }

    /**
     * Returns the segment responsible for a given key, that is, the rightmost segment having key <= the sought key.
     * NOTE: Needed because on inserts we need a mutable non-const type
     * @param key the value of the element to search for
     * @return an iterator to the segment responsible for the given key
     */
    auto mutable_segment_for_key(const K &key) {
        if constexpr (EpsilonRecursive == 0) {
            return std::prev(std::upper_bound(segments.begin(), segments.begin() + segments_count(), key));
        }

        return segments.begin() + by_level_segment_ix_for_key(key, 0);
    }

public:

    static constexpr size_t epsilon_value = Epsilon;
    size_t reduced_epsilon_value = (size_t) ((float) Epsilon * fill_ratio);
    static constexpr size_t epsilon_recursive_value = EpsilonRecursive;
    size_t reduced_epsilon_recursive_value = (size_t) ((float) EpsilonRecursive * fill_ratio);

    /**
     * Constructs an empty index.
     */
    BufferedPGMIndex() = default;

    /**
     * Constructs the index on the given sorted vector.
     * @param data the vector of keys to be indexed, must be sorted
     */
    explicit BufferedPGMIndex(const EntryVector &data) : BufferedPGMIndex(data.begin(), data.end()) {}

    /**
     * Constructs the index on the sorted keys in the range [first, last).
     * NOTE: Constructs the index obeying reduced epsilon, not the original epsilon.
     * This allows us to be confident that each segment will be able to absorb at least a certain
     * number of inserts.
     * @param first, last the range containing the sorted keys to be indexed
     */
    template<typename RandomIt>
    BufferedPGMIndex(RandomIt first, RandomIt last)
        : n(std::distance(first, last)),
          segments(),
          levels_offsets() {
        build(first, last, reduced_epsilon_value, reduced_epsilon_recursive_value, segments, levels_offsets);
    }

    /**
     * Returns the approximate position and the range where @p key can be found.
     * @param key the value of the element to search for
     * @return a struct with the approximate position and bounds of the range
     */
    BufferApproxPos search(const K &key) const {
        auto k = key;
        auto seg_iter = segment_for_key(k);
        auto ix = (*seg_iter)(k);
        auto data_iter = seg_iter->ix_to_data_iterator(ix);
        Iterator pos = Iterator(this, seg_iter, data_iter);
        Iterator lo = Iterator(pos);
        Iterator hi = Iterator(pos);
        lo.go_back_by(Epsilon + 2);
        hi.advance_by(Epsilon + 3);
        return {pos, lo, hi};
    }

    /**
     * Finds the entry of an element with key equivalent to @p key. Returns an iterator pointing to that
     * entry.
     * NOTE: Does not search in the insert/delete buffers!
     * @param key the value of the element to search for
     * @return an iterator to the entry with key equivalent to @p key, or end() if no such element is found
    */
   Iterator findIterator(const K &key) const {
        BufferApproxPos range = search(key);
        Iterator it = range.lo;
        while (it != range.hi) {
            Entry p = *it;
            if (p.first == key) {
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
    Entry findEntry(const K &key) const {
        Iterator it = findIterator(key);
        if (it != end()) {
            return *it;
        }
        auto seg = *segment_for_key(key);
        for (auto& p : seg.buffer) {
            if (p.first == key) {
                return p;
            }
        }
        return std::make_pair(std::numeric_limits<K>::max(), std::numeric_limits<V>::max());
    }

    V find(const K& key) const {
        return findEntry(key).second;
    }

    /**
     * Inserts a key-value pair into the index.
     * @param key the key of the element to insert
     * @param value the value of the element to insert
     */
    void insert(K key, V value) {
        size_t seg_ix = by_level_segment_ix_for_key(key, 0);
        Entry e = std::make_pair(key, value);
        bool needs_split = segments[seg_ix].insert(e);
        if (needs_split) {
            split(seg_ix);
        }
    }

    /**
     * 
    */
    void split(size_t splitting_ix) {
        Segment splitting_seg = segments[splitting_ix];
        bool is_internal = splitting_ix < last_segment_ix_for_level(0);
        // How many total elements are going to be in this newly made thing
        size_t n = splitting_seg.data.size() + splitting_seg.buffer.size();
        std::vector<Segment> new_segs;
        size_t proper_ix = 0; // Index into proper data
        size_t buffer_ix = 0; // Index into buffer data
        std::vector<Entry> temporary_fill;
        auto in_fun = [&](auto i) {
            // Is the next insert coming from proper data or buffer
            bool is_proper;
            if (proper_ix >= splitting_seg.data.size()) {
                is_proper = false;
            } else if (buffer_ix >= splitting_seg.buffer.size()) {
                is_proper = true;
            } else {
                K next_proper = splitting_seg[proper_ix];
                K next_buffer = splitting_seg[buffer_ix];
                is_proper = next_proper < next_buffer;
            }
            // Get the entry, put it in the temp array, return the key with rank
            Entry e = is_proper ? splitting_seg.data[proper_ix++] : splitting_seg.buffer[buffer_ix++];
            temporary_fill.push_back(e);
            return std::pair<K, size_t>(e.first, temporary_fill.size());
        };
        auto out_fun = [&](auto cs) {
            Segment new_seg = Segment(this, cs, temporary_fill.begin(), temporary_fill.end());
            new_segs.push_back(new_seg);
            temporary_fill.clear();
        };
        auto build_segments = [&](auto in_fun, auto out_fun) {
            auto n_segments = internal::make_segmentation_par(n, is_internal ? epsilon_value : reduced_epsilon_value, in_fun, out_fun);
            return n_segments;
        };
    }

    /**
     * Returns the number of segments in the last level of the index.
     * @return the number of segments
     */
    size_t segments_count() const { return segments.empty() ? 0 : levels_offsets[1] - 1; }

    /**
     * Returns the number of levels of the index.
     * @return the number of levels of the index
     */
    size_t height() const { return levels_offsets.size() - 1; }

    /**
     * Returns the size of the index in bytes.
     * @return the size of the index in bytes
     */
    size_t size_in_bytes() const { return segments.size() * sizeof(Segment) + levels_offsets.size() * sizeof(size_t); }

    Iterator begin() const { return Iterator(this, segments.begin(), segments.front().data.begin()); }
    Iterator end() const {
        const auto& last_valid_seg =std::prev(segments.end());
        return Iterator(this, last_valid_seg, last_valid_seg->data.begin()); 
    }

    /**
     * A helper function that returns the index of the first segment in a level
    */
    size_t first_segment_ix_for_level(size_t lev_num) const {
        return levels_offsets[lev_num];
    }

    /**
     * A helper function that returns the index of the last segment in a level
     * NOTE: This returns the index of a SENTINEL segment (one without real data) 
     * that's there at the max range of the key to signal the end
    */
    size_t last_segment_ix_for_level(size_t lev_num) const {
        return levels_offsets[lev_num + 1] - 1;
    }

    void print_tree() {
        std::cout << "Tree size: " << height() << " levels, " << segments.size() << " segments" << std::endl;
        // Confusing because the level to height stuff is off by one
        int level = height() - 1;
        size_t offset = levels_offsets[level];
        for (level; level >= 0; --level) {
            std::cout << "Level: " << level << std::endl;
            size_t start_ix = first_segment_ix_for_level(level);
            size_t end_ix = last_segment_ix_for_level(level);
            for (size_t cur_ix = start_ix; cur_ix < end_ix; ++cur_ix) {
                std::cout << "(ix:" << cur_ix << ", " << segments[cur_ix].get_first_proper_key() << ", " << segments[cur_ix].get_last_proper_key() << ")" << " - ";
            }
            std::cout << std::endl;
        }
    }
};

#pragma pack(push, 1)

template<typename K, typename V, size_t Epsilon, size_t EpsilonRecursive, typename Floating>
struct BufferedPGMIndex<K, V, Epsilon, EpsilonRecursive, Floating>::Segment {
    friend class BufferedPGMIndex;
    using buffered_pgm_type = BufferedPGMIndex<K, V, Epsilon, EpsilonRecursive, Floating>;

    const buffered_pgm_type *super; ///< The index this segment belongs to
    K first_x = 0;                  ///< The first x value in this segment
    Floating slope;                 ///< The slope of the segment.
    int32_t intercept;              ///< The intercept of the segment.
    size_t num_inplaces;            ///< The number of in-place inserts that have been performed on this segment
    size_t max_buffer_size = 128;   ///< The size of the buffer for out-of-place inserts
    float split_threshold = 0.75;   ///< How full does buffer need to be before trigger a split/retrain
    size_t split_neighbors = 0;     ///< How many neighbors to split with
    EntryVector data;               ///< The data stored in this segment in sorted order by key
    EntryVector buffer;             ///< A buffer of inserts waiting to come to this segment
 
    Segment() = default;

    Segment(const buffered_pgm_type *super, K first_x, Floating slope, int32_t intercept) : 
        super(super),
        first_x(first_x),
        slope(slope),
        intercept(intercept) {};

    explicit Segment(const buffered_pgm_type *super, size_t n) : super(super), slope(0), intercept(0) {
        data.push_back(std::make_pair(std::numeric_limits<K>::max(), std::numeric_limits<V>::max()));
    };

    template<typename RandomIt>
    explicit Segment(
        const buffered_pgm_type *super,
        const typename internal::OptimalPiecewiseLinearModel<K, size_t>::CanonicalSegment &cs,
        RandomIt first, RandomIt last
    )
        : super(super) {
        first_x = cs.get_first_x();
        auto[cs_slope, cs_intercept] = cs.get_floating_point_segment(first_x);
        if (cs_intercept > std::numeric_limits<decltype(intercept)>::max())
            throw std::overflow_error("Change the type of Segment::intercept to int64");
        slope = cs_slope;
        intercept = cs_intercept;
        size_t distance = std::distance(first, last);
        data.reserve(distance);
        std::copy(first, last, std::back_inserter(data));
        buffer.reserve(max_buffer_size);
    }

    friend inline bool operator<(const Segment &s, const K &k) { return s.data[0].first < k; }
    friend inline bool operator<(const K &k, const Segment &s) { return k < s.data[0].first; }
    friend inline bool operator<(const Segment &s, const Segment &t) { return s.data[0].first < t.data[0].first; }

    operator K() { return first_x; };

    /**
     * Returns the approximate position of the specified key.
     * @param k the key whose position must be approximated
     * @return the approximate position of the specified key
     */
    inline size_t operator()(const K &k) const {
        auto pos = int64_t(slope * (k - first_x)) + intercept;
        return pos > 0 ? size_t(pos) : 0ull;
    }

    /**
     * Does the exact same as the above function but is more readable
    */
    inline size_t predict_pos(const K &k) const {
        return this->operator()(k);
    }

    /**
     * Checks if the buffer can absorb another out of place write
     * @return true if the buffer can absorb another out of place write
     */
    inline bool buffer_has_space() const {
        return buffer.size() >= (size_t) ((float) max_buffer_size) * split_threshold;
    }

    data_iterator ix_to_data_iterator(size_t offset) const {
        if (offset < 0) {
            return data.cbegin();
        }
        if (offset >= data.size()) {
            return std::prev(data.cend());
        }
        return data.cbegin() + offset;
    }

    /**
     * A helper function to return the first key in the segment
     * NOTE: Does not include buffer!
    */
    K get_first_proper_key() const {
        return data.front().first;
    }

        /**
         * A helper function to return the last key in the segment
         * NOTE: Does not include buffer!
        */
    K get_last_proper_key() const {
        return data.back().first;
    }

    /**
     * A function to perform an insert into a segment. Returns a boolean indicating whether the
     * buffer is full _after_ the insert (in which case the parent should trigger
     * a retrain). 
     * NOTE: If the key already exists it updates the value
     * @return true (parent needs to retrain) or false
    */
    bool insert(const Entry &e) {
        // First check if the key already exists in the buffer
        auto existing = std::lower_bound(buffer.begin(), buffer.end(), e);
        if (existing != buffer.end() && existing->first == e.first) {
            existing->second = e.second;
            return false;
        }
        // Then see if it already exists "properly" in the segment
        existing = std::lower_bound(data.begin(), data.end(), e);
        if (existing != data.end() && existing->first == e.first) {
            existing->second = e.second;
            return false;
        }

        // It does not exist in the segment!
        // If we can't take any more in place updates send it to the buffer
        bool is_buffered_insert = false;
        if (super->reduced_epsilon + num_inplaces >= Epsilon) {
            // If there isn't space for an in-place insert, make it buffered
            is_buffered_insert = true;
        }
        // OR if it would get inserted at the beginning make it buffered
        if (existing == data.begin()) {
            // If the data would end up at the beginning of a segment, make it buffered
            is_buffered_insert = true;
        }

        size_t predicted_pos = (*this)(e.first);
        size_t actual_pos = std::distance(data.begin(), existing);
        size_t difference = predicted_pos > actual_pos ? predicted_pos - actual_pos : actual_pos - predicted_pos;
        if (difference > Epsilon) {
            // If the predicted position is too far away from the actual position, make it buffered
            // NOTE: THIS is the true "out-of-place" insert
            is_buffered_insert = true;
        }

        if (is_buffered_insert) {
            // Insert into sorted position in buffer
            auto location = std::lower_bound(buffer.begin(), buffer.end(), e);
            buffer.insert(location, e);
            return buffer_has_space();
        } else {
            // Insert into sorted position in data
            data.insert(existing, e);
            num_inplaces++;
            
            return false;
        }
        
        return buffer_has_space();
    }

    /**
     * A helper function that returns a pair of arrays for `paddingTop` and `paddingBottom`
     * as outlined in the plan of attack in the notion
     * @return a pair of arrays for `paddingTop` and `paddingBottom`
    */
   std::pair<std::vector<size_t>, std::vector<size_t>> get_padding() const {
        std::vector<size_t> padding_top;
        std::vector<size_t> padding_bottom;

        K last_key;
        for (size_t ix=0; ix < data.size(); ix++) {
            auto p = data[ix];

            if (p.first == last_key) {
                continue;
            }
            last_key = p.first;

            auto approx_pos = (*this)(p.first);
            if (approx_pos < 0) {
                approx_pos = 0;
            }
            if (approx_pos > data.size()) {
                approx_pos = data.size();
            }

            padding_top.push_back(ix + Epsilon - approx_pos);
            padding_bottom.push_back(approx_pos - (ix - Epsilon));
        }

        return std::make_pair(padding_top, padding_bottom);
    }
};

template<typename K, typename V, size_t Epsilon, size_t EpsilonRecursive, typename Floating>
struct BufferedPGMIndex<K, V, Epsilon, EpsilonRecursive, Floating>::Iterator {
    friend class BufferedPGMIndex;
    using buffered_pgm_type = BufferedPGMIndex<K, V, Epsilon, EpsilonRecursive, Floating>;

    struct Cursor {
        segment_iterator seg_iter;
        data_iterator data_iter;
        Cursor() = default;
        Cursor(
            const segment_iterator seg_iter
        ) : segment_iterator(seg_iter) {};
    };

    const buffered_pgm_type *super; //< Pointer to the buffered_pgm that is being iterated
    Cursor current;                 //< The current cursor

    void advance() {
        auto end = super->end();
        if (current.seg_iter == end.current.seg_iter && current.data_iter == end.current.data_iter) {
            return;
        }
        current.data_iter++;
        if (current.data_iter == current.seg_iter->data.end()) {
            current.seg_iter++;
            if (current.seg_iter != super->segments.end()) {
                current.data_iter = current.seg_iter->data.begin();
            } else {
                *this = super->end();
            }
        }
    }

    void advance_by(size_t n) {
        while (n--) advance();
    }

    void go_back() {
        auto begin = super->begin();
        if (current.seg_iter == begin.current.seg_iter && current.data_iter == begin.current.data_iter) {
            return;
        }
        if (current.data_iter != current.seg_iter->data.begin()) {
            current.data_iter--;
        } else {
            if (current.seg_iter != super->segments.begin()) {
                current.seg_iter--;
                current.data_iter = current.seg_iter->data.end();
                current.data_iter--;
            } else {
                *this = super->begin();
            }
        }
    }

    void go_back_by(size_t n) {
        while (n--) go_back();
    }

    Iterator &operator++() {
        advance();
        return *this;
    }

    Iterator operator++(int) {
        Iterator tmp = *this;
        advance();
        return tmp;
    }

    Iterator &operator+=(size_t n) {
        advance_by(n);
        return *this;
    }

    Iterator &operator--() {
        go_back();
        return *this;
    }

    Iterator operator--(int) {
        Iterator tmp = *this;
        go_back();
        return tmp;
    }

    Iterator &operator-=(size_t n) {
        go_back_by(n);
        return *this;
    }

    bool operator==(const Iterator &other) const {
        return current.seg_iter == other.current.seg_iter && current.data_iter == other.current.data_iter;
    }

    bool operator!=(const Iterator &other) const {
        return !(*this == other);
    }

    const Entry &operator*() const {
        return *current.data_iter;
    }

    const Entry *operator->() const {
        return &(*current.data_iter);
    }

    Iterator() = default;
    // Defining an iterator by an iterator into the segment list and an iterator into the data list for that segment
    Iterator(
        const buffered_pgm_type *super,
        const segment_iterator seg_iter,
        const data_iterator data_iter
    ) : super(super) {
        current = Cursor();
        current.seg_iter = seg_iter;
        current.data_iter = data_iter;
    };
    // For copying an iterator
    Iterator(const Iterator& other) : super(other.super), current(other.current) {};
};

#pragma pack(pop)

}