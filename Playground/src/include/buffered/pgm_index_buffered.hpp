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
template<typename K, typename V, size_t Epsilon = 64, size_t EpsilonRecursive = 4, typename Floating = float>
class BufferedPGMIndex {
protected:
    static_assert(Epsilon > 0);
    struct Segment;
    class Iterator;
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

    // NOTE: In original this is static, loosening for now
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

        // Ignores last element, not immediately clear why the secodn `last -= ignore_last` is needed
        // but in testing it seems to be necessary
        auto ignore_last = std::prev(last)->first == std::numeric_limits<K>::max(); // max() is the sentinel value
        auto last_n = n - ignore_last;
        last -= ignore_last;

        auto build_level = [&](auto epsilon, auto in_fun, auto out_fun) {
            auto n_segments = internal::make_segmentation_par(last_n, epsilon, in_fun, out_fun);
            if (last_n > 1 && segments.back().slope == 0) {
                // Here we need to ensure that keys > *(last-1) are approximated to a position == prev_level_size
                segments.emplace_back(std::prev(last)->first + 1, 0, last_n);
                ++n_segments;
            }
            segments.emplace_back(last_n); // Add the sentinel segment
            return n_segments;
        };

        bool has_first = true;
        RandomIt seg_first = first;
        RandomIt seg_last = first;

        auto in_fun = [&](auto i) {
            auto x = keys[i];
            if (!has_first) {
                seg_first = std::next(first, i);
                has_first = true;
            }
            seg_last = std::next(first, i + 1);
            // Here there is an adjustment for inputs with duplicate keys: at the end of a run of duplicate keys equal
            // to x=first[i] such that x+1!=first[i+1], we map the values x+1,...,first[i+1]-1 to their correct rank i
            auto flag = i > 0 && i + 1u < n && x == keys[i - 1] && x != keys[i + 1] && x + 1 != keys[i + 1];
            return std::pair<K, size_t>(x + flag, i);
        };
        auto out_fun = [&](auto cs) {
            // Print seg_first and seg_last
            has_first = false;
            segments.emplace_back(cs, seg_first, seg_last);
        };
        last_n = build_level(epsilon, in_fun, out_fun);
        levels_offsets.push_back(levels_offsets.back() + last_n + 1);

        // Build upper levels
        while (epsilon_recursive && last_n > 1) {
            // - 2 because of the sentinel segment that exists at every level
            auto offset = levels_offsets[levels_offsets.size() - 2];
            auto in_fun_rec = [&](auto i) { return std::pair<K, size_t>(segments[offset + i].key, i); };
            last_n = build_level(epsilon_recursive, in_fun_rec, out_fun);
            levels_offsets.push_back(levels_offsets.back() + last_n + 1);
        }
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

        auto it = segments.begin() + *(levels_offsets.end() - 2);
        for (auto l = int(height()) - 2; l >= 0; --l) {
            auto level_begin = segments.begin() + levels_offsets[l];
            auto pos = std::min<size_t>((*it)(key), std::next(it)->intercept);
            auto lo = level_begin + PGM_SUB_EPS(pos, EpsilonRecursive + 1);

            static constexpr size_t linear_search_threshold = 8 * 64 / sizeof(Segment);
            if constexpr (EpsilonRecursive <= linear_search_threshold) {
                for (; std::next(lo)->key <= key; ++lo)
                    continue;
                it = lo;
            } else {
                auto level_size = levels_offsets[l + 1] - levels_offsets[l] - 1;
                auto hi = level_begin + PGM_ADD_EPS(pos, EpsilonRecursive, level_size);
                it = std::prev(std::upper_bound(lo, hi, key));
            }
        }
        return it;
    }

    /**
     * Returns the segment responsible for a given key, that is, the rightmost segment having key <= the sought key.
     * @param key the value of the element to search for
     * @return an iterator to the segment responsible for the given key
     */
    auto mutable_segment_for_key(const K &key) {
        if constexpr (EpsilonRecursive == 0) {
            return std::prev(std::upper_bound(segments.begin(), segments.begin() + segments_count(), key));
        }

        auto it = segments.begin() + *(levels_offsets.end() - 2);
        for (auto l = int(height()) - 2; l >= 0; --l) {
            auto level_begin = segments.begin() + levels_offsets[l];
            auto pos = std::min<size_t>((*it)(key), std::next(it)->intercept);
            auto lo = level_begin + PGM_SUB_EPS(pos, EpsilonRecursive + 1);

            static constexpr size_t linear_search_threshold = 8 * 64 / sizeof(Segment);
            if constexpr (EpsilonRecursive <= linear_search_threshold) {
                for (; std::next(lo)->key <= key; ++lo)
                    continue;
                it = lo;
            } else {
                auto level_size = levels_offsets[l + 1] - levels_offsets[l] - 1;
                auto hi = level_begin + PGM_ADD_EPS(pos, EpsilonRecursive, level_size);
                it = std::prev(std::upper_bound(lo, hi, key));
            }
        }
        return it;
    }

public:

    static constexpr size_t epsilon_value = Epsilon;

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
     * @param first, last the range containing the sorted keys to be indexed
     */
    template<typename RandomIt>
    BufferedPGMIndex(RandomIt first, RandomIt last)
        : n(std::distance(first, last)),
          segments(),
          levels_offsets() {
        build(first, last, Epsilon, EpsilonRecursive, segments, levels_offsets);
    }

    /**
     * Returns the approximate position and the range where @p key can be found.
     * @param key the value of the element to search for
     * @return a struct with the approximate position and bounds of the range
     */
    BufferApproxPos search(const K &key) const {
        auto k = key;
        auto seg_iter = segment_for_key(k);
        auto ix = std::min<size_t>((*seg_iter)(k), std::next(seg_iter)->intercept);
        auto data_iter = seg_iter->ix_to_data_iterator(ix);
        Iterator pos = Iterator(this, seg_iter, data_iter);
        Iterator lo = Iterator(pos);
        Iterator hi = Iterator(pos);
        lo.go_back_by(Epsilon);
        hi.advance_by(Epsilon);
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

    /**
     * 
    */
    V find(const K& key) const {
        return findEntry(key).second;
    }

    /**
     * Inserts a key-value pair into the index.
     * @param key the key of the element to insert
     * @param value the value of the element to insert
     */
    void insert(const K& key, const V& value) {
        // NOTE: Just a toy proof-of-concept implementation at present
        // Just goes and puts stuff in buffer
        // Does not rebuild the index
        // Does not scheudle lazy merge (yet)
        auto seg = mutable_segment_for_key(key);
        seg->buffer.push_back(std::make_pair(key, value));
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
    Iterator end() const { return Iterator(this, segments.end(), segments.back().data.end()); }
};

#pragma pack(push, 1)

template<typename K, typename V, size_t Epsilon, size_t EpsilonRecursive, typename Floating>
struct BufferedPGMIndex<K, V, Epsilon, EpsilonRecursive, Floating>::Segment {
    K key;                  ///< The first key that the segment indexes.
    Floating slope;         ///< The slope of the segment.
    int32_t intercept;      ///< The intercept of the segment.
    EntryVector data;       ///< The data stored in this segment in sorted order by key
    EntryVector buffer;     ///< A buffer of inserts waiting to come to this segment

    Segment() = default;

    Segment(K key, Floating slope, int32_t intercept) : key(key), slope(slope), intercept(intercept) {};

    explicit Segment(size_t n) : key(std::numeric_limits<K>::max()), slope(), intercept(n) {};

    template<typename RandomIt>
    explicit Segment(
        const typename internal::OptimalPiecewiseLinearModel<K, size_t>::CanonicalSegment &cs,
        RandomIt first, RandomIt last
    )
        : key(cs.get_first_x()) {
        auto[cs_slope, cs_intercept] = cs.get_floating_point_segment(key);
        if (cs_intercept > std::numeric_limits<decltype(intercept)>::max())
            throw std::overflow_error("Change the type of Segment::intercept to int64");
        slope = cs_slope;
        intercept = cs_intercept;
        size_t distance = std::distance(first, last);
        data.reserve(distance);
        std::copy(first, last, std::back_inserter(data));
    }

    friend inline bool operator<(const Segment &s, const K &k) { return s.key < k; }
    friend inline bool operator<(const K &k, const Segment &s) { return k < s.key; }
    friend inline bool operator<(const Segment &s, const Segment &t) { return s.key < t.key; }

    operator K() { return key; };

    /**
     * Returns the approximate position of the specified key.
     * @param k the key whose position must be approximated
     * @return the approximate position of the specified key
     */
    inline size_t operator()(const K &k) const {
        auto pos = int64_t(slope * (k - key)) + intercept;
        return pos > 0 ? size_t(pos) : 0ull;
    }

    data_iterator ix_to_data_iterator(size_t ix) const {
        size_t offset = ix - intercept >= 0 && ix - intercept ? ix - intercept : 0;
        return data.cbegin() + offset;
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
    Iterator(const Iterator& other) : super(other.super), current(other.current) {}
};

#pragma pack(pop)

}