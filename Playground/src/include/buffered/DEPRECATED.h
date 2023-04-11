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
#include <iostream>

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
   * Inserts are handled by keeping a buffer on top of every segment of keys that "want" to live in this segment
   * but can't because their inclusion would destroy the Epsilon bounds of the segment. More logic concerning
   * the process of merging and deleting is coming soon.
   *
   * @tparam K the type of the indexed keys
   * @tparam V the type of the indexed values
   */
  template <typename K, typename V>
  class BufferedPGMIndex
  {
  public:
    struct Segment;
    struct Iterator;
    using Level = typename std::vector<Segment>;
    // Convenient shorthand for type of data and buffer
    using Entry = typename std::pair<K, V>;
    using EntryVector = typename std::vector<Entry>;
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
    size_t n; ///< The number of elemens in the structure
    std::vector<Level> levels;

    /****** TUNABLE PARAMS ******/
    float fill_ratio = 0.5;
    float fill_ratio_recursive = 0.5;
    size_t epsilon_value = 64;
    size_t reduced_epsilon_value = (size_t)((float)epsilon_value * fill_ratio);
    size_t epsilon_recursive_value = 8;
    size_t reduced_epsilon_recursive_value = (size_t)((float)epsilon_recursive_value * fill_ratio);
    /****** END TUNABLE PARAMS ******/

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
    void build(RandomIt first, RandomIt last, size_t epsilon, size_t epsilon_recursive);

    /**
     * Helper function that returns the index of the segment in a given level that should be used
     * to index a key.
     */
    size_t by_level_segment_ix_for_key(const K &key, size_t goal_level);

    /**
     * Returns the segment responsible for a given key, that is, the rightmost segment having key <= the sought key.
     * @param key the value of the element to search for
     * @return an iterator to the segment responsible for the given key
     */
    auto segment_for_key(const K &key);

    /**
     * Returns the segment responsible for a given key, that is, the rightmost segment having key <= the sought key.
     * NOTE: Needed because on inserts we need a mutable non-const type
     * @param key the value of the element to search for
     * @return an iterator to the segment responsible for the given key
     */
    auto mutable_segment_for_key(const K &key);

    /**
     * Constructs the index on the sorted keys in the range [first, last).
     * NOTE: Constructs the index obeying reduced epsilon, not the original epsilon.
     * This allows us to be confident that each segment will be able to absorb at least a certain
     * number of inserts.
     * @param first, last the range containing the sorted keys to be indexed
     */
    template <typename RandomIt>
    BufferedPGMIndex(size_t epsilon, size_t recursive_epsilon, RandomIt first, RandomIt last);

    size_t get_internal_offset_into_level(size_t level, size_t seg_ix);

    /**
     * Returns the approximate position and the range where @p key can be found.
     * @param key the value of the element to search for
     * @return a struct with the approximate position and bounds of the range
     */
    BufferApproxPos search(const K &key);

    /**
     * Finds the entry of an element with key equivalent to @p key. Returns an iterator pointing to that
     * entry.
     * NOTE: Does not search in the insert/delete buffers!
     * @param key the value of the element to search for
     * @return an iterator to the entry with key equivalent to @p key, or end() if no such element is found
     */
    Iterator findIterator(const K &key);

    /**
     * Finds the entry of an element with key equivalent to @p key.
     * @param key the value of the element to search for
     * @return the entry of the element with key equivalent to @p key, or the maximum value of @p K V if no such element is found
     */
    Entry findEntry(const K &key);

    V find(const K &key);

    void insert(K key, V value);

    /**
     * A helper function for handling inserts at non-base levels (level > 0).
     * Note that during split, we can produce an arbitrary number of new segments. Internal
     * segments have no insert buffers, but are trained at a lower epsilon bound, meaning they
     * can absorb some but not all inserts.
     * @param splitting_ix - The index of the segment that is splitting
     * @param new_segs - The new segments generated by retraining this segment on data + buffer
     */
    void internal_insert(size_t split_level, size_t splitting_ix, std::vector<Segment> &new_segs);

    /**
     * A helper function to split a segment (meaning retrain it on the data + buffer + neighbors)
     * and recursively trigger necessary inserts up the tree
     * @param splitting_ix - The index of the segment that is being retrained / split
     */
    void split(size_t split_level, size_t splitting_ix);

    /**
     * Returns the number of segments in the last level of the index.
     * @return the number of segments
     */
    size_t segments_count() const;

    /**
     * Returns the number of segments in the full index, including internal
     */
    size_t full_segments_count() const;

    /**
     * Returns the number of levels of the index.
     * @return the number of levels of the index
     */
    size_t height() const;

    /**
     * Returns the size of the index in bytes.
     * @return the size of the index in bytes
     */
    size_t size_in_bytes() const;

    Iterator begin() const;

    Iterator end() const;

    void print_tree(int lowest_level = 0);
  };

#pragma pack(push, 1)

  template <typename K, typename V>
  struct BufferedPGMIndex<K, V>::Segment
  {
    friend class BufferedPGMIndex;
    using buffered_pgm_type = BufferedPGMIndex<K, V>;

    const buffered_pgm_type *super; ///< The index this segment belongs to
    K first_x = 0;                  ///< The first x value in this segment
    float slope;                    ///< The slope of the segment.
    int32_t intercept;              ///< The intercept of the segment.
    size_t num_inplaces = 0;        ///< The number of in-place inserts that have been performed on this segment
    size_t max_buffer_size = 16;    ///< The size of the buffer for out-of-place inserts
    float split_threshold = 0.75;   ///< How full does buffer need to be before trigger a split/retrain
    size_t split_neighbors = 0;     ///< How many neighbors to split with
    EntryVector data;               ///< The data stored in this segment in sorted order by key
    EntryVector buffer;             ///< A buffer of inserts waiting to come to this segment

    Segment(const buffered_pgm_type *super, K first_x, float slope, int32_t intercept);

    Segment(const buffered_pgm_type *super, size_t n);

    template <typename RandomIt>
    Segment(
        const buffered_pgm_type *super,
        const typename internal::OptimalPiecewiseLinearModel<K, size_t>::CanonicalSegment &cs,
        RandomIt first, RandomIt last);

    friend inline bool operator<(const Segment &s, const K &k) { return s.data[0].first < k; }
    friend inline bool operator<(const K &k, const Segment &s) { return k < s.data[0].first; }
    friend inline bool operator<(const Segment &s, const Segment &t) { return s.data[0].first < t.data[0].first; }

    /**
     * Returns the approximate position of the specified key.
     * @param k the key whose position must be approximated
     * @return the approximate position of the specified key
     */
    inline size_t predict_pos(const K &k) const
    {
      auto pos = int64_t(slope * (k - first_x)) + intercept;
      pos = pos > 0 ? size_t(pos) : 0ull;
      pos = pos < data.size() ? pos : data.size() - 1;
      return pos;
    }

    /**
     * Checks if the buffer can absorb another out of place write
     * @return true if the buffer can absorb another out of place write
     */
    inline bool buffer_has_space() const
    {
      return buffer.size() >= (size_t)((float)max_buffer_size) * split_threshold;
    }

    inline data_iterator ix_to_data_iterator(size_t offset) const
    {
      if (offset < 0)
      {
        return data.cbegin();
      }
      if (offset >= data.size())
      {
        return std::prev(data.cend());
      }
      return data.cbegin() + offset;
    }

    /**
     * A helper function to return the first key in the segment
     * NOTE: Does not include buffer!
     */
    inline K get_first_proper_key() const
    {
      return data.front().first;
    }

    /**
     * A helper function to return the last key in the segment
     * NOTE: Does not include buffer!
     */
    inline K get_last_proper_key() const
    {
      return data.back().first;
    }

    /**
     * Helpful for various things
     */
    inline Entry to_entry()
    {
      return data[0];
    }

    /**
     * A function to perform an insert into a segment. Returns a boolean indicating whether the
     * buffer is full _after_ the insert (in which case the parent should trigger
     * a retrain).
     * NOTE: If the key already exists it updates the value
     * @return true (parent needs to retrain) or false
     */
    bool insert(const Entry &e);
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
      Cursor(const segment_iterator seg_iter);
      Cursor(const segment_iterator seg_iter, const data_iterator data_iter);
    };

    const buffered_pgm_type *super; //< Pointer to the buffered_pgm that is being iterated
    Cursor current;                 //< The current cursor

    void advance();

    void go_back();

    inline void advance_by(size_t n)
    {
      while (n--)
        advance();
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

}