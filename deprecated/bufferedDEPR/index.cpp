#include "buffered/pgm_index_buffered.h"

namespace pgm
{
  template <typename K, typename V>
  template <typename RandomIt>
  void BufferedPGMIndex<K, V>::build(RandomIt first, RandomIt last, size_t epsilon, size_t epsilon_recursive)
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
    base_level.reserve(n / epsilon * epsilon);

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
      EntryVector rec_entries;
      auto in_fun_rec = [&](auto i)
      {
        Entry e = last_level[i].data[0];
        rec_entries.push_back(e);
        return std::pair<K, size_t>(e.first, rec_entries.size());
      };
      auto out_fun_rec = [&](auto cs)
      {
        Segment new_segment = Segment(this, cs, rec_entries.begin(), rec_entries.end());
        next_level.push_back(new_segment);
        rec_entries.clear();
      };
      last_n = build_level(epsilon_recursive, in_fun_rec, out_fun_rec);
      levels.push_back(next_level);
      last_level = next_level;
    }
  }

  template <typename K, typename V>
  size_t BufferedPGMIndex<K, V>::by_level_segment_ix_for_key(const K &key, size_t goal_level)
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

  template <typename K, typename V>
  auto BufferedPGMIndex<K, V>::segment_for_key(const K &key)
  {
    return levels[0].begin() + by_level_segment_ix_for_key(key, 0);
  }

  template <typename K, typename V>
  auto BufferedPGMIndex<K, V>::mutable_segment_for_key(const K &key)
  {
    return levels[0].begin() + by_level_segment_ix_for_key(key, 0);
  }

  template <typename K, typename V>
  template <typename RandomIt>
  BufferedPGMIndex<K, V>::BufferedPGMIndex(size_t epsilon, size_t epsilon_recursive, RandomIt first, RandomIt last)
      : epsilon_value(epsilon), epsilon_recursive_value(epsilon_recursive), n(std::distance(first, last))
  {
    build(first, last, reduced_epsilon_value, reduced_epsilon_recursive_value);
  }

  template <typename K, typename V>
  typename BufferedPGMIndex<K, V>::BufferApproxPos BufferedPGMIndex<K, V>::search(const K &key)
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

  template <typename K, typename V>
  typename BufferedPGMIndex<K, V>::Iterator BufferedPGMIndex<K, V>::findIterator(const K &key)
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

  template <typename K, typename V>
  typename BufferedPGMIndex<K, V>::Entry BufferedPGMIndex<K, V>::findEntry(const K &key)
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

  template <typename K, typename V>
  V BufferedPGMIndex<K, V>::find(const K &key)
  {
    return findEntry(key).second;
  }

  template <typename K, typename V>
  void BufferedPGMIndex<K, V>::insert(K key, V value)
  {
    size_t seg_ix = by_level_segment_ix_for_key(key, 0);
    Entry e = std::make_pair(key, value);
    bool needs_split = levels[0][seg_ix].insert(e);
    if (needs_split)
    {
      split(0, seg_ix);
    }
  }

  template <typename K, typename V>
  void BufferedPGMIndex<K, V>::internal_insert(size_t split_level, size_t splitting_ix, std::vector<Segment> &new_segs)
  {
    // QUESTION: Should this be another function in the Segment struct, once we've given
    // this struct a firmer notion of internal?
    if (split_level == height() - 1)
    {
      if (new_segs.size() == 1)
      {
        // In this case, although the epsilon bound on the root was broken, it was
        // fixed during retraining
        // All we need to do is replace the root with the new segment
        std::cout << "Reset root" << std::endl;
        return;
      }
      else
      {
        /*
        levels[split_level].erase(levels[split_level].begin() + splitting_ix);
        levels[split_level].insert(levels[split_level].begin() + splitting_ix, new_segs.begin(), new_segs.end());
        // Now levels[0] contains all the segments of levels[1], but levels[0]
        */
        std::cout << "ROOT INTERNAL INSERT: " << new_segs.size() << std::endl;
        return;
      }
    }
    // The segment that is being split. NOTE: Still has original elements/position at call time
    Segment split_seg = levels[split_level][splitting_ix];
    // A representative element from the original segment (used to find indexes)
    Entry representative = split_seg.data[0];
    // The index of the parent of this segment in the level above
    size_t parent_ix = by_level_segment_ix_for_key(representative.first, split_level + 1);
    Segment *parent_seg = &levels[split_level + 1][parent_ix];
    // Get the index in the parent's data array
    auto og_iter_into_parent = std::lower_bound(parent_seg->data.begin(), parent_seg->data.end(), representative);

    // The first question we must answer is whether or not this internal segment
    // can absorb these new segments.
    // FIRST must have space
    bool parent_has_space = reduced_epsilon_recursive_value + parent_seg->num_inplaces + new_segs.size() - 1 < epsilon_recursive_value;
    // TODO: SECOND must avoid the edge case where the first val of new segments is less than previous
    bool can_absorb = parent_has_space;

    // Next we remove the old segment from the parent's data array
    parent_seg->data.erase(og_iter_into_parent);
    // TODO: auto new_iter_into_parent = std::lower_bound(parent_seg.data.begin(), parent_seg.data.end(), new_segs.front().to_entry());
    // Now add in the new entries into the parent's data array
    std::vector<Entry> new_entries;
    for (auto &s : new_segs)
    {
      new_entries.push_back(s.data[0]);
    }
    parent_seg->data.insert(og_iter_into_parent, new_entries.begin(), new_entries.end());

    // Now do the same thing but for the actual levels
    levels[split_level].erase(levels[split_level].begin() + splitting_ix);
    levels[split_level].insert(levels[split_level].begin() + splitting_ix, new_segs.begin(), new_segs.end());

    if (can_absorb)
    {
      parent_seg->num_inplaces += new_segs.size() - 1;
    }
    else
    {
      // We need to split this node as well.
      split(split_level + 1, parent_ix);
    }
  }

  template <typename K, typename V>
  void BufferedPGMIndex<K, V>::split(size_t split_level, size_t splitting_ix)
  {
    Segment splitting_seg = levels[split_level][splitting_ix];
    bool is_internal = split_level > 0;
    // How many total elements are going to be in this newly made thing
    size_t n = splitting_seg.data.size() + splitting_seg.buffer.size();
    std::vector<Segment> new_segs;
    size_t proper_ix = 0; // Index into proper data
    size_t buffer_ix = 0; // Index into buffer data
    std::vector<Entry> temporary_fill;
    auto in_fun = [&](auto i)
    {
      // Is the next insert coming from proper data or buffer
      bool is_proper;
      if (proper_ix >= splitting_seg.data.size())
      {
        is_proper = false;
      }
      else if (buffer_ix >= splitting_seg.buffer.size())
      {
        is_proper = true;
      }
      else
      {
        K next_proper = splitting_seg.data[proper_ix].first;
        K next_buffer = splitting_seg.buffer[buffer_ix].first;
        is_proper = next_proper < next_buffer;
      }
      // Get the entry, put it in the temp array, return the key with rank
      Entry e = is_proper ? splitting_seg.data[proper_ix++] : splitting_seg.buffer[buffer_ix++];
      temporary_fill.push_back(e);
      return std::pair<K, size_t>(e.first, temporary_fill.size());
    };
    auto out_fun = [&](auto cs)
    {
      Segment new_seg = Segment(this, cs, temporary_fill.begin(), temporary_fill.end());
      if (new_seg.data.size() > 0)
      {
        // TODO: I guess this means the construction algo is returning a blank seg
        // at the end. This may be intended behavior, but worth investigating
        new_segs.push_back(new_seg);
      }
      temporary_fill.clear();
    };
    auto build_segments = [&](auto in_fun, auto out_fun)
    {
      auto n_segments = internal::make_segmentation_par(n, is_internal ? reduced_epsilon_recursive_value : reduced_epsilon_value, in_fun, out_fun);
      return n_segments;
    };
    size_t n_segments = build_segments(in_fun, out_fun);
    internal_insert(split_level, splitting_ix, new_segs);
  }

  template <typename K, typename V>
  size_t BufferedPGMIndex<K, V>::segments_count() const
  {
    return levels.empty() ? 0 : levels[0].size();
  }

  template <typename K, typename V>
  size_t BufferedPGMIndex<K, V>::full_segments_count() const
  {
    size_t sum = 0;
    for (auto &l : levels)
    {
      sum += l.size();
    }
    return sum;
  }

  template <typename K, typename V>
  size_t BufferedPGMIndex<K, V>::height() const
  {
    return levels.size();
  }

  template <typename K, typename V>
  size_t BufferedPGMIndex<K, V>::size_in_bytes() const
  {
    // TODO: Fix
    return 0;
  }

  template <typename K, typename V>
  typename BufferedPGMIndex<K, V>::Iterator BufferedPGMIndex<K, V>::begin() const
  {
    auto first_seg = levels[0].begin();
    return Iterator(this, first_seg, first_seg->data.begin());
  }

  template <typename K, typename V>
  typename BufferedPGMIndex<K, V>::Iterator BufferedPGMIndex<K, V>::end() const
  {
    auto last_seg = levels[0].end();
    return Iterator(this, last_seg, last_seg->data.end());
  }

  template <typename K, typename V>
  void BufferedPGMIndex<K, V>::print_tree(int lowest_level)
  {
    std::cout << "Tree size: " << height() << " levels, " << full_segments_count() << " segments" << std::endl;
    int level = height() - 1;
    for (level; level >= lowest_level; --level)
    {
      std::cout << "Level: " << level << std::endl;
      for (size_t cur_ix = 0; cur_ix < levels[level].size(); ++cur_ix)
      {
        std::cout << "(ix:" << cur_ix << ", " << levels[level][cur_ix].get_first_proper_key() << ", " << levels[level][cur_ix].get_last_proper_key() << ", sz:" << levels[level][cur_ix].data.size() << ")"
                  << " - ";
      }
      std::cout << std::endl;
    }
  }
}
