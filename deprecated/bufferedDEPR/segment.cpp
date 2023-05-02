#include "buffered/pgm_index_buffered.h"

namespace pgm
{
  template <typename K, typename V>
  using Index = BufferedPGMIndex<K, V>;

  template <typename K, typename V>
  Index<K, V>::Segment::Segment(
      const buffered_pgm_type *super,
      K first_x,
      float slope,
      int32_t intercept) : super(super),
                           first_x(first_x),
                           slope(slope),
                           intercept(intercept){};

  template <typename K, typename V>
  Index<K, V>::Segment::Segment(const Index<K, V> *super, size_t n) : super(super), slope(0), intercept(0)
  {
    data.push_back(std::make_pair(std::numeric_limits<K>::max(), std::numeric_limits<V>::max()));
  };

  template <typename K, typename V>
  template <typename RandomIt>
  Index<K, V>::Segment::Segment(
      const buffered_pgm_type *super,
      const typename internal::OptimalPiecewiseLinearModel<K, size_t>::CanonicalSegment &cs,
      RandomIt first, RandomIt last)
      : super(super)
  {
    first_x = cs.get_first_x();
    auto [cs_slope, cs_intercept] = cs.get_floating_point_segment(first_x);
    if (cs_intercept > std::numeric_limits<decltype(intercept)>::max())
      throw std::overflow_error("Change the type of Segment::intercept to int64");
    slope = cs_slope;
    intercept = cs_intercept;
    size_t distance = std::distance(first, last);
    data.reserve(distance);
    std::copy(first, last, std::back_inserter(data));
    buffer.reserve(max_buffer_size);
  }

  template <typename K, typename V>
  bool Index<K, V>::Segment::insert(const Entry &e)
  {
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
    bool is_buffered_insert = false;
    if (super->reduced_epsilon_value + num_inplaces >= super->epsilon_value)
    {
      // If there isn't space for an in-place insert, make it buffered
      is_buffered_insert = true;
    }
    // OR if it would get inserted at the beginning make it buffered
    if (existing == data.begin())
    {
      // If the data would end up at the beginning of a segment, make it buffered
      is_buffered_insert = true;
    }

    size_t predicted_pos = (*this)(e.first);
    size_t actual_pos = std::distance(data.begin(), existing);
    size_t difference = predicted_pos > actual_pos ? predicted_pos - actual_pos : actual_pos - predicted_pos;
    if (difference > super->epsilon_value)
    {
      // If the predicted position is too far away from the actual position, make it buffered
      // NOTE: THIS is the true "out-of-place" insert
      is_buffered_insert = true;
    }

    if (is_buffered_insert)
    {
      // Insert into sorted position in buffer
      auto location = std::lower_bound(buffer.begin(), buffer.end(), e);
      buffer.insert(location, e);
      return buffer_has_space();
    }
    else
    {
      // Insert into sorted position in data
      data.insert(existing, e);
      num_inplaces++;

      return false;
    }

    return buffer_has_space();
  }

}