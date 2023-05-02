#include "buffered/pgm_index_buffered.h"

namespace pgm
{
  template <typename K, typename V>
  using Index = BufferedPGMIndex<K, V>;

  template <typename K, typename V>
  Index<K, V>::Iterator::Cursor::Cursor(const segment_iterator seg_iter)
      : seg_iter(seg_iter) {}

  template <typename K, typename V>
  Index<K, V>::Iterator::Cursor::Cursor(const segment_iterator seg_iter, const data_iterator data_iter)
      : seg_iter(seg_iter), data_iter(data_iter) {}

  template <typename K, typename V>
  void Index<K, V>::Iterator::advance()
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

  template <typename K, typename V>
  void Index<K, V>::Iterator::go_back()
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

}