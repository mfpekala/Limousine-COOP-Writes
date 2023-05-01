#include <cstddef>
#include <cstdint>
#include <vector>
#include <string>

struct TreeShape
{
  std::vector<size_t> level_sizes = {};

  std::string encode()
  {
    std::string result;
    for (size_t i = 0; i < level_sizes.size(); ++i)
    {
      result += std::to_string(level_sizes[i]);
      if (i < level_sizes.size() - 1)
        result += "#";
    }
    return result;
  }
};

struct ReadProfile
{
  size_t num_data = 0;
  size_t num_buffer = 0;
  size_t num_ne = 0;

  std::string encode()
  {
    return std::to_string(num_data) + "#" + std::to_string(num_buffer) + "#" + std::to_string(num_ne);
  }
};

struct SplitHistory
{
  std::vector<size_t> splits_by_level = {};
  std::vector<size_t> data_movement_by_level = {};

  std::string encode()
  {
    std::string result;
    for (size_t i = 0; i < splits_by_level.size(); ++i)
    {
      result += std::to_string(splits_by_level[i]);
      if (i < splits_by_level.size() - 1)
        result += "#";
    }
    result += "@";
    for (size_t i = 0; i < data_movement_by_level.size(); ++i)
    {
      result += std::to_string(data_movement_by_level[i]);
      if (i < data_movement_by_level.size() - 1)
        result += "#";
    }
    return result;
  }
};
