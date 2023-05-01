
class Metric:
    @staticmethod
    def decode(s: str) -> "Metric":
        raise NotImplementedError

class TreeShape(Metric):
    def __init__(self, level_sizes: list[int]):
        self.level_sizes = level_sizes

    @staticmethod
    def decode(s: str) -> "TreeShape":
        raw = s.split("#")
        return TreeShape([int(x) for x in raw])

class ReadProfile(Metric):
    def __init__(self, num_data: int, num_buffer: int, num_error: int):
        self.num_data = num_data
        self.num_buffer = num_buffer
        self.num_error = num_error

    @staticmethod
    def decode(s: str) -> "ReadProfile":
        raw = s.split("#")
        return ReadProfile(int(raw[0]), int(raw[1]), int(raw[0]))

class SplitHistory(Metric):
    def __init__(self, splits_by_level, data_movement_by_level):
        self.splits_by_level = splits_by_level
        self.data_movement_by_level = data_movement_by_level
    
    @staticmethod
    def decode(s: str) -> "SplitHistory":
        splits, mvmt = s.split("@")
        raw1 = splits.split("#")
        raw2 = mvmt.split("#")
        return SplitHistory(
            [int(0 if x == '' else x) for x in raw1],
            [int(0 if x == '' else x) for x in raw2]
        )