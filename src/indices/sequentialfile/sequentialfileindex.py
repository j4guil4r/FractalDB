from typing import Any, List
from .sequentialfile import SequentialFile, Record

class SequentialFileIndex:
    def __init__(self, rebuild_threshold: int = 64, index_block_size: int = 128):
        self.file = SequentialFile(rebuild_threshold, index_block_size)

    def add_record(self, key: Any, value: Any = None) -> None:
        self.file.add(Record(key, value))

    def search(self, key: Any) -> List[Record]:
        return self.file.search(key)

    def range_search(self, start_key: Any, end_key: Any) -> List[Record]:
        return self.file.range_search(start_key, end_key)

    def remove_record(self, key: Any) -> bool:
        return self.file.remove(key)
