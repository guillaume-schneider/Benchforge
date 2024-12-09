from typing import List
from benchforge.data.model import Data, Sample, DataGenerationType


class DataProcessResult:
    """
    Represents the result of processing a dataset with an algorithm.

    Attributes:
        _time (float): Time taken to process the dataset.
        _data (Data): The dataset processed.
        _process (Process): The process applied to the dataset.

    Methods:
        time (float): Getter for the processing time.
        data_size (int): Returns the size of the processed dataset.
        process_id (str): Returns the process identifier.
        data_id (str): Returns the dataset identifier.
        generation_type (DataGenerationType): Returns the data generation type.
        __repr__(): Returns a string representation of the result.
    """
    def __init__(self, _time: float, data: Data = None,
                 process_id: str = "") -> None:
        self._time = _time
        self._data = data
        self._process_id = process_id

    @property
    def time(self) -> float:
        return self._time

    @property
    def data_size(self) -> int:
        return len(self._data)

    @property
    def process_id(self) -> str:
        return self._process_id

    @property
    def data_id(self) -> str:
        if self._data is None:
            return ""
        return self._data.data_id

    @property
    def generation_type(self) -> DataGenerationType:
        if self._data is None:
            return DataGenerationType.UNKNOWN
        return self._data.generation_type

    def __repr__(self) -> str:
        return f"DataProcessResult(id={self._process_id}, " \
            + f"time={self.time}, data_id={self.data_id})"


class SampleProcessResult:
    def __init__(self, results: List[DataProcessResult] = None) -> None:
        self._results = [] if results is None else results
        self._process_id: int = results[0].process_id \
            if len(self._results) > 0 else ""
        self._type: DataGenerationType = self._results[0].generation_type \
            if len(self._results) > 0 else DataGenerationType.UNKNOWN

    def add(self, result: DataProcessResult):
        if self._process_id == "":
            self._process_id = result.process_id
            self._type = result.generation_type
        self._results.append(result)

    def remove(self, result: DataProcessResult):
        self._results.remove(result)

    @property
    def results(self):
        return self._results

    @property
    def generation_type(self):
        return self._type

    @property
    def process_id(self):
        return self._process_id

    @property
    def times(self) -> List[float]:
        return [result.time for result in self._results]

    @property
    def data_sizes(self) -> List[int]:
        return [result.data_size for result in self._results]

    def __iter__(self):
        return iter(self._results)

    def __len__(self):
        return len(self._results)

    def __contains__(self, data: Data):
        return data in self._results

    def __getitem__(self, index: int) -> Data:
        return self._results[index]

    def __setitem__(self, index: int, value: Data):
        self._results[index] = value

    def __delitem__(self, index: int):
        del self._results[index]

    def __repr__(self):
        return f"Sample({self._results})"

    def __str__(self):
        return f"Sample with {len(self._results)} data items."

    def __eq__(self, other):
        if isinstance(other, Sample):
            return self._results == other._results
        return False

    def __bool__(self):
        return bool(self._results)

    def clear(self):
        self._results.clear()
