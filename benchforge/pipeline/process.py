from typing import Callable, List
from dataclasses import dataclass
from benchforge.pipeline.result import SampleProcessResult, DataProcessResult
from benchforge.data.model import Sample, Data
import time


@dataclass
class Process:
    """
    Encapsulates a processing algorithm or function.

    Attributes:
        _id (str): Unique identifier for the process.
        _function (Callable): The function to apply on a dataset.

    Methods:
        run(values: List[int]): Executes the process function on the provided
        dataset.
        identifier (str): Getter for the process identifier.
        function (Callable): Getter for the process function.
        __repr__(): Returns a string representation of the process.
    """
    def __init__(self, _id: str, function: Callable[[List[int]], None]) -> None:
        self._id = _id
        self._function = function

    def run(self, values: list[int]):
        self._function(values)

    @property
    def identifier(self) -> str:
        return self._id

    @property
    def function(self) -> Callable:
        return self._function

    def __repr__(self) -> str:
        return f"Process(id={self._id}, function={self._function})"


class DataProcessor:
    def __init__(self) -> None:
        self._process: dict[str, Process] = {}

    def add(self, process: Process) -> None:
        self._process.update({process.identifier: process})

    def get(self, _id: str) -> Process:
        return self._process.get(_id)

    def remove(self, _id: str) -> None:
        self._process.pop(_id)

    def run(self, process_id: str, sample: Sample) -> SampleProcessResult:
        def _run_single(_process_id: str, data: Data):
            process = self.get(_process_id)
            start_time = time.perf_counter()
            process.run(data.values)
            process_time = time.perf_counter() - start_time
            return DataProcessResult(process_time, data, _process_id)

        result = SampleProcessResult()
        for data in sample:
            result.add(_run_single(process_id, data))
        return result

    @property
    def process(self) -> dict[str, Process]:
        return self._process

    def clear(self):
        self._process.clear()
