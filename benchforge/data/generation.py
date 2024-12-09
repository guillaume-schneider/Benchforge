from typing import List
from benchforge.data.model import DataGenerationType, Sample, Data
import random
import copy


class DataGenerator:
    """
    Generates datasets of varying sizes and configurations for algorithm
    benchmarking.

    Attributes:
        _base_values (List[List[int]]): Stores the base values for dataset
        generation.

    Methods:
        generate(step_size: int, sample_count: int): Generates datasets with
        incremental sizes.
        get(generation_type: DataGenerationType) -> Sample: Creates a Sample object
        with specified data generation types.
        clear(): Clears all stored datasets.
    """
    def __init__(self) -> None:
        self._base_values: List[List[int]] = []

    def generate(self, step_size: int, sample_count=1) -> None:
        self._base_values = []
        max_range = step_size * sample_count
        for i in range(1, sample_count + 1):
            current_size = step_size * i
            self._base_values.append(
                [random.randint(0, max_range)
                 for _ in range(current_size)]
            )

    def get(self, generation_type: DataGenerationType) -> Sample:
        sample = Sample()
        for i in range(len(self._base_values)):
            base_values = self._base_values[i]
            match generation_type:
                case DataGenerationType.NOT_CHANGED:
                    values = copy.deepcopy(base_values)
                    _type = DataGenerationType.NOT_CHANGED
                case DataGenerationType.SORTED:
                    values = sorted(base_values)
                    _type = DataGenerationType.SORTED
                case DataGenerationType.REVERSED:
                    values = sorted(base_values, reverse=True)
                    _type = DataGenerationType.REVERSED
                case DataGenerationType.RANDOM:
                    values = copy.deepcopy(base_values)
                    random.shuffle(values)
                    _type = DataGenerationType.RANDOM

            _id = str(hash(tuple(base_values)))
            sample.add(Data(values, _type, _id))
        return sample

    def clear(self):
        self._base_values.clear()
