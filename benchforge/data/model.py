from enum import Enum
from dataclasses import dataclass
from typing import List


class DataGenerationType(Enum):
    """
    Enum representing types of data generation for algorithm benchmarking.

    Attributes:
        UNKNOWN: Undefined data type.
        NOT_CHANGED: Data remains unchanged from its initial state.
        SORTED: Data is sorted in ascending order.
        REVERSED: Data is sorted in descending order.
        RANDOM: Data is randomized.
    """
    UNKNOWN = "unknown"
    NOT_CHANGED = "not changed"
    SORTED = "sorted"
    REVERSED = "reversed"
    RANDOM = "randomized"


@dataclass
class Data:
    """
    Represents a single dataset used for algorithm benchmarking.

    Attributes:
        _id (str): Unique identifier for the dataset.
        _values (List[int]): List of integers in the dataset.
        _type (DataGenerationType): The type of data generation (e.g., sorted,
        random).

    Methods:
        __len__(): Returns the number of elements in the dataset.
        data_id (str): Getter for the unique identifier of the dataset.
        generation_type (DataGenerationType): Getter for the data generation type.
        values (List[int]): Getter for the list of dataset values.
        __repr__(): Returns a string representation of the dataset.
    """
    def __init__(self, values: List[int] = [],
                    gen_type: DataGenerationType = DataGenerationType.NOT_CHANGED,
                        _id: str = "") \
            -> None:
        self._id = _id
        self._values: List[int] = values
        self._type: DataGenerationType = gen_type

    def __len__(self) -> int:
        return len(self._values)

    @property
    def data_id(self) -> str:
        return self._id

    @property
    def generation_type(self) -> DataGenerationType:
        return self._type

    @property
    def values(self) -> List[List[int]]:
        return self._values

    def __repr__(self) -> str:
        return f"Data(id={self._id}, values={self._values}, type={self._type})"


class Sample:
    """
    Represents a collection of datasets for algorithm benchmarking.

    Attributes:
        _datas (List[Data]): List of Data objects in the sample.
        _type (DataGenerationType): The type of data generation for the sample.

    Methods:
        add(data: Data): Adds a Data object to the sample.
        remove(data: Data): Removes a Data object from the sample.
        get(_id: str) -> Data | None: Retrieves a Data object by its identifier.
        generation_type (DataGenerationType): Getter and setter for the data
        generation type.
        __iter__(): Returns an iterator over the datasets.
        __len__(): Returns the number of datasets in the sample.
        __contains__(data: Data): Checks if a Data object exists in the sample.
        __getitem__(index: int) -> Data: Retrieves a dataset by index.
        __setitem__(index: int, value: Data): Updates a dataset at a specific index.
        __delitem__(index: int): Deletes a dataset at a specific index.
        clear(): Clears all datasets in the sample.
        __repr__(): Returns a string representation of the sample.
        __str__(): Returns a user-friendly string for the sample.
        __eq__(other: Sample): Checks if two samples are equal.
        __bool__(): Returns True if the sample contains datasets.
    """
    def __init__(self) -> None:
        self._datas: list[Data] = []
        self._type: DataGenerationType = DataGenerationType.UNKNOWN

    def add(self, data: Data):
        self._datas.append(data)

    def remove(self, data: Data):
        self._datas.remove(data)

    def get(self, _id: str) -> Data | None:
        for data in self._datas:
            if data.data_id == _id:
                return data
        return None

    @property
    def generation_type(self):
        return self._type

    @generation_type.setter
    def generation_type(self, value):
        self._type = value

    def __iter__(self):
        return iter(self._datas)

    def __len__(self):
        return len(self._datas)

    def __contains__(self, data: Data):
        return data in self._datas

    def __getitem__(self, index: int) -> Data:
        return self._datas[index]

    def __setitem__(self, index: int, value: Data):
        self._datas[index] = value

    def __delitem__(self, index: int):
        del self._datas[index]

    def __repr__(self):
        return f"Sample({self._datas})"

    def __str__(self):
        return f"Sample with {len(self._datas)} data items."

    def __eq__(self, other):
        if isinstance(other, Sample):
            return self._datas == other._datas
        return False

    def __bool__(self):
        return bool(self._datas)

    def clear(self):
        self._datas.clear()
