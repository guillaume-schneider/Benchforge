import random
from enum import Enum
import os
import psutil
import time
import copy
import matplotlib.pyplot as plt
from typing import Callable, List
from dataclasses import dataclass
import sys


sys.setrecursionlimit(10000)


@DeprecationWarning
def isSorted(_list):
    for i in range(len(_list) - 1):
        if _list[i] > _list[i + 1]:
            return False
    return True


@DeprecationWarning
def areSorted(lists):
    for i in range(len(lists)):
        if not isSorted(lists[i]):
            return (False, i)
    return (True, 0)


@DeprecationWarning
def create_data(nlist=15, nval=200):
    def fill_list(nlist, nval):
        for i in range(1, nlist + 1):
            s = nval * i
            dataRandom = s * [0]
            dataSorted = s * [0]
            dataInversed = s * [0]
            for j in range(s):
                dataRandom[j] = j
                dataSorted[j] = j
                dataInversed[j] = j

            dataInversed.reverse()
            random.shuffle(dataRandom)

            listDataRandom.append(dataRandom)
            listDataSorted.append(dataSorted)
            listDataInversedSorted.append(dataInversed)
            sizeArrays.append(s)

    # Création de listes de taille incrémentale et de contenu aléatoire
    listDataRandom = []
    listDataSorted = []
    listDataInversedSorted = []
    sizeArrays = []
    fill_list(nlist, nval)

    return (sizeArrays, listDataRandom, listDataSorted, listDataInversedSorted)


def set_cpu_affinity(cpu_id=0):
    """Force the program to run on a specific CPU core."""
    p = psutil.Process(os.getpid())
    p.cpu_affinity([cpu_id])


@DeprecationWarning
def executerTri(fct_tri, color, nom, nlist=15, nval=200, surplace=True):
    set_cpu_affinity(cpu_id=0)  # Lock the program to CPU core 0

    axis, listDataRandom, listDataSorted, listDataInvertedSorted = create_data(nlist, nval)

    toplotRandom = []
    toplotSorted = []
    toplotInverted = []

    dataTestRandom = copy.deepcopy(listDataRandom)
    dataTestSorted = copy.deepcopy(listDataSorted)
    dataTestInverted = copy.deepcopy(listDataInvertedSorted)

    for i in range(len(axis)):
        time1 = time.time()
        if surplace:
            fct_tri(dataTestRandom[i])
        else:
            dataTestRandom[i] = fct_tri(dataTestRandom[i])
        time2 = time.time()
        toplotRandom.append(time2 - time1)

        time3 = time.time()
        if surplace:
            fct_tri(dataTestSorted[i])
        else:
            dataTestSorted[i] = fct_tri(dataTestSorted[i])
        time4 = time.time()
        toplotSorted.append(time4 - time3)

        time5 = time.time()
        if surplace:
            fct_tri(dataTestInverted[i])
        else:
            dataTestInverted[i] = fct_tri(dataTestInverted[i])
        time6 = time.time()
        toplotInverted.append(time6 - time5)

    (ok1, ipb1) = areSorted(dataTestRandom)
    (ok2, ipb2) = areSorted(dataTestSorted)
    (ok3, ipb3) = areSorted(dataTestInverted)

    if not ok1:
        print(nom + ' data random incorrect, liste #' + str(ipb1))
    else:
        plt.plot(axis, toplotRandom, '-' + color, label=nom + ' (random)')
    if not ok2:
        print(nom + ' data Sorted incorrect, liste #' + str(ipb2))
    else:
        plt.plot(axis, toplotSorted, '--' + color, label=nom + ' (Sorted)')
    if not ok3:
        print(nom + ' data Inverted incorrect, liste #' + str(ipb3))
    else:
        plt.plot(axis, toplotInverted, ':' + color, label=nom + ' (Inverted)')
        plt.legend()


class Singleton(type):
    """
    A metaclass for creating Singleton classes. Ensures only one instance
    of the class is created and reused across the application.
    """
    _instances = {}

    def __call__(cls, *args, **kwargs):
        if cls not in cls._instances:
            # If an instance does not already exist, create one
            instance = super().__call__(*args, **kwargs)
            cls._instances[cls] = instance
        return cls._instances[cls]


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
        _type (DataGenerationType): The type of data generation (e.g., sorted, random).

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
        generation_type (DataGenerationType): Getter and setter for the data generation type.
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


class DataGenerator:
    """
    Generates datasets of varying sizes and configurations for algorithm benchmarking.

    Attributes:
        _base_values (List[List[int]]): Stores the base values for dataset generation.

    Methods:
        generate(step_size: int, sample_count: int): Generates datasets with incremental sizes.
        get(generation_type: DataGenerationType) -> Sample: Creates a Sample object with specified data generation types.
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


@dataclass
class Process:
    """
    Encapsulates a processing algorithm or function.

    Attributes:
        _id (str): Unique identifier for the process.
        _function (Callable): The function to apply on a dataset.

    Methods:
        run(values: List[int]): Executes the process function on the provided dataset.
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


@dataclass
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
                 process: Process = None) -> None:
        self._time = _time
        self._data = data
        self._process = process

    @property
    def time(self) -> float:
        return self._time

    @property
    def data_size(self) -> int:
        return len(self._data)

    @property
    def process_id(self) -> str:
        if self._process is None:
            return ""
        return self._process.identifier

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
        return f"DataProcessResult(id={self._process.identifier}, " \
            + f"time={self.time}, data_id={self.data_id})"


class SampleProcessResult:
    def __init__(self, results: List[DataProcessResult] = None) -> None:
        self._results = [] if results is None else results
        self._process_id: int = results[0].process_id if len(self._results) > 0 else ""
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
        def _run_single(proc_id: str, data: Data):
            process = self.get(proc_id)
            start_time = time.perf_counter()
            process.run(data.values)
            process_time = time.perf_counter() - start_time
            return DataProcessResult(process_time, data, process)

        result = SampleProcessResult()
        for data in sample:
            result.add(_run_single(process_id, data))
        return result

    @property
    def process(self) -> dict[str, Process]:
        return self._process

    def clear(self):
        self._process.clear()


@dataclass
class ChartProperties:
    label: str
    style: str
    color: str


class ProcessChart:
    def __init__(self, process_id: str) -> None:
        self._process_id = process_id
        self._match_gen_id: dict[DataGenerationType, ChartProperties] = {}
        self._results: list[SampleProcessResult] = []

    def _get_by_gen_id(self, gen_id: DataGenerationType):
        if gen_id not in self._match_gen_id:
            self._match_gen_id.update({gen_id: ChartProperties("Default label",
                                                               "-", "r")})
        return self._match_gen_id[gen_id]

    def add_result(self, result: SampleProcessResult) -> None:
        label = self._get_by_gen_id(result.generation_type).label
        if label == "Default label" or label == "":
            self[result.generation_type].label = result.generation_type.value
        self._results.append(result)

    @property
    def results(self) -> list[SampleProcessResult]:
        return self._results

    @property
    def process_id(self) -> str:
        return self._process.identifier

    @property
    def generation_types(self) -> DataGenerationType:
        return self._match_gen_id.keys()

    def __getitem__(self, gen_type: DataGenerationType) -> ChartProperties:
        return self._get_by_gen_id(gen_type)

    def __setitem__(self, gen_type: DataGenerationType, value: ChartProperties):
        if not isinstance(value, ChartProperties):
            raise ValueError("value passed must be ChartProperties.")
        self._match_gen_id[gen_type] = value

    def __delitem__(self, gen_type: DataGenerationType):
        del self._match_gen_id[gen_type]

    def __repr__(self):
        gen_id_repr = {key: vars(value) for key, value in self._match_gen_id.items()}
        return (
            f"ProcessChart("
            f"process_id={self._process_id}, "
            f"generation_types={list(self._match_gen_id.keys())}, "
            f"results_count={len(self._results)}, "
            f"match_gen_id={gen_id_repr})"
        )

    def clear(self):
        self._results.clear()


class ProcessChartDisplayer:
    def __init__(self) -> None:
        self._process_charts: dict[str, ProcessChart] = {}

    def _get_process_chart(self, process_id: str) -> ProcessChart:
        if process_id not in self._process_charts:
            self._process_charts.update({process_id: ProcessChart(process_id)})
        return self._process_charts[process_id]

    def add_result(self, result: SampleProcessResult) -> None:
        if result.process_id not in self._process_charts:
            self._process_charts.update({result.process_id: ProcessChart(result.process_id)})
        self._process_charts[result.process_id].add_result(result)

    def get_result_by_process(self, process_id: str) -> list[SampleProcessResult]:
        if process_id not in self._process_charts:
            return []
        return self._process_charts[process_id].results

    def display_by_process(self, process_id: str) -> None:
        results = self.get_result_by_process(process_id)
        fig, ax = plt.subplots()

        ax.set_xlabel("Sample size (unit)")
        ax.set_ylabel("Process time (second)")
        ax.set_title(process_id)

        for result in results:
            chart_properties = self[process_id][result.generation_type]
            color = chart_properties.color
            style = chart_properties.style
            label = chart_properties.label
            ax.plot(result.data_sizes, result.times, style + color, label=label)

        ax.legend()
        plt.show()

    def clear_by_process(self, process_id: str) -> None:
        if process_id in self._process_charts:
            self._process_charts[process_id].clear()

    def __getitem__(self, process_id: str) -> ProcessChart:
        return self._get_process_chart(process_id)

    def __setitem__(self, process_id: str, value: ProcessChart):
        if process_id not in self._process_charts:
            self._process_charts.update({process_id: ProcessChart(process_id)})
        self._process_charts[process_id] = value

    def __delitem__(self, process_id: str):
        del self._process_charts[process_id]

    def clear(self):
        self._process_charts.clear()


def partition(A, start, end):
    """
    Partition the array into elements less than or equal to the pivot
    and those greater than the pivot.

    Parameters:
        A: list[int] - The array to partition.
        start: int - Start index of the portion to partition.
        end: int - End index of the portion to partition.

    Returns:
        int - The index of the pivot after partitioning.
    """
    val = A[end]
    iPivot = start

    for i in range(start, end):
        if A[i] <= val:
            A[i], A[iPivot] = A[iPivot], A[i]
            iPivot += 1

    A[iPivot], A[end] = A[end], A[iPivot]
    return iPivot


def _quick_sort(A, start, end):
    """
    Perform QuickSort on the array.

    Parameters:
        A: list[int] - The array to sort.
        start: int - Start index of the portion to sort.
        end: int - End index of the portion to sort.

    Returns:
        None (sorts the array in place).
    """
    if start >= end:
        return
    iPivot = partition(A, start, end)
    _quick_sort(A, start, iPivot - 1)
    _quick_sort(A, iPivot + 1, end)


def quick_sort(A):
    return _quick_sort(A, 0, len(A) - 1)


def bubble_sort(A):
    n = len(A)
    while n > 1:
        newn = 0
        for i in range(1, n):
            if A[i - 1] > A[i]:
                A[i - 1], A[i] = A[i], A[i - 1]
                newn = i
        n = newn
    return A


def swap(A, index_1, index_2):
    buffer = A[index_1]
    A[index_1] = A[index_2]
    A[index_2] = buffer


def selection_sort(A):
    n = len(A)
    for i in range(n):
        j_min = i
        for j in range(i + 1, n):
            if A[j] < A[j_min]:
                j_min = j
        if j_min != i:
            swap(A, i, j_min)
    return A


def insertion_sort(A):
    n = len(A)
    for i in range(1, n):
        x = A[i]
        j = i - 1
        while j >= 0 and A[j] > x:
            A[j + 1] = A[j]
            j -= 1
        A[j + 1] = x
    return A


def heapify(A, n, i):
    """
    Maintain the heap property for a subtree rooted at index `i`.

    Parameters:
        A: list[int] - The array to heapify.
        n: int - Size of the heap.
        i: int - Index of the current root.

    Returns:
        None (modifies the array in place).
    """
    largest = i
    l = 2 * i + 1
    r = 2 * i + 2

    if l < n and A[i] < A[l]:
        largest = l

    if r < n and A[largest] < A[r]:
        largest = r

    if largest != i:
        A[i], A[largest] = A[largest], A[i]
        heapify(A, n, largest)


def heap_sort(A):
    """
    Perform heap sort on the input array.

    Parameters:
        A: list[int] - The array to sort.

    Returns:
        list[int] - The sorted array.
    """
    n = len(A)

    for i in range(n // 2 - 1, -1, -1):
        heapify(A, n, i)

    for i in range(n - 1, 0, -1):
        A[i], A[0] = A[0], A[i]
        heapify(A, i, 0)

    return A


def merge_sort(A, left, mid, right):
    """
    Merge two sorted subarrays into a single sorted array.

    Parameters:
        A: list[int] - The array to merge.
        left: int - Start index of the first subarray.
        mid: int - End index of the first subarray.
        right: int - End index of the second subarray.

    Returns:
        None (modifies the array in place).
    """
    L = A[left:mid + 1]
    R = A[mid + 1:right + 1]

    i = j = 0
    k = left

    while i < len(L) and j < len(R):
        if L[i] <= R[j]:
            A[k] = L[i]
            i += 1
        else:
            A[k] = R[j]
            j += 1
        k += 1

    while i < len(L):
        A[k] = L[i]
        i += 1
        k += 1

    while j < len(R):
        A[k] = R[j]
        j += 1
        k += 1


def _top_down_merge_sort(A, left, right):
    """
    Perform top-down merge sort on the array.

    Parameters:
        A: list[int] - The array to sort.
        left: int - Start index.
        right: int - End index.

    Returns:
        None (sorts the array in place).
    """
    if left >= right:
        return

    mid = (left + right) // 2
    _top_down_merge_sort(A, left, mid)
    _top_down_merge_sort(A, mid + 1, right)
    merge_sort(A, left, mid, right)


def top_down_merge_sort(A):
    return _top_down_merge_sort(A, 0, len(A) - 1)


class AlgorithmType(Enum):
    pass


class SortType(AlgorithmType):
    BUBBLE_SORT = "Bubble Sort"
    INSERTION_SORT = "Insert Sort"
    SELECTION_SORT = "Selection Sort"
    HEAP_SORT = "Heap Sort"
    QUICK_SORT = "Quick Sort"
    MERGE_SORT = "Merge Sort"


SORTS = {
    SortType.BUBBLE_SORT: Process(SortType.BUBBLE_SORT.value,
                                    bubble_sort),
    SortType.INSERTION_SORT: Process(SortType.INSERTION_SORT.value,
                                    insertion_sort),
    SortType.SELECTION_SORT: Process(SortType.SELECTION_SORT.value,
                                        selection_sort),
    SortType.HEAP_SORT: Process(SortType.HEAP_SORT.value,
                                    heap_sort),
    SortType.QUICK_SORT: Process(SortType.QUICK_SORT.value,
                                    quick_sort),
    SortType.MERGE_SORT: Process(SortType.MERGE_SORT.value,
                                    top_down_merge_sort)
}

SORTS_CONFIGURATION = \
    {
        SortType.BUBBLE_SORT: {
            DataGenerationType.NOT_CHANGED:
                ChartProperties(label="not changed - general case O(n^2)",
                                style="", color=""),
            DataGenerationType.RANDOM:
                ChartProperties("randomized - general case O(n^2)", "", ""),
            DataGenerationType.REVERSED:
                ChartProperties("reversed - worst case O(n^2) ", "", ""),
            DataGenerationType.SORTED:
                ChartProperties("sorted - best case O(n)", "", ""),
        },
        SortType.INSERTION_SORT: {
            DataGenerationType.NOT_CHANGED:
                ChartProperties(label="not changed - general case O(n^2)",
                                style="", color=""),
            DataGenerationType.RANDOM:
                ChartProperties("randomized - general case O(n^2)", "", ""),
            DataGenerationType.REVERSED:
                ChartProperties("reversed - worst case O(n^2) ", "", ""),
            DataGenerationType.SORTED:
                ChartProperties("sorted - best case O(n)", "", ""),
        },
        SortType.SELECTION_SORT: {
            DataGenerationType.NOT_CHANGED:
                ChartProperties(label="not changed O(n^2)",
                                style="", color=""),
            DataGenerationType.RANDOM:
                ChartProperties("randomized O(n^2)", "", ""),
            DataGenerationType.REVERSED:
                ChartProperties("reversed O(n^2) ", "", ""),
            DataGenerationType.SORTED:
                ChartProperties("sorted O(n^2)", "", ""),
        },
        SortType.HEAP_SORT: {
            DataGenerationType.NOT_CHANGED:
                ChartProperties(label="not changed O(n log n)",
                                style="", color=""),
            DataGenerationType.RANDOM:
                ChartProperties("randomized O(n log n)", "", ""),
            DataGenerationType.REVERSED:
                ChartProperties("reversed O(n log n) ", "", ""),
            DataGenerationType.SORTED:
                ChartProperties("sorted O(n log n)", "", ""),
        },
        SortType.QUICK_SORT: {
            DataGenerationType.NOT_CHANGED:
                ChartProperties(label="not changed - best case O(n log n)",
                                style="", color=""),
            DataGenerationType.RANDOM:
                ChartProperties("randomized - worst case O(n log n)", "", ""),
            DataGenerationType.REVERSED:
                ChartProperties("reversed - worst case O(n^2) ", "", ""),
            DataGenerationType.SORTED:
                ChartProperties("sorted - best case O(n^2)", "", ""),
        },
        SortType.MERGE_SORT: {
            DataGenerationType.NOT_CHANGED:
                ChartProperties(label="not changed O(n log n)",
                                style="", color=""),
            DataGenerationType.RANDOM:
                ChartProperties("randomized O(n log n)", "", ""),
            DataGenerationType.REVERSED:
                ChartProperties("reversed O(n log n) ", "", ""),
            DataGenerationType.SORTED:
                ChartProperties("sorted O(n log n)", "", ""),
        },
    }


class AlgorithmTester(metaclass=Singleton):
    """
    Main entry point for benchmarking algorithms with various datasets.

    Attributes:
        _generator (DataGenerator): Generates datasets for testing.
        _processor (DataProcessor): Runs algorithms on datasets.
        _displayer (ProcessChartDisplayer): Displays benchmark results.
        _sample_ids (Dict[AlgorithmType, List[DataGenerationType]]): Stores enabled generation types for each algorithm.
        _sample_configuration (Dict[AlgorithmType, Tuple[int, int]]): Stores configuration for sample sizes and counts.

    Methods:
        add_algorithm(algorithm: Process): Adds an algorithm for benchmarking.
        set_display_configuration(algorithm_type: AlgorithmType, _type: DataGenerationType, chart_properties: ChartProperties): Configures display properties for charts.
        set_sample_configuration(algorithm_type: AlgorithmType, sample_size: int, sample_count: int): Configures sample generation.
        enable_sample_type(algorithm_type: AlgorithmType, _type: DataGenerationType): Enables a generation type for testing.
        test(_type: AlgorithmType): Runs the benchmark for a specific algorithm.
        clear(): Resets all configurations and clears stored data.
    """
    def __init__(self) -> None:
        self._generator = DataGenerator()
        self._processor = DataProcessor()
        self._displayer = ProcessChartDisplayer()

        self._sample_ids: dict[AlgorithmType, list[DataGenerationType]] = {}
        self._sample_configuration: dict[AlgorithmType, tuple[int, int]] = {}

    def add_algorithm(self, algorithm: Process):
        self._processor.add(algorithm)

    def set_display_configuration(self, algorithm_type: AlgorithmType,
                                    _type: DataGenerationType,
                                        chart_properties: ChartProperties):
        if chart_properties is None:
            chart_properties = ChartProperties("Default label", "-", "r")

        self._displayer[algorithm_type.value][_type] = chart_properties

    def set_sample_configuration(self, algorithm_type: AlgorithmType,
                                    sample_size: int, sample_count: int):
        self._sample_configuration.update({algorithm_type.value: (sample_size,
                                                                  sample_count)})

    def enable_sample_type(self, algorithm_type: AlgorithmType,
                            _type: DataGenerationType):
        if algorithm_type.value not in self._sample_ids:
            self._sample_ids.update({algorithm_type.value: []})
        self._sample_ids[algorithm_type.value].append(_type)

    def test(self, _type: AlgorithmType):
        set_cpu_affinity()
        if _type.value in self._sample_configuration:
            sample_configuration = self._sample_configuration[_type.value]
            sample_size = sample_configuration[0]
            sample_count = sample_configuration[1]
            self._generator.generate(sample_size, sample_count)
        else:
            self._generator.generate(step_size=200, sample_count=15)

        for generation_type in self._sample_ids[_type.value]:
            sample = self._generator.get(generation_type)
            result = self._processor.run(_type.value, sample)
            self._displayer.add_result(result)

        self._displayer.display_by_process(_type.value)
        self._displayer.clear_by_process(_type.value)

    def clear(self):
        self._sample_ids.clear()
        self._sample_configuration.clear()
        self._displayer.clear()
        self._generator.clear()
        self._processor.clear()


def configure_sorts():
    AlgorithmTester().clear()
    for sort_id, sort_process in SORTS.items():
        AlgorithmTester().add_algorithm(sort_process)

    for sort_id, configuration in SORTS_CONFIGURATION.items():
        for generation_type, chart_properties in configuration.items():
            AlgorithmTester().enable_sample_type(sort_id, generation_type)
            AlgorithmTester().set_display_configuration(sort_id,
                                                        generation_type,
                                                        chart_properties)
            AlgorithmTester().set_sample_configuration(sort_id, 250, 15)
