import sys
import os
from enum import Enum


sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
MAX_RECURSION = 10000
sys.setrecursionlimit(MAX_RECURSION)


from benchforge.visual.display import ChartProperties
from benchforge.pipeline.process import Process
from benchforge.main import Benchmark
from benchforge.data.model import DataGenerationType


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
    left = 2 * i + 1
    right = 2 * i + 2

    if left < n and A[i] < A[left]:
        largest = left

    if right < n and A[largest] < A[right]:
        largest = right

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


def get_sorts_benchmark():
    benchmark = Benchmark()
    benchmark.clear()
    for sort_id, sort_process in SORTS.items():
        benchmark.add_algorithm(sort_process)

    for sort_id, configuration in SORTS_CONFIGURATION.items():
        for generation_type, chart_properties in configuration.items():
            benchmark.enable_sample_type(sort_id, generation_type)
            benchmark.set_display_configuration(sort_id,
                                                        generation_type,
                                                        chart_properties)
            benchmark.set_sample_configuration(sort_id, 250, 15)
    return benchmark
