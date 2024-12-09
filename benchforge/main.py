from enum import Enum
from benchforge.data.generation import DataGenerator
from benchforge.data.model import DataGenerationType
from benchforge.pipeline.process import DataProcessor, Process
from benchforge.visual.display import ProcessChartDisplayer, ChartProperties
import psutil
import os


def set_cpu_affinity(cpu_id=0):
    """Force the program to run on a specific CPU core."""
    p = psutil.Process(os.getpid())
    p.cpu_affinity([cpu_id])


class AlgorithmType(Enum):
    pass


class Benchmark:
    """
    Main entry point for benchmarking algorithms with various datasets.

    Attributes:
        _generator (DataGenerator): Generates datasets for testing.
        _processor (DataProcessor): Runs algorithms on datasets.
        _displayer (ProcessChartDisplayer): Displays benchmark results.
        _sample_ids (Dict[AlgorithmType, List[DataGenerationType]]): Stores
        enabled generation types for each algorithm.
        _sample_configuration (Dict[AlgorithmType, Tuple[int, int]]): Stores
        configuration for sample sizes and counts.

    Methods:
        add_algorithm(algorithm: Process): Adds an algorithm for benchmarking.
        set_display_configuration(algorithm_type: AlgorithmType,
        _type: DataGenerationType, chart_properties: ChartProperties):
        Configures display properties for charts.
        set_sample_configuration(algorithm_type: AlgorithmType, sample_size: int,
        sample_count: int): Configures sample generation.
        enable_sample_type(algorithm_type: AlgorithmType,
        _type: DataGenerationType): Enables a generation type for testing.
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
