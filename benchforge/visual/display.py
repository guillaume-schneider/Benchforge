from dataclasses import dataclass
from benchforge.data.model import DataGenerationType
from benchforge.pipeline.result import SampleProcessResult
import matplotlib.pyplot as plt


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
            result_dict = {result.process_id: ProcessChart(result.process_id)}
            self._process_charts.update(result_dict)
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
