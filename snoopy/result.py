import os
from abc import ABC, abstractmethod
from collections import OrderedDict as OrdD
from timeit import default_timer
from typing import Dict, NamedTuple

import numpy as np

from ._logging import get_logger
from .nn import ProgressResult

_logger = get_logger(__name__)


class Observer(ABC):
    @abstractmethod
    def on_update(self, name: str, progress_result: ProgressResult):
        pass


class TimeMeasuringObserver(Observer):
    def __init__(self, num_measurements: int, file: str):
        self._last_time = None
        self._batch_times = []
        self._num_measurements_left = num_measurements
        self._file = file

    def on_update(self, name: str, progress_result: ProgressResult):
        new_time = default_timer()
        if self._last_time is not None:
            self._batch_times.append(new_time - self._last_time)
        self._last_time = new_time

        self._num_measurements_left -= 1
        if self._num_measurements_left == 0:
            with open(self._file, "w") as f:
                f.write(str(self._batch_times))

            exit(5)


class ResultStoringObserver(Observer):
    def __init__(self):
        # Order is needed so that 'state_string' retains order of embeddings
        self._result_history = OrdD()

    def on_update(self, name: str, progress_result: ProgressResult):
        if name not in self._result_history:
            self._result_history[name] = []

        self._result_history[name].append(progress_result)

        state_string = "Current result: "
        for name in self._result_history:
            state_string += f"\n\t{name}: {self._result_history[name][-1]}"
        _logger.info(state_string)

    def store(self, output_files_path: str, filename_mapping: Dict[str, str] = None):
        # Use name given to embedding as filename
        if filename_mapping is None:
            filename_mapping = {x: x for x in self._result_history}

        assert set(self._result_history.keys()) == set(filename_mapping.keys()), \
            f"Filename should be specified for each embedding!"

        for key in self._result_history:
            chain = self._result_history[key]
            x = [i.num_train_points_processed for i in chain]
            y = [i.num_errors for i in chain]

            _logger.debug(f"Storing embedding '{key}' to file '{filename_mapping[key]}'")
            np.savez(os.path.join(output_files_path, filename_mapping[key]), n=x, err=y)


class SinglePlotResult:
    def __init__(self, n: np.ndarray, err: np.ndarray):
        self.n = n
        self.err = np.divide(err, err[0])


class PlotResult(NamedTuple):
    overall: SinglePlotResult
    individual: Dict[str, SinglePlotResult]


# This observer assumes that on_update is called after every pull of any arm
# Whereas ResultStoringObserver is meant to plot progression of each arm independently, this observer aims
# to plot progression of all arms together
class TimeProgressionObserver(Observer):
    def __init__(self):
        # Assumption is that this is equal to the number of arm pulls
        self._num_updates = 0

        # History of changes for arms, where x denote the point in time (in terms of pulls of all arms)
        self._xs = OrdD()
        self._ys = OrdD()

        # History of changes for the minimum among arms
        self._x_min_curve = []
        self._y_min_curve = []

    def on_update(self, name: str, progress_result: ProgressResult):
        if name not in self._xs:
            self._xs[name] = []
            self._ys[name] = []

        if progress_result.num_train_points_processed == 0:
            self._xs[name].append(0)
        else:
            self._xs[name].append(self._num_updates)
        self._ys[name].append(progress_result.num_errors)

        # When notified of initial error more than once, do not consider it as a lever pull
        if self._num_updates > 0 and progress_result.num_train_points_processed == 0:
            return

        # Update minimum of arms
        min_error = min([self._ys[key][-1] for key in self._ys])
        self._x_min_curve.append(self._num_updates)
        self._y_min_curve.append(min_error)

        # Receiving initial error does not constitute a lever pull, so only update when lever has actually been pulled
        self._num_updates += 1

    def get_plot_data(self) -> PlotResult:
        individual = {}
        for key in self._xs:
            individual[key] = SinglePlotResult(n=self._xs[key], err=self._ys[key])

        overall = SinglePlotResult(n=np.array(self._x_min_curve), err=np.array(self._y_min_curve))
        return PlotResult(individual=individual, overall=overall)
