import os
import sys
from collections import OrderedDict as OrdD
from copy import copy
from dataclasses import dataclass
from typing import Callable, Dict, List, Optional, Tuple
from typing import OrderedDict

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from matplotlib.axes import Axes
from matplotlib.figure import Figure
from mpl_toolkits.axes_grid1.inset_locator import mark_inset

from snoopy.nn import Arm
from snoopy.result import PlotResult, SinglePlotResult, TimeProgressionObserver
from snoopy.strategy import SuccessiveHalvingAlgorithm, UniformAllocationAlgorithm
from snoopy.tests.test_strategy import ArmSimulator
from times import times

date_month_published = {
    # https://arxiv.org/abs/1404.5997
    'alexnet': "2014-04",

    # https://arxiv.org/abs/1905.11946
    'efficientnet_b0': "2019-05-0",
    'efficientnet_b1': "2019-05-1",
    'efficientnet_b2': "2019-05-2",
    'efficientnet_b3': "2019-05-3",
    'efficientnet_b4': "2019-05-4",
    'efficientnet_b5': "2019-05-5",
    'efficientnet_b6': "2019-05-6",
    'efficientnet_b7': "2019-05-7",

    # https://arxiv.org/abs/1409.4842
    'googlenet': "2014-09-0",

    # https://arxiv.org/abs/1512.00567
    'inception': "2015-12",
    'pca_128': "0-3",
    'pca_32': "0-1",
    'pca_64': "0-2",
    'raw': "0-0",

    # https://arxiv.org/abs/1603.05027
    'resnet_101_v2': "2016-03-2",
    'resnet_152_v2': "2016-03-3",
    'resnet_50_v2': "2016-03-1",

    # https://arxiv.org/abs/1409.1556
    'vgg16': "2014-09-1",
    'vgg19': "2014-09-2",

    # https://arxiv.org/abs/1810.04805
    'bert_cased_pool': "2018-10-1",
    'bert_uncased_pool': "2018-10-2",
    'bert_cased': "2018-10-3",
    'bert_uncased': "2018-10-4",
    'bert_cased_large_pool': "2018-10-5",
    'bert_uncased_large_pool': "2018-10-6",
    'bert_cased_large': "2018-10-7",
    'bert_uncased_large': "2018-10-8",

    # https://www.jmlr.org/papers/volume3/bengio03a/bengio03a.pdf
    'nnlm_128': "2003-03-3",
    'nnlm_128_normalization': "2003-03-4",
    'nnlm_50_normalization': "2003-03-2",
    'nnlm_50': "2003-03-1",

    # https://arxiv.org/abs/1802.05365
    'elmo': "2018-02",

    # https://arxiv.org/abs/1803.11175
    'use': "2018-03-1",
    'use_large': "2018-03-2",

    # https://arxiv.org/abs/1906.08237
    'xlnet': "2019-06-1",
    'xlnet_large': "2019-06-2"
}


class DatasetEmbeddings:
    def __init__(self, data_name: str, folder_path: str, cosine_distance: bool):
        self.data_name = data_name
        self.cosine_distance = cosine_distance
        if data_name in {"cifar10", "cifar100", "mnist"}:
            self.embed_names = ['alexnet', 'efficientnet_b0', 'efficientnet_b1', 'efficientnet_b2', 'efficientnet_b3',
                                'efficientnet_b4', 'efficientnet_b5', 'efficientnet_b6', 'efficientnet_b7',
                                'googlenet', 'inception', 'resnet_101_v2', 'resnet_152_v2', 'resnet_50_v2', 'vgg16',
                                'vgg19', 'pca_32', 'pca_64', 'pca_128', 'raw']

        else:
            self.embed_names = ['bert_cased', 'bert_cased_pool', 'bert_uncased_large_pool', 'nnlm_128',
                                'nnlm_50_normalization', 'xlnet', 'bert_cased_large', 'bert_uncased',
                                'bert_uncased_pool', 'nnlm_128_normalization', 'use', 'xlnet_large',
                                'bert_cased_large_pool', 'bert_uncased_large', 'elmo', 'nnlm_50', 'use_large']

        global date_month_published
        self.embed_names = sorted(self.embed_names, key=lambda x: date_month_published[x], reverse=True)

        if cosine_distance:
            cosine_str = "-cosine"
        else:
            cosine_str = ""

        path_to_convergence_curves = os.path.join(folder_path, data_name)
        self.embedding_to_cc = OrdD({})
        for name in self.embed_names:
            self.embedding_to_cc[name] = np.load(os.path.join(path_to_convergence_curves,
                                                              f"{data_name}-{name}-errs{cosine_str}-0.0.npy"))

        if data_name == "cifar10":
            self.num_classes, self.sota = 10, 0.0063
        elif data_name == "imagenet":
            self.num_classes_, self.sota = 1000, 0.115
        elif data_name == "cifar100":
            self.num_classes, self.sota = 100, 0.0649
        elif data_name == "mnist":
            self.num_classes, self.sota = 10, 0.0016
        elif data_name == "yelp":
            self.num_classes, self.sota = 5, 0.278
        elif data_name == "imdb_reviews":
            self.num_classes, self.sota = 2, 0.0379
        elif data_name == "sst2":
            self.num_classes, self.sota = 2, 0.032
        else:
            self.num_classes, self.sota = None, None


@dataclass
class ZoomInParams:
    capture_x_min: float
    capture_y_min: float
    view_x_min: float
    view_y_min: float
    corner_1: int = 1
    corner_2: int = 2


def _zoom_in(axes, axes_inner, zoom_params: ZoomInParams, view_x_span_perc: float, view_y_span_perc: float):
    line_params = {"ec": "0.5", "ls": "--", "lw": 1.5, "alpha": 0.5}

    # https://matplotlib.org/3.1.0/gallery/ticks_and_spines/spines.html
    # https://stackoverflow.com/a/7780971
    # Set params of bigger rectangle
    for pos in ["top", "bottom", "left", "right"]:
        axes_inner.spines[pos].set(**line_params)

    axes_inner.grid(False)
    final_limits = axes.axis()
    x_span = final_limits[1] - final_limits[0]
    y_span = final_limits[3] - final_limits[2]
    axes_inner.set_xlim(zoom_params.capture_x_min, zoom_params.capture_x_min + view_x_span_perc * x_span)
    axes_inner.set_ylim(zoom_params.capture_y_min, zoom_params.capture_y_min + view_y_span_perc * y_span)
    axes_inner.set_xticklabels('')
    axes_inner.set_yticklabels('')

    # Set params of smaller rectangle and connecting lines
    mark_inset(axes, axes_inner, loc1=zoom_params.corner_1, loc2=zoom_params.corner_2, **line_params)


def _legend(figure, axes_with_legend, axes_with_desired_lines, num_col: int):
    def remove_inset_label(lines_, labels_):
        lines_updated, labels_updated = [], []
        for i in range(len(labels_)):
            if labels_[i] != "indicate_inset":
                lines_updated.append(lines_[i])
                labels_updated.append(labels_[i])

        return lines_updated, labels_updated

    plt.legend()
    axes_with_legend.get_legend().set_visible(False)
    lines, labels = remove_inset_label(*axes_with_desired_lines.get_legend_handles_labels())
    lines_copy = [copy(i) for i in lines]
    # https://stackoverflow.com/a/48308814
    for line in lines_copy:
        line.set_linewidth(3.0)
    figure.legend(lines_copy, labels, loc='upper center', ncol=num_col, prop={'size': 20})
    plt.subplots_adjust(top=0.75, bottom=0.15)


def _ber_np(arr: np.ndarray, c: int) -> np.ndarray:
    value_to_sqrt = 1 - (c / (c - 1)) * arr
    value_to_sqrt[value_to_sqrt < 0.0] = 0.0
    return np.divide(arr, 1 + np.sqrt(value_to_sqrt))


class PlotDataGenerator:
    @staticmethod
    def get_date_line(arms: OrderedDict[str, Arm]) -> PlotResult:
        global date_month_published
        # Sort according to date
        names_with_dates = sorted([(name, date_month_published[name.split("-")[0]]) for name in arms.keys()],
                                  reverse=True, key=lambda x: x[1])

        # All plots
        xs = OrdD({arm_name: [0] for arm_name in arms})
        ys = OrdD({arm_name: [arms[arm_name].initial_error.num_errors] for arm_name in arms})

        # Infer initial error from first arm
        n = [0]
        err = [list(arms.values())[0].initial_error.num_errors]

        pull_cnt = 1
        lowest_err_so_far = None
        for name, date in names_with_dates:
            arm = arms[name]
            while arm.can_progress():
                result = arm.progress()
                if lowest_err_so_far:
                    error_to_append = min(lowest_err_so_far, result.num_errors)
                else:
                    error_to_append = result.num_errors
                xs[name].append(pull_cnt)
                ys[name].append(error_to_append)
                n.append(pull_cnt)
                err.append(error_to_append)
                pull_cnt += 1

            if lowest_err_so_far is None or err[-1] < lowest_err_so_far:
                lowest_err_so_far = err[-1]

        individual = {i: SinglePlotResult(n=np.array(xs[i]), err=np.array(ys[i])) for i in arms}
        return PlotResult(overall=SinglePlotResult(n=np.array(n), err=np.array(err)), individual=individual)

    @staticmethod
    def get_perfect_line_one_only(arms: OrderedDict[str, Arm]) -> PlotResult:
        x_overall = [0]
        y_overall = [list(arms.values())[0].initial_error.num_errors]

        # All plots
        x_individual = OrdD({arm_name: [0] for arm_name in arms})
        y_individual = OrdD({arm_name: [arms[arm_name].initial_error.num_errors] for arm_name in arms})

        temp_results = {}
        for arm_name in arms:
            arm = arms[arm_name]
            x = []
            y = []
            i = 0
            while arm.can_progress():
                result = arm.progress()
                i += 1
                x.append(i)
                y.append(result.num_errors)

            temp_results[arm_name] = (x, y)

        best_arms = list(map(lambda q: q[0], sorted([(i, temp_results[i][1][-1]) for i in arms], key=lambda q: q[1])))

        first = True
        for best_arm in best_arms:
            last_x = x_overall[-1]
            x_to_add = [i + last_x for i in temp_results[best_arm][0]]

            x_individual[best_arm].extend(x_to_add)
            x_overall.extend(x_to_add)
            y_individual[best_arm].extend(temp_results[best_arm][1])

            if first:
                y_overall.extend(temp_results[best_arm][1])
                first = False
            else:
                last_y = y_overall[-1]
                y_overall.extend([last_y for _ in range(len(x_to_add))])

        individual = OrdD({i: SinglePlotResult(n=np.array(x_individual[i]), err=np.array(y_individual[i])) for i in
                           arms})
        overall = SinglePlotResult(n=np.array(x_overall), err=np.array(y_overall))
        return PlotResult(overall=overall, individual=individual)

    @staticmethod
    def get_perfect_line(arms: OrderedDict[str, Arm]) -> PlotResult:
        individual = {}
        arm_names_left = list(arms.keys())

        # How many rounds of pulling there are (length of 'original' optimal convergence curve)
        num_pull_rounds = 0

        # How many times all arms were pulled (length of convergence curve)
        num_all_pulls = 0

        # All plots
        xs = OrdD({arm_name: [0] for arm_name in arms})
        ys = OrdD({arm_name: [arms[arm_name].initial_error.num_errors] for arm_name in arms})

        # Optimal plot
        x = [0]

        # Infer initial error from first arm
        first_arm = list(arms.values())[0]
        y = [first_arm.initial_error.num_errors]

        while len(arm_names_left) > 0:
            arm_names_to_remove = []

            at_least_one_arm_pulled_in_current_round = False
            for arm_name in arm_names_left:
                arm = arms[arm_name]
                if arm.can_progress():
                    at_least_one_arm_pulled_in_current_round = True
                    num_all_pulls += 1
                    result = arm.progress()
                    xs[arm_name].append(num_pull_rounds + 1)
                    ys[arm_name].append(result.num_errors)

                else:
                    arm_names_to_remove.append(arm_name)

            if at_least_one_arm_pulled_in_current_round:
                num_pull_rounds += 1

                # Value of optimal convergence curve is minimum of all arms
                # If one of the arms has converged, its last value can always be taken as a minimum
                x.append(num_pull_rounds)
                y.append(min([chain[-1] for chain in list(ys.values())]))

            arm_names_left = [i for i in arm_names_left if i not in arm_names_to_remove]

        # Extend optimal curve to the length of all other curves by replicating last error value
        last_y = y[-1]
        for i in range(num_all_pulls - num_pull_rounds):
            x.append(num_pull_rounds + i + 1)
            y.append(last_y)

        for arm_name in arms:
            individual[arm_name] = SinglePlotResult(n=xs[arm_name], err=ys[arm_name])

        overall = SinglePlotResult(n=np.array(x), err=np.array(y))
        return PlotResult(individual=individual, overall=overall)

    @staticmethod
    def get_uniform_line(arms: OrderedDict[str, Arm]) -> PlotResult:
        observer = TimeProgressionObserver()
        uaa = UniformAllocationAlgorithm(arms, observer)
        uaa.uniform_allocation()
        return observer.get_plot_data()

    @staticmethod
    def get_sh_line(arms: OrderedDict[str, Arm]) -> PlotResult:
        observer = TimeProgressionObserver()
        shs = SuccessiveHalvingAlgorithm(arms, observer)
        shs.successive_halving_with_doubling(eta=2, snoopy=False)
        return observer.get_plot_data()

    @staticmethod
    def get_sh_imp_line(arms: OrderedDict[str, Arm]) -> PlotResult:
        observer = TimeProgressionObserver()
        shs = SuccessiveHalvingAlgorithm(arms, observer)
        shs.successive_halving_with_doubling(eta=2, snoopy=True)
        return observer.get_plot_data()


def plot_main_lines_with_info(dse: DatasetEmbeddings, step_sizes: Dict[str, int], plot_filename: str = None,
                              eps: float = None, with_time: bool = True, time_x_axis: bool = False,
                              zoom_in: bool = False, zoom_params: ZoomInParams = ZoomInParams(0.0, 0.0, 0.05, 0.5),
                              plot_handles: Tuple[Figure, Axes] = None) -> Optional[Tuple[Figure, Axes]]:
    def prepare_arms(name_to_data: OrderedDict[str, np.ndarray], step: int) -> List[OrderedDict[str, Arm]]:
        min_num_reps = None
        for name in name_to_data:
            current_num_reps = name_to_data[name].shape[0]
            if min_num_reps is None or current_num_reps < min_num_reps:
                min_num_reps = current_num_reps

        arms_list = []

        for repetition in range(min_num_reps):
            arms = OrdD()
            for embedding_name in name_to_data:
                data = name_to_data[embedding_name]
                n = np.arange(data.shape[1])
                errs = data[repetition, :]

                arms[embedding_name] = ArmSimulator(x=n[1:], y=errs[1:], initial_error=errs[0], increment=int(step))

            arms_list.append(arms)

        return arms_list

    def get_stats(arms_list_copy: List[OrderedDict[str, Arm]], data_fn: Callable,
                  perfect_results: Optional[List[PlotResult]], label: str) -> Optional[List[PlotResult]]:
        if perfect_results is None:
            perfect_strategy = True
        else:
            perfect_strategy = False

        # List of all plot results
        results_list = []

        # What will be used for plotting
        n = np.array([])
        err = np.array([])
        first = True
        times_per_run = []

        # Go through all repetitions
        for repetition_index, arms in enumerate(arms_list_copy):
            result: PlotResult = data_fn(arms)
            results_list.append(result)
            if first:
                first = False
                n = result.overall.n
                err = result.overall.err

            else:
                err = np.vstack([err, result.overall.err])

            # For sequential CCC
            if with_time:
                # Create an array of embedding names used in a specific run of a sequential CCC
                which_embedding_was_used = ["empty"] + ["" for _ in range(result.overall.n.size - 1)]
                for embedding_name in result.individual:
                    embedding_n = result.individual[embedding_name].n
                    for index in embedding_n:
                        if index != 0:
                            which_embedding_was_used[index] = embedding_name

                assert "" not in which_embedding_was_used, "For some index, an embedding was not set"

                # Transform embedding names to times
                times_per_run.append([times[dse.data_name][
                                          embedding_name.split("-")[0]][
                                          str(step_sizes[label])] / 3600.0
                                      for embedding_name in which_embedding_was_used])

        # 1. PREPARE DATA THAT WILL BE PLOTTED
        if time_x_axis:
            times_matrix = np.array(times_per_run)
            x = np.cumsum(np.mean(times_matrix, axis=0)[1:])

        else:
            x = n[1:]

        # Listed all errors at first x value, then at second, ...
        err_plot = _ber_np(err[:, 1:], c=dse.num_classes)
        if label == "Perfect":
            line = "dashed"
        else:
            line = "solid"
        ax_1.fill_between(x, np.percentile(err_plot, 5.0, axis=0), np.percentile(err_plot, 95.0, axis=0),
                          alpha=0.2)
        ax_1.plot(x, np.median(err_plot, axis=0), label=f"{label}", linestyle=line, linewidth=2)
        if zoom_in:
            axins.fill_between(x, np.percentile(err_plot, 5.0, axis=0), np.percentile(err_plot, 95.0, axis=0),
                               alpha=0.2)
            axins.plot(x, np.median(err_plot, axis=0), label=f"{label}", linestyle=line, linewidth=2)

        # 2. CALCULATE STATS
        # Time in "number of pulls"
        if perfect_strategy:
            perfect_results = results_list

        times_perfect = []

        # Actual time in seconds
        times_actual_perfect = []

        for i in range(len(perfect_results)):
            perfect_errs = _ber_np(perfect_results[i].overall.err, c=dse.num_classes)
            current_errs = _ber_np(results_list[i].overall.err, c=dse.num_classes)

            # Time to perfect (in terms of arm pulls)
            # Last reported perfect error (= value of perfect curve where the curve is flat)
            perfect_err_final = perfect_errs[-1]
            num_arm_pulls_to_perfect = np.argmax(current_errs <= perfect_err_final * (1 + eps))
            times_perfect.append(num_arm_pulls_to_perfect)

            # Info about times_per_run[i]:
            # # 1. Time at first index is 0, so it does not count
            # # 2. In order to have num_arm_pulls_to_perfect inclusive, + 1 is added
            if with_time:
                times_actual_perfect.append(np.sum(times_per_run[i][:(num_arm_pulls_to_perfect + 1)]))

        times_perfect = np.array(times_perfect)
        times_actual_perfect = np.array(times_actual_perfect)

        statistics = ""
        statistics += f"{label}:"
        statistics += f"\nTime (pulls) μ: {np.mean(times_perfect):.4f}, " \
                      f"σ: {np.std(times_perfect, ddof=1):.4f}"
        if with_time:
            statistics += f"\nTime (h) μ: {np.mean(times_actual_perfect):.4f}, " \
                          f"σ: {np.std(times_actual_perfect, ddof=1):.4f}"
        statistics += "\n"
        plt_txt[label] = statistics

        # Perfect curve
        if perfect_strategy is None:
            return results_list

    assert with_time or not time_x_axis, \
        "In order for x axis to be time, with_actual_time needs to be set to True"

    if eps is None:
        eps = 1e-15

    if time_x_axis:
        x_label = "Time (h)"
    else:
        x_label = "Number of arm pulls"

    # Text for statistics, added by each strategy on its own
    plt_txt = {}

    # Produce new plot only if first call
    if plot_handles is None:
        f, (ax_1, ax_2) = plt.subplots(1, 2, figsize=(13, 6), sharey="col")
    else:
        f, ax_1 = plot_handles
        ax_2 = None

    # Prepare zoomed-in plot
    if zoom_in:
        axins = ax_1.inset_axes(
            [zoom_params.view_x_min, zoom_params.view_y_min, 0.6, 0.4])
    else:
        axins = None

    res = get_stats(prepare_arms(dse.embedding_to_cc, step_sizes["Perfect"]),
                    PlotDataGenerator.get_perfect_line_one_only, perfect_results=None,
                    label="Perfect")
    # get_stats(prepare_arms(dse.embedding_to_cc, step_sizes["Date"]), PlotDataGenerator.get_date_line,
    #           perfect_results=res, label="Date")
    get_stats(prepare_arms(dse.embedding_to_cc, step_sizes["Uniform allocation"]),
              PlotDataGenerator.get_uniform_line,
              perfect_results=res, label="Uniform allocation")
    get_stats(prepare_arms(dse.embedding_to_cc, step_sizes["SH w/o tangents"]),
              PlotDataGenerator.get_sh_line,
              perfect_results=res, label="SH w/o tangents")
    get_stats(prepare_arms(dse.embedding_to_cc, step_sizes["SH w/ tangents"]),
              PlotDataGenerator.get_sh_imp_line,
              perfect_results=res, label="SH w/ tangents")

    plt_txt_string = "\n".join(list(plt_txt.values()))

    # Set limits
    limits: Tuple = ax_1.axis()

    if dse.data_name in {"cifar10", "cifar100"} and dse.cosine_distance and time_x_axis:
        ax_1.axis((-0.03, 1.5, limits[2], limits[3]))

    elif dse.data_name == "mnist" and dse.cosine_distance and time_x_axis:
        ax_1.axis((-0.01, 0.5, limits[2], limits[3]))

    elif dse.data_name == "sst2" and dse.cosine_distance and time_x_axis:
        ax_1.axis((-0.08, 4.0, 0.1275, 0.24))

    elif dse.data_name == "imdb_reviews" and dse.cosine_distance and time_x_axis:
        ax_1.axis((-0.06, 3, limits[2], limits[3]))

    elif dse.data_name == "yelp" and dse.cosine_distance and time_x_axis:
        ax_1.axis((-0.8, 40, limits[2], limits[3]))

    # Plot the zoomed-in part and the connecting lines
    if zoom_in:
        _zoom_in(ax_1, axins, zoom_params=zoom_params, view_x_span_perc=0.15, view_y_span_perc=0.1)

    # Set y label the first time
    if plot_handles is None:
        ax_1.set(xlabel=x_label, ylabel="BER estimate")

    # Plot legend the second time
    else:
        _legend(f, ax_1, ax_1, num_col=2)
        ax_1.set(xlabel=x_label)

    print(plt_txt_string)
    # plt.annotate(plt_txt_string, xy=(0.05, 0.95), va="top", ha="left", xycoords='axes fraction', fontsize=6)

    if plot_handles is not None:
        if plot_filename:
            # f.savefig(plot_filename)
            f.savefig(plot_filename, bbox_inches="tight", pad_inches=0.02)
        else:
            plt.show()

    else:
        return f, ax_2


def error_evaluation(dse: DatasetEmbeddings,
                     ghp: bool = False,
                     plot_handles: Tuple[Figure, Axes] = None,
                     zoom_in: bool = False,
                     zoom_params: ZoomInParams = ZoomInParams(0.0, 0.0, 0.05, 0.5),
                     plot_filename: str = None) -> Optional[Tuple[Figure, Axes]]:
    def comp_sub_error_both(upper: float, lower: float, v: float):
        if upper < v:
            return v - upper
        elif lower > v:
            return lower - v
        else:
            return 0.0

    global names
    assert ghp or plot_handles is None, "Cannot pass handles for 1NN plot!"

    def ber(nn_err: float, c: int) -> float:
        value_to_sqrt = 1 - (c / (c - 1)) * nn_err
        if value_to_sqrt < 0.0:
            value_to_sqrt = 0.0
        return nn_err / (1 + np.sqrt(value_to_sqrt))

    # https://stackoverflow.com/a/50685454
    def equally_spaced(arr: list, num_sampled: int):
        arr_np = np.array(arr)
        indices = np.round(np.linspace(0, len(arr) - 1, num_sampled)).astype(np.int32)
        return arr_np[indices].tolist()

    with open("errors_data.txt", "r") as f:
        all_info = eval(f.read())[dse.data_name]

    if dse.cosine_distance:
        dict_key = "1-NN cosine"
        dict_key_loo = "1-NN cosine LOO"
    else:
        dict_key = "1-NN"
        dict_key_loo = "1-NN LOO"

    if plot_handles is None:
        f, (ax_1, ax_2) = plt.subplots(1, 2, figsize=(13, 6), sharey="col")
    else:
        f, ax_1 = plot_handles
        ax_2 = None

    embed_names_with_errors = [(i, all_info[i]["test"]["0.0"][dict_key][0]) for i in all_info.keys()]
    embeddings_sorted_by_error = list(map(lambda q: q[0], sorted(embed_names_with_errors, key=lambda q: q[1])))
    best_ghp_embedding = min([(i, all_info[i]["test"]["0.0"]["GHP Lower"][0]) for i in all_info.keys()],
                             key=lambda q: q[1])[0]
    best_loo_embedding = min([(i, all_info[i]["test"]["0.0"][dict_key_loo][0]) for i in all_info.keys()],
                             key=lambda q: q[1])[0]

    # Use raw always if available
    if not ghp:
        if dse.data_name in {"cifar10", "cifar100", "mnist"}:
            chosen_embeddings = ["raw"]
            embeddings_sorted_by_error.remove("raw")
            chosen_embeddings.extend(equally_spaced(embeddings_sorted_by_error, 4))
        # Otherwise not
        else:
            chosen_embeddings = equally_spaced(embeddings_sorted_by_error, 5)
    else:
        chosen_embeddings = ["GHP", "1-NN LOO", embeddings_sorted_by_error[0]]

    # Left -> plot of label noise
    def plot_left(axes):
        first = True
        upper_bounds = []
        lower_bounds = []
        for embedding in chosen_embeddings:
            if embedding == "GHP":
                current_dict = all_info[best_ghp_embedding]["test"]
            elif embedding == "1-NN LOO":
                current_dict = all_info[best_loo_embedding]["test"]
            else:
                current_dict = all_info[embedding]["test"]

            x = []
            y = []
            y_low = []
            y_high = []

            # Error variables
            e_both_mean = []
            e_both_std = []
            # End of score variables

            max_c = (dse.num_classes - 1.0) / dse.num_classes
            for noise_level in sorted(list(current_dict.keys())):
                noise_level_float = float(noise_level)
                lower_bound = noise_level_float * max_c
                upper_bound = dse.sota + noise_level_float * (max_c - dse.sota)

                # Only once construct LB and UB plots
                if first:
                    upper_bounds.append(upper_bound)
                    lower_bounds.append(lower_bound)
                x.append(noise_level_float)

                # Construct plot for each
                if embedding == "GHP":
                    values_ber = current_dict[noise_level]["GHP Lower"]
                elif embedding == "1-NN LOO":
                    values = current_dict[noise_level][dict_key_loo]
                    values_ber = np.array([ber(i, dse.num_classes) for i in values])
                else:
                    values = current_dict[noise_level][dict_key]
                    values_ber = np.array([ber(i, dse.num_classes) for i in values])

                # Plot variables
                y.append(np.median(values_ber))
                y_low.append(np.percentile(values_ber, 5.0))
                y_high.append(np.percentile(values_ber, 95.0))

                # Score variables
                errors = [comp_sub_error_both(upper=upper_bound, lower=lower_bound, v=value_ber)
                          for value_ber in values_ber]
                e_both_mean.append(np.mean(errors))
                e_both_std.append(np.std(errors))

            # Handle errors
            # print(f"{dse.data_name} - {embedding}: {np.mean(e_both_mean):.3f} ({np.mean(e_both_std):.3f})")

            if first:
                first = False
                axes.plot(x, lower_bounds, label="$\ell_\mathcal{D}(\\rho)$", color="black")
                axes.plot(x, upper_bounds, label="$u_\mathcal{D}(\\rho)$", color="black", linestyle="dashed")

            if embedding == "1-NN LOO":
                label = f"1NN LOO"
            elif embedding == "GHP":
                label = f"GHP"
            else:
                if ghp:
                    label = f"1NN"
                else:
                    label = names[embedding]
            axes.plot(x, y, label=label, linewidth=2)
            axes.fill_between(x, y_low, y_high, alpha=0.2)

        limits = axes.axis()
        axes.axis((0.0, 1.0, 0.0, min(limits[3], 1.0)))

    def plot_right(axes):
        # Right -> plot of convergence
        start, stop = None, None
        for embedding in chosen_embeddings:
            # Read data and compute BER
            data = dse.embedding_to_cc[embedding]
            data = _ber_np(np.divide(data, data[0, 0])[:, 1:], dse.num_classes)
            x = np.arange(data.shape[1]) + 1

            # Cut first 10 %
            num_training_pts = data.shape[1]
            start_index = num_training_pts // 10
            start, stop = start_index, num_training_pts

            # Plot
            data = data[:, start_index:]
            x = x[start_index:]
            y = np.median(data, axis=0)
            y_low = np.percentile(data, 5.0, axis=0)
            y_high = np.percentile(data, 95.0, axis=0)

            axes.plot(x, y, label=embedding, linewidth=2)
            axes.fill_between(x, y_low, y_high, alpha=0.2)

        limits = axes.axis()
        axes.axis((start, stop, limits[2], limits[3]))

    plot_left(ax_1)

    # Zoomed in the relevant part
    # https://matplotlib.org/gallery/subplots_axes_and_figures/zoom_inset_axes.html
    # https://matplotlib.org/3.1.1/api/_as_gen/mpl_toolkits.axes_grid1.inset_locator.mark_inset.html
    if zoom_in:
        axins = ax_1.inset_axes([zoom_params.view_x_min, zoom_params.view_y_min, 0.4, 0.4])
        plot_left(axins)
        _zoom_in(ax_1, axins, zoom_params=zoom_params, view_x_span_perc=0.1, view_y_span_perc=0.1)

    ax_1.set(xlabel="Label noise $\\rho$")

    # Only set y label on the left subplot
    if plot_handles is None:
        ax_1.set(ylabel="BER estimate")

    if not ghp:
        plot_right(ax_2)
        ax_2.set(xlabel="Training samples")
        ticks = ax_2.get_xticks().astype(np.int32)
        if np.all(ticks % 1000) == 0:
            new_ticks = []
            for tick in ticks:
                if tick == 0:
                    new_ticks.append("0")
                else:
                    new_ticks.append(f"{tick // 1000}K")

            ax_2.set_xticklabels(new_ticks)
        _legend(f, ax_2, ax_1, num_col=4)

    # For ghp set legend only the second time!
    elif ghp and plot_handles is not None:
        _legend(f, ax_1, ax_1, num_col=3)

    # Show image only if 1NN plot or second call of ghp
    if not ghp or plot_handles is not None:
        if plot_filename:
            # f.savefig(plot_filename)
            f.savefig(plot_filename, bbox_inches="tight", pad_inches=0.02)
        else:
            plt.show()

    if ghp and plot_handles is None:
        return f, ax_2


names = {
    'GHP': "GHP lower bound",
    'alexnet': "AlexNet",
    'efficientnet_b0': "EfficientNet-B0",
    'efficientnet_b1': "EfficientNet-B1",
    'efficientnet_b2': "EfficientNet-B2",
    'efficientnet_b3': "EfficientNet-B3",
    'efficientnet_b4': "EfficientNet-B4",
    'efficientnet_b5': "EfficientNet-B5",
    'efficientnet_b6': "EfficientNet-B6",
    'efficientnet_b7': "EfficientNet-B7",
    'googlenet': "GoogLeNet",
    'inception': "InceptionV3",
    'pca_128': "$\\mathrm{PCA}_{128}$",
    'pca_32': "$\\mathrm{PCA}_{32}$",
    'pca_64': "$\\mathrm{PCA}_{64}$",
    'raw': "RAW",
    'resnet_101_v2': "ResNet101-V2",
    'resnet_152_v2': "ResNet152-V2",
    'resnet_50_v2': "ResNet50-V2",
    'vgg16': "VGG16",
    'vgg19': "VGG19",

    'bert_cased_pool': "BERT BCP",
    'bert_uncased_pool': "BERT BUP",
    'bert_cased': "BERT BC",
    'bert_uncased': "BERT BU",
    'bert_cased_large_pool': "BERT LCP",
    'bert_uncased_large_pool': "BERT LUP",
    'bert_cased_large': "BERT LC",
    'bert_uncased_large': "BERT LU",
    'nnlm_128': "NNLM-en (128)",
    'nnlm_128_normalization': "NNLM-en (Norm., 128)",
    'nnlm_50_normalization': "NNLM-en (Norm., 50)",
    'nnlm_50': "NNLM-en (50)",
    'elmo': "ELMo",
    'use': "USE",
    'use_large': "USE Large",
    'xlnet': "XLNet",
    'xlnet_large': "XLNet Large"
}

if __name__ == "__main__":
    dataset_path = sys.argv[1]

    # Seaborn config
    sns.set_theme(font_scale=2.25, context="paper", style="whitegrid")

    # Data to analyze
    cifar100 = DatasetEmbeddings("cifar100", dataset_path, cosine_distance=True)
    cifar10 = DatasetEmbeddings("cifar10", dataset_path, cosine_distance=True)
    mnist = DatasetEmbeddings("mnist", dataset_path, cosine_distance=True)
    yelp = DatasetEmbeddings("yelp", dataset_path, cosine_distance=True)
    sst2 = DatasetEmbeddings("sst2", dataset_path, cosine_distance=True)
    imdb = DatasetEmbeddings("imdb_reviews", dataset_path, cosine_distance=True)

    step_sizes_ = {
        "cifar100": {
            "Perfect": 2048,
            "Uniform allocation": 2048,
            "SH w/o tangents": 1024,
            "SH w/ tangents": 1024,
        },

        "yelp": {
            "Perfect": 8192,
            "Uniform allocation": 8192,
            "SH w/o tangents": 8192,
            "SH w/ tangents": 4096,
        },
        "imdb_reviews": {
            "Perfect": 1024,
            "Uniform allocation": 1024,
            "SH w/o tangents": 512,
            "SH w/ tangents": 512,
        },
        "sst2": {
            "Perfect": 512,
            "Uniform allocation": 512,
            "SH w/o tangents": 512,
            "SH w/ tangents": 512,
        },
        "cifar10": {
            "Perfect": 2048,
            "Uniform allocation": 2048,
            "SH w/o tangents": 1024,
            "SH w/ tangents": 1024,
        },
        "mnist": {
            "Perfect": 2048,
            "Uniform allocation": 2048,
            "SH w/o tangents": 1024,
            "SH w/ tangents": 512,
        },
    }

    # 1. step_sizes: What batch size / step size to use for each strategy
    # 2. num_classes: Self explanatory, no need to set
    # 3. eps: What is counted as reaching lowest error -> (1 + eps) * lowest_error, set to None for no eps not to
    # 0.0! This only affects computed stats, not plots.
    # 4. with_time: whether to include actual time (h) stats, can be True only for CIFAR-100 and YELP
    # 5. time_x_axis: if True time (h) is x axis, otherwise number of pulls
    # 6. plot_filename: if set, the plot is saved otherwise it is displayed
    # 7. add_stats: whether stats are also added to the plot and displayed
    # 8. title: plot title
    print("CIFAR-10")
    p = plot_main_lines_with_info(
        dse=cifar10, step_sizes=step_sizes_["cifar10"], eps=0.01, with_time=True,
        time_x_axis=True,
        plot_filename=None, zoom_in=True,
        zoom_params=ZoomInParams(0.25, 0.051, 0.3, 0.55, 2, 4)
    )
    print("CIFAR-100")
    plot_main_lines_with_info(
        dse=cifar100, step_sizes=step_sizes_["cifar100"], eps=0.01, with_time=True,
        time_x_axis=True, zoom_in=True,
        zoom_params=ZoomInParams(0.225, 0.18, 0.3, 0.55, 2, 4),
        plot_filename="cifar10-cifar100-strategies.png",
        plot_handles=p
    )
    print("MNIST")
    p2 = plot_main_lines_with_info(
        dse=mnist, step_sizes=step_sizes_["mnist"], eps=0.01, with_time=True,
        time_x_axis=True,
        plot_filename=None, zoom_in=True,
        zoom_params=ZoomInParams(0.15, 0.013, 0.3, 0.55, 3, 4)
    )
    print("SST2")
    plot_main_lines_with_info(
        dse=sst2, step_sizes=step_sizes_["sst2"], eps=0.01, with_time=True,
        time_x_axis=True, zoom_in=True,
        zoom_params=ZoomInParams(1.25, 0.1335, 0.3, 0.55, 3, 4),
        plot_filename="mnist-sst2-strategies.png",
        plot_handles=p2
    )
    print("IMDB")
    p3 = plot_main_lines_with_info(
        dse=imdb, step_sizes=step_sizes_["imdb_reviews"], eps=0.01, with_time=True,
        time_x_axis=True,
        plot_filename=None, zoom_in=True,
        zoom_params=ZoomInParams(0.7, 0.107, 0.3, 0.55, 2, 4)
    )
    print("YELP")
    plot_main_lines_with_info(
        dse=yelp, step_sizes=step_sizes_["yelp"], eps=0.01, with_time=True,
        time_x_axis=True, zoom_in=True,
        zoom_params=ZoomInParams(13, 0.2715, 0.3, 0.55, 3, 4),
        plot_filename="imdb-yelp-strategies.png",
        plot_handles=p3
    )

    # NN
    error_evaluation(dse=cifar10,
                     plot_filename="cifar10.png",
                     zoom_in=True,
                     zoom_params=ZoomInParams(0.0, 0.05, 0.55, 0.05, 2, 3),
                     ghp=False)

    error_evaluation(dse=cifar100,
                     plot_filename="cifar100.png",
                     zoom_in=True,
                     zoom_params=ZoomInParams(0.0, 0.19, 0.55, 0.05, 2, 3),
                     ghp=False)

    error_evaluation(dse=mnist,
                     plot_filename="mnist.png",
                     zoom_in=True,
                     zoom_params=ZoomInParams(0.0, 0.0, 0.55, 0.05, 2, 4),
                     ghp=False)

    error_evaluation(dse=imdb,
                     plot_filename="imdb.png",
                     zoom_in=True,
                     zoom_params=ZoomInParams(0.8, 0.42, 0.55, 0.05, 1, 2),
                     ghp=False)

    error_evaluation(dse=yelp,
                     plot_filename="yelp.png",
                     zoom_in=True,
                     zoom_params=ZoomInParams(0.0, 0.27, 0.55, 0.05, 2, 3),
                     ghp=False)

    error_evaluation(dse=sst2,
                     plot_filename="sst2.png",
                     zoom_in=True,
                     zoom_params=ZoomInParams(0.65, 0.37, 0.55, 0.05),
                     ghp=False)
