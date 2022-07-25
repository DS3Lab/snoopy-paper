from collections import OrderedDict as OrdD, defaultdict as dd
from dataclasses import dataclass
from math import ceil, floor, log
from typing import List, OrderedDict

from .base import Strategy, StrategyConfig, _get_arm, _notify_initial_error
from .._logging import get_logger
from ..embedding import EmbeddingDatasetsTuple
from ..nn import Arm, KNNAlgorithmType, knn_algorithm_factory
from ..result import Observer

_logger = get_logger(__name__)


@dataclass(frozen=True)
class SuccessiveHalvingStrategyConfig(StrategyConfig):
    train_size: int
    test_size: int
    reduce_factor: int
    snoopy: bool


class SuccessiveHalvingAlgorithm:
    def __init__(self, arms: OrderedDict[str, Arm], observer: Observer):
        _notify_initial_error(arms, observer)

        self._arms = arms
        self._observer = observer

        # Number of times an armed has been pulled for each arm
        self._pulls_performed = {arm_name: 0 for arm_name in self._arms}

        # Each arms has a mapping: num pulls -> error, for fast access
        # -1 is returned when certain number of pulls cannot be reached by that arm
        self._partial_results = {arm_name: dd(lambda: -1) for arm_name in self._arms}
        for arm_name in arms:
            self._partial_results[arm_name][0] = arms[arm_name].initial_error.num_errors

    def _successive_halving(self, budget: int, eta: int, snoopy: bool, arm_names: List[str] = None) -> None:
        # _logger.debug("(Inner) Running successive halving")
        # Names of remaining arms
        if arm_names is None:
            arm_names = list(self._arms.keys())

        # Number of arms remaining / competing
        abs_big_s_k = len(arm_names)  # = n

        # Budget
        big_b = budget

        # Cumulative sum of arm pulls over iterations
        big_r_k = 0

        # Arms eliminated by tangent approach if 'snoopy' is used
        non_promising_arms = set()

        # One iteration of successive halving
        # Last iteration will have k = ceil(log_eta(n))
        log_value = max(1, ceil(log(abs_big_s_k, eta)))

        for k in range(log_value):
            # Number of pulls in current iteration
            r_k = floor(big_b / (abs_big_s_k * log_value))
            if r_k == 0:
                break

            # _logger.debug(f"(Inner) Iteration: {k}, target num. pulls: {big_r_k + r_k}, "
            #               f"{', '.join(arm_names)} are left")
            num_removed_in_current_iteration = 0

            # Iterate through arms that are still competing
            for arm_name in arm_names:
                current_arm = self._arms[arm_name]

                # Get lowest error of any other arm
                if len(arm_names) > 1:
                    lowest_errors = [self._partial_results[i][min(big_r_k + r_k, self._pulls_performed[i])]
                                     for i in arm_names if i != arm_name]

                    error_to_beat = sorted(lowest_errors)[(abs_big_s_k // eta) - 1]
                else:
                    error_to_beat = None

                # Try to pull it r_k times
                for pull_index in range(r_k):
                    # Cumulative number of pulls required by current iteration sub step
                    target_num_pulls = big_r_k + pull_index + 1

                    # Only try to pull if arm is 'behind' the number of pulls required
                    pulls_performed = self._pulls_performed[arm_name]
                    if pulls_performed < target_num_pulls:
                        # Determine whether arm is 'not promising' = its tangent lower bound is worse than what was
                        # already achieved by some other arm
                        if snoopy and pulls_performed >= 2 and len(arm_names) > 1:
                            last_error = self._partial_results[arm_name][pulls_performed]
                            slope = self._partial_results[arm_name][pulls_performed - 1] - last_error

                            # Ensure that tangent is not increasing
                            if slope < 0:
                                slope = 0

                            pull_difference = big_r_k + r_k - pulls_performed
                            lower_bound = last_error - slope * pull_difference

                            if lower_bound > error_to_beat and \
                                    len(arm_names) - num_removed_in_current_iteration > abs_big_s_k // eta:
                                non_promising_arms.add(arm_name)
                                num_removed_in_current_iteration += 1
                                break

                        # Arm can be pulled -> pull it
                        if current_arm.can_progress():
                            # Pull arm
                            current_progress = current_arm.progress()
                            # _logger.debug(f"(Inner) Pulled arm: {arm_name}")

                            # Notify observer of progress
                            self._observer.on_update(arm_name, current_progress)

                            # Store partial result for potential reuse
                            self._pulls_performed[arm_name] = target_num_pulls
                            self._partial_results[arm_name][target_num_pulls] = current_progress.num_errors

                        # Arm cannot be pulled
                        else:
                            break

            # Update cumulative sum of arm pulls
            big_r_k += r_k

            # Determine indices of arms that are still competing after current iteration
            # Those are the arms that are among (1 / eta) * remaining arms with smallest losses
            arms_and_losses = [(arm_name, self._partial_results[arm_name][big_r_k]) for arm_name in arm_names if
                               arm_name not in non_promising_arms]

            # End if one of the arms could not execute all pulls
            if -1 in map(lambda x: x[1], arms_and_losses):
                break

            sorted_arms_and_losses = sorted(arms_and_losses, key=lambda x: x[1])
            arms_and_losses_to_retain = sorted_arms_and_losses[:(abs_big_s_k // eta)]
            arm_names = list(map(lambda x: x[0], arms_and_losses_to_retain))
            abs_big_s_k //= eta

            # Do not execute all iterations in case that next iteration would not return retain any arm
            # This is needed when number of arms is not a power of eta
            if len(arm_names) == 0:
                break

    def successive_halving_with_doubling(self, eta: int, snoopy: bool):
        # Set B <- n
        initial_budget = len(self._arms)

        # Arms that have not reached the end yet
        arm_names_for_next_run = list(self._arms.keys())

        # Budget to start each iteration with
        current_budget = initial_budget

        # While there exist arms that have not reached the end yet
        while len(arm_names_for_next_run) > 0:
            # Run successive halving
            # _logger.debug(f"(Outer) Running with budget: {current_budget}")
            # _logger.debug(f"(Outer) Remaining arms: {', '.join(arm_names_for_next_run)}")
            self._successive_halving(
                budget=current_budget,
                eta=eta,
                snoopy=snoopy,
                arm_names=arm_names_for_next_run
            )

            # 'Doubling trick'
            current_budget *= 2

            # If one of the arms reached the end in the last run, do not use it anymore and reset budget
            arm_names_to_remove = []
            for arm_name in arm_names_for_next_run:
                if not self._arms[arm_name].can_progress():
                    # _logger.debug(f"(Outer) Removing arm: {arm_name}")
                    arm_names_to_remove.append(arm_name)

            if len(arm_names_to_remove) > 0:
                arm_names_for_next_run = [i for i in arm_names_for_next_run if i not in arm_names_to_remove]
                current_budget = initial_budget


class SuccessiveHalvingStrategy(Strategy):

    def __init__(self, config: SuccessiveHalvingStrategyConfig):
        self._train_size = config.train_size
        self._test_size = config.test_size
        self._knn = knn_algorithm_factory(KNNAlgorithmType.TOP_K)
        self._reduce_factor = config.reduce_factor
        self._snoopy = config.snoopy

    def execute(self, datasets: OrderedDict[str, EmbeddingDatasetsTuple], observer: Observer) -> None:
        # Prepare 'arms' that can be 'pulled'
        arms = OrdD()
        for dataset_name in datasets:
            # _logger.debug(f"Preparing test dataset for embedding '{dataset_name}'")
            arms[dataset_name] = _get_arm(datasets[dataset_name], self._train_size, self._test_size, self._knn)
            # _logger.debug(f"Test dataset for embedding '{dataset_name}' is now prepared")

        sha = SuccessiveHalvingAlgorithm(arms, observer)

        sha.successive_halving_with_doubling(
            eta=self._reduce_factor,
            snoopy=self._snoopy
        )
