from typing import OrderedDict

from snoopy.nn import Arm
from snoopy.result import Observer
from .base import _notify_initial_error
from .._logging import get_logger

_logger = get_logger(__name__)


class BestFirstAlgorithm:
    def __init__(self, arms: OrderedDict[str, Arm], observer: Observer):
        _notify_initial_error(arms, observer)

        self._arms = arms
        self._observer = observer

    def best_first(self):
        num_pulls_per_arm = {arm_name: 0 for arm_name in self._arms}
        slope_per_arm = {arm_name: 0 for arm_name in self._arms}
        num_errors_per_arm = {arm_name: self._arms[arm_name].initial_error.num_errors for arm_name in self._arms}

        # Run two rounds to assess initial slope
        for pull_round in range(2):
            for arm_name in self._arms:
                arm = self._arms[arm_name]
                assert arm.can_progress(), "Arm should be able to be pulled at least twice!"

                # Pull arm and compute slope
                result = arm.progress()
                num_pulls_per_arm[arm_name] += 1
                slope_per_arm[arm_name] = num_errors_per_arm[arm_name] - result.num_errors
                num_errors_per_arm[arm_name] = result.num_errors

                # Update observer immediately after the pull
                self._observer.on_update(arm_name, result)

        pullable = list(self._arms.keys())
        while len(pullable) > 0:
            # Sorted by number of pulls decreasing, break ties by the number of errors
            sorted_pullable = sorted(pullable, key=lambda name: (-num_pulls_per_arm[name], num_errors_per_arm[name]))

            for i in range(len(sorted_pullable)):
                # _logger.debug(f"Trying: {sorted_pullable[i]}")

                is_i_suitable = True
                name_i = sorted_pullable[i]
                error_i = num_errors_per_arm[name_i]

                for j in range(i + 1, len(sorted_pullable)):
                    name_j = sorted_pullable[j]

                    # Check if tangent of competitor is below candidate
                    errors_j = num_errors_per_arm[name_j]
                    slope_j = slope_per_arm[name_j]

                    # Ensure that slope is not negative
                    if slope_j < 0:
                        slope_j = 0

                    pull_diff = num_pulls_per_arm[name_i] - num_pulls_per_arm[name_j]
                    lower_bound_j = errors_j - slope_j * pull_diff

                    # _logger.debug(f"\tComparing with: {name_i}, Can: {error_i}, Com: {lower_bound_j}")

                    if lower_bound_j < error_i:
                        is_i_suitable = False
                        break

                # No tangent is below candidate -> Pull arm
                if is_i_suitable:
                    # _logger.debug(f"\tCandidate {name_i} is suitable")
                    candidate_arm = self._arms[name_i]
                    result = candidate_arm.progress()

                    # Update values
                    num_pulls_per_arm[name_i] += 1
                    slope_per_arm[name_i] = num_errors_per_arm[name_i] - result.num_errors
                    num_errors_per_arm[name_i] = result.num_errors

                    # Notify observer
                    self._observer.on_update(name_i, result)

                    # Start from the beginning
                    break

            # Remove arms that cannot be pulled anymore
            pullable = [name for name in pullable if self._arms[name].can_progress()]
