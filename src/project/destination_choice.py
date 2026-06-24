from __future__ import annotations

import math
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from mesa.discrete_space import Cell

    from .agents import SchellingAgent
    from .utility import LocationEvaluation


@dataclass(frozen=True)
class DestinationCandidate:
    """A satisfactory destination and its utility evaluation."""

    cell: Cell
    evaluation: LocationEvaluation
    value_improvement: float


@dataclass(frozen=True)
class DestinationSearchSummary:
    """Diagnostic information from one destination search."""

    visible_cells: int
    affordable_cells: int
    satisfactory_cells: int


class DestinationChoicePolicy(ABC):
    """Interface for interchangeable destination-selection rules."""

    @abstractmethod
    def choose_destination(
        self,
        agent: SchellingAgent,
    ) -> DestinationCandidate | None:
        """Choose a destination for an agent, or return None."""
        raise NotImplementedError


class RandomSatisficingChoice(DestinationChoicePolicy):
    """Choose uniformly among satisfactory empty destinations.

    A destination is satisfactory when:

    1. it lies within the household's fixed vision radius;
    2. it is empty;
    3. it satisfies the affordability condition;
    4. it provides the required proportional utility improvement.

    The required improvement is

        max(
            minimum_absolute_improvement,
            s * abs(current_utility)
        ),

    where `s` is the model-wide satisficing threshold.

    The policy chooses uniformly at random among all satisfactory
    destinations rather than choosing the destination with maximum utility.
    """

    def choose_destination(
        self,
        agent: SchellingAgent,
    ) -> DestinationCandidate | None:
        """Choose one random satisfactory destination."""
        satisfactory_candidates = (
            self.find_satisfactory_destinations(agent)
        )

        if not satisfactory_candidates:
            return None

        chosen_cell = agent.model.random.choice(
            satisfactory_candidates
        )

        evaluation, value_improvement = (
            agent.evaluate_destination(chosen_cell)
        )

        return DestinationCandidate(
            cell=chosen_cell,
            evaluation=evaluation,
            value_improvement=value_improvement,
        )

    def find_satisfactory_destinations(
        self,
        agent: SchellingAgent,
    ) -> list[Cell]:
        """Return all visible destinations satisfying the threshold."""
        candidate_cells = self._get_candidate_cells(agent)

        satisfactory_candidates: list[Cell] = []

        affordable_count = 0

        required_improvement = self._required_improvement(
            agent
        )

        aspiration_utility = (
            agent.current_value
            + required_improvement
        )

        utility_policy = agent.model.utility_policy

        for cell in candidate_cells:
            evaluation = (
                utility_policy.evaluate_for_search(
                    agent=agent,
                    cell=cell,
                )
            )

            if not evaluation.affordable:
                continue

            affordable_count += 1

            # Affordable locations should normally have finite utility.
            if not math.isfinite(evaluation.value):
                continue

            value_improvement = (
                evaluation.value
                - agent.current_value
            )

            # If the current residence is unaffordable, its utility may
            # equal -inf. In that case, every affordable destination with
            # finite utility is an improvement.
            if (
                math.isinf(agent.current_utility)
                and agent.current_utility < 0.0
            ):
                value_improvement = math.inf
            elif not math.isfinite(value_improvement):
                continue

            if evaluation.value < aspiration_utility:
                continue

            satisfactory_candidates.append(cell)

        agent.last_search_summary = (
            DestinationSearchSummary(
                visible_cells=len(candidate_cells),
                affordable_cells=affordable_count,
                satisfactory_cells=len(
                    satisfactory_candidates
                ),
            )
        )

        return satisfactory_candidates

    @staticmethod
    def _required_improvement(
        agent: SchellingAgent,
    ) -> float:
        """Return the minimum utility improvement required.

        This implements the multiplicative rule

            required improvement
            =
            s * abs(current utility),

        with a small absolute lower bound to ensure that a household with
        zero current utility still requires a genuine improvement.
        """
        current_utility = agent.current_utility
        current_value = agent.current_value

        if not math.isfinite(current_utility):
            # An agent at an infeasible current location should accept
            # any affordable destination with finite utility.
            return 0.0

        return max(
            agent.model.minimum_absolute_improvement,
            agent.model.satisficing_threshold
            * abs(current_value),
        )

    @staticmethod
    def _get_candidate_cells(
        agent: SchellingAgent,
    ) -> list[Cell]:
        """Return empty cells within the agent's fixed vision radius."""
        nearby_cells = agent.cell.get_neighborhood(
            radius=agent.vision_radius
        )

        return [
            cell
            for cell in nearby_cells
            if cell.is_empty
        ]
