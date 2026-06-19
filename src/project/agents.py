from __future__ import annotations

import math
from typing import TYPE_CHECKING

from mesa.discrete_space import CellAgent

if TYPE_CHECKING:
    from mesa.discrete_space import Cell

    from .destination_choice import (
        DestinationCandidate,
        DestinationSearchSummary,
    )
    from .utility import LocationEvaluation


class SchellingAgent(CellAgent):
    """Household agent in the income-segregation model.

    Each household has heterogeneous economic and behavioural attributes:

    - current and initial income;
    - income-similarity preference theta_i;
    - discount factor beta_i;
    - risk-aversion parameter rho_i;
    - rationality parameter lambda_i;
    - a fixed vision radius based on initial income.

    Every household searches for a satisfactory destination each period.
    Satisfaction is based on the number of consecutive periods in which
    the household has not moved.
    """

    def __init__(
        self,
        model,
        cell: Cell,
        income: float,
        income_similarity_preference: float,
        discount_factor: float,
        risk_aversion: float,
        rationality: float,
    ) -> None:
        """Create a household agent."""
        super().__init__(model)

        self._validate_parameters(
            income=income,
            income_similarity_preference=(
                income_similarity_preference
            ),
            discount_factor=discount_factor,
            risk_aversion=risk_aversion,
            rationality=rationality,
        )

        self.cell = cell

        # -----------------------------------------------------
        # Economic state
        # -----------------------------------------------------
        self.initial_income = income
        self.income = income

        # Vision is calculated once from initial income:
        #
        #     v_i = round(g * y_i(0) + 1)
        #
        # and remains fixed as income changes.
        raw_vision = (
            self.model.vision_income_scale
            * self.initial_income
            + 1.0
        )

        self.vision_radius = min(
            self.model.maximum_vision_radius,
            max(1, int(round(raw_vision))),
        )

        # -----------------------------------------------------
        # Behavioural parameters
        # -----------------------------------------------------
        self.income_similarity_preference = (
            income_similarity_preference
        )
        self.discount_factor = discount_factor
        self.risk_aversion = risk_aversion
        self.rationality = rationality

        # -----------------------------------------------------
        # Current-location utility state
        # -----------------------------------------------------
        self.current_utility = 0.0
        self.current_value = 0.0
        self.current_location_affordable = True

        # -----------------------------------------------------
        # Residence-duration satisfaction
        # -----------------------------------------------------
        self.steps_since_move = 0
        self.moved_this_step = False
        self.satisfied = False

        # -----------------------------------------------------
        # Search and movement diagnostics
        # -----------------------------------------------------
        self.last_search_summary: (
            DestinationSearchSummary | None
        ) = None

        self.last_move_successful: bool | None = None
        self.last_destination_coordinate: (
            tuple[int, ...] | None
        ) = None
        self.last_destination_utility: float | None = None
        self.last_value_improvement: float | None = None
        self.last_destination_value: float | None = None

    @staticmethod
    def _validate_parameters(
        *,
        income: float,
        income_similarity_preference: float,
        discount_factor: float,
        risk_aversion: float,
        rationality: float,
    ) -> None:
        """Validate household-specific parameters."""
        if income <= 0.0:
            raise ValueError(
                "income must be strictly positive."
            )

        if income_similarity_preference < 0.0:
            raise ValueError(
                "income_similarity_preference must be "
                "non-negative."
            )

        if not 0.0 <= discount_factor <= 1.0:
            raise ValueError(
                "discount_factor must lie between 0 and 1."
            )

        if not 0.0 <= risk_aversion < 5.0:
            raise ValueError(
                "risk_aversion must satisfy "
                "0 <= rho < 5."
            )

        if rationality < 0.0:
            raise ValueError(
                "rationality must be non-negative."
            )

    def evaluate_location(
        self,
        cell: Cell,
        *,
        is_current_location: bool = False,
    ) -> LocationEvaluation:
        """Evaluate a cell using the model's utility policy."""
        return self.model.utility_policy.evaluate(
            agent=self,
            cell=cell,
            is_current_location=is_current_location,
        )

    def assign_state(self) -> None:
        """Store utility and transformed value at the current cell."""
        evaluation = self.evaluate_location(
            self.cell,
            is_current_location=True,
        )

        self.current_utility = evaluation.utility
        self.current_value = evaluation.value
        self.current_location_affordable = (
            evaluation.affordable
        )

    def evaluate_destination(
        self,
        destination: Cell,
    ) -> tuple[LocationEvaluation, float]:
        """Evaluate a destination relative to the current value."""
        evaluation = self.evaluate_location(
            destination,
            is_current_location=False,
        )

        value_improvement = (
            evaluation.value
            - self.current_value
        )

        return evaluation, value_improvement

    def change_income(self) -> None:
        """Update income using neighbourhood-dependent GBM.

        The drift is

            eta * max(
                0,
                (mean_y_j - y_i) / mean_y_j
            ).

        The discrete update is

            y_i(t+1)
            =
            y_i(t) exp(
                drift
                - sigma^2 / 2
                + sigma epsilon_i
            ).
        """
        coordinate = self.cell.coordinate

        local_mean_income = float(
            self.model.grid.mean_neighbor_income.data[
                coordinate
            ]
        )

        if local_mean_income > 0.0:
            spillover = max(
                0.0,
                (
                    local_mean_income
                    - self.income
                )
                / local_mean_income,
            )
        else:
            spillover = 0.0

        drift = (
            self.model.income_growth_scaling
            * spillover
        )

        volatility = self.model.income_volatility

        shock = float(
            self.model.rng.normal(
                loc=0.0,
                scale=1.0,
            )
        )

        growth_factor = math.exp(
            drift
            - 0.5 * volatility**2
            + volatility * shock
        )

        self.income *= growth_factor

    def attempt_move(self) -> bool:
        """Search for and move to a satisfactory destination."""
        self.model.move_attempts += 1

        candidate: DestinationCandidate | None = (
            self.model.destination_choice_policy
            .choose_destination(self)
        )

        if candidate is None:
            self.model.failed_searches += 1

            self.last_move_successful = False
            self.last_destination_coordinate = None
            self.last_destination_utility = None
            self.last_utility_improvement = None
            self.last_destination_value = None
            self.last_value_improvement = None

            return False

        self.last_destination_coordinate = (
            candidate.cell.coordinate
        )
        self.last_destination_utility = (
            candidate.evaluation.utility
        )
        self.last_destination_utility = (
            candidate.evaluation.utility
        )

        self.last_destination_value = (
            candidate.evaluation.value
        )

        self.last_value_improvement = (
            candidate.value_improvement
        )

        self.move_to(candidate.cell)

        self.moved_this_step = True
        self.last_move_successful = True
        self.model.successful_moves += 1

        return True

    def update_satisfaction(self) -> None:
        """Update satisfaction from consecutive non-moving periods."""
        if self.moved_this_step:
            self.steps_since_move = 0
        else:
            self.steps_since_move += 1

        self.satisfied = (
            self.steps_since_move
            >= self.model.steps_until_satisfied
        )

        if self.satisfied:
            self.model.satisfied_count += 1

    def step(self) -> None:
        """Search for a satisfactory destination every period."""
        self.moved_this_step = False
        self.attempt_move()
