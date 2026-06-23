from __future__ import annotations

import math
from typing import TYPE_CHECKING

import numpy as np
from mesa.discrete_space import CellAgent

COLORS = [
    "#8B0000",  # darkest red
    "#D73027",
    "#FC8D59",
    "#FEE08B",
    "#D9EF8B",
    "#91CF60",
    "#1A9850",  # darkest green
]

if TYPE_CHECKING:
    from mesa.discrete_space import Cell

    from .game import (
        ApplicationGameRecord,
    )
    from .destination_choice import (
        DestinationCandidate,
        DestinationSearchSummary,
    )
    from .utility import LocationEvaluation


class SchellingAgent(CellAgent):
    """Household agent in the gentrification model.
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

        self.initial_income = income
        self.income = income

        raw_vision = (
            self.model.vision_income_scale
            * self.initial_income
            + 1.0
        )

        self.vision_radius = min(
            self.model.maximum_vision_radius,
            max(1, int(round(raw_vision))),
        )

        self.income_similarity_preference = (
            income_similarity_preference
        )
        self.discount_factor = discount_factor
        self.risk_aversion = risk_aversion
        self.rationality = rationality

        self.income_percentile = 0.0

        self.colour = "blue"

        self.current_utility = 0.0
        self.current_value = 0.0
        self.current_location_affordable = True

        self.steps_since_move = 0
        self.moved_this_step = False
        self.satisfied = False

        self.last_search_summary: (
            DestinationSearchSummary | None
        ) = None

        self.last_move_successful: bool | None = None
        self.last_destination_coordinate: (
            tuple[int, ...] | None
        ) = None
        self.last_destination_utility: float | None = None
        self.last_destination_value: float | None = None
        self.last_value_improvement: float | None = None

        self.last_game_record: (
            ApplicationGameRecord | None
        ) = None

        self.last_qre_move_probability: float | None = None
        self.last_qre_accept_probability: float | None = None
        self.last_ne_move_probability: float | None = None
        self.last_qre_ne_move_gap: float | None = None

        self.pending_destination: (
            DestinationCandidate | None
        ) = None

    @staticmethod
    def _validate_parameters(
        *,
        income: float,
        income_similarity_preference: float,
        discount_factor: float,
        risk_aversion: float,
        rationality: float,
    ) -> None:
        if income <= 0.0:
            raise ValueError(
                "income must be strictly positive."
            )

        if income_similarity_preference < 0.0:
            raise ValueError(
                "income_similarity_preference must be non-negative."
            )

        if not 0.0 <= discount_factor <= 1.0:
            raise ValueError(
                "discount_factor must lie between 0 and 1."
            )

        # if not 0.0 <= risk_aversion < 5.0:
        #     raise ValueError(
        #         "risk_aversion must satisfy 0 <= rho < 5."
        #     )

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
        return self.model.utility_policy.evaluate(
            agent=self,
            cell=cell,
            is_current_location=is_current_location,
        )

    def assign_state(self) -> None:
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
            #TODO: don't forget this change
            spillover = (local_mean_income
                    - self.income)/ local_mean_income
        else:
            spillover = 0.0

        drift = (
            self.model.income_growth_scaling
            * spillover + 0.1
        )

        #TODO: make volatility a function of local income variance
        volatility = np.sqrt(self.model.grid.neighbor_income_variance.data[coordinate])

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

    def prepare_move_decision(self) -> None:
        """Choose a candidate and play the game without moving yet."""
        self.moved_this_step = False
        self.pending_destination = None

        self.model.move_attempts += 1

        candidate = (
            self.model.destination_choice_policy
            .choose_destination(self)
        )

        if candidate is None:
            self.model.failed_searches += 1

            self.last_move_successful = False
            self.last_destination_coordinate = None
            self.last_destination_utility = None
            self.last_destination_value = None
            self.last_value_improvement = None

            self.last_game_record = None
            self.last_qre_move_probability = None
            self.last_qre_accept_probability = None
            self.last_ne_move_probability = None
            self.last_qre_ne_move_gap = None
            return

        game_record = (
            self.model.application_game_policy.play(
                agent=self,
                candidate=candidate,
            )
        )

        self.model.game_records_this_step.append(
            game_record
        )

        if self.model.keep_game_history:
            self.model.game_history.append(
                game_record
            )

        self.last_game_record = game_record

        self.last_qre_move_probability = (
            game_record.qre.p_move
        )
        self.last_qre_accept_probability = (
            game_record.qre.p_accept
        )
        self.last_ne_move_probability = (
            game_record.nash.closest_move_probability
        )
        self.last_qre_ne_move_gap = (
            game_record.nash.move_probability_gap
        )

        if not game_record.qre.converged:
            self.model.qre_nonconvergence_count += 1

        outcome = (
            game_record.realized_game_outcome
        )

        if outcome == "move_accept":
            self.model.move_accept_outcomes += 1
        elif outcome == "move_reject":
            self.model.move_reject_outcomes += 1
        elif outcome == "stay_accept":
            self.model.stay_accept_outcomes += 1
        else:
            self.model.stay_reject_outcomes += 1

        self.last_destination_coordinate = (
            candidate.cell.coordinate
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

        if not game_record.household_chose_move:
            self.model.voluntary_stays += 1
            self.last_move_successful = False
            return

        self.pending_destination = candidate

    def execute_pending_move(self) -> None:
        """Execute a previously prepared move after conflict resolution."""
        if self.pending_destination is None:
            return

        self.move_to(
            self.pending_destination.cell
        )

        self.moved_this_step = True
        self.last_move_successful = True
        self.model.successful_moves += 1

        self.pending_destination = None

    def cancel_pending_move_due_to_conflict(
        self,
    ) -> None:
        """Cancel a selected move because another agent won the vacancy."""
        if self.pending_destination is None:
            return

        self.pending_destination = None
        self.moved_this_step = False
        self.last_move_successful = False
        self.model.destination_conflicts += 1

    def update_satisfaction(self) -> None:
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


    def update_visuals(self):

        q10, q25, q50, q75, q90, q99 = (
            self.model.income_thresholds
        )

        income = self.income

        if income >= q99:
            self.colour = COLORS[5]
        elif income >= q90:
            self.colour = COLORS[4]
        elif income >= q75:
            self.colour = COLORS[3]
        elif income >= q50:
            self.colour = COLORS[2]
        elif income >= q25:
            self.colour = COLORS[1]
        else:
            self.colour = COLORS[0]
