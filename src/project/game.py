from __future__ import annotations

import math
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import TYPE_CHECKING

import numpy as np

from .nash import (
    NashAnalysis,
    TwoByTwoNashSolver,
)

if TYPE_CHECKING:
    from .agents import SchellingAgent
    from .destination_choice import DestinationCandidate


@dataclass(frozen=True)
class NeighborhoodUtilityChange:
    """Raw neighbourhood utility before and after household entry."""

    current_utility: float
    updated_utility: float
    delta_utility: float

    resident_count: int
    applicant_utility: float


@dataclass(frozen=True)
class GamePayoffs:
    """Raw utility and transformed value matrices.

    Rows:
        0 = MOVE
        1 = STAY

    Columns:
        0 = ACCEPT
        1 = REJECT
    """

    household_utilities: np.ndarray
    neighborhood_utilities: np.ndarray

    household_values: np.ndarray
    neighborhood_values: np.ndarray

    delta_agent_utility: float
    delta_agent_value: float

    neighborhood_change: NeighborhoodUtilityChange


@dataclass(frozen=True)
class LogitQRESolution:
    """One fixed point of the simultaneous logit responses."""

    p_move: float
    p_accept: float

    p_move_accept: float
    p_move_reject: float
    p_stay_accept: float
    p_stay_reject: float

    converged: bool
    iterations: int
    residual: float


@dataclass(frozen=True)
class ApplicationGameRecord:
    """Complete record of one household-destination game."""

    step_number: int
    agent_id: int
    candidate_coordinate: tuple[int, ...]

    payoffs: GamePayoffs
    qre: LogitQRESolution
    nash: NashAnalysis

    household_chose_move: bool
    neighborhood_chose_accept: bool

    realized_game_outcome: str
    realized_household_value: float
    realized_neighborhood_value: float


class NeighborhoodUtilityPolicy(ABC):
    """Interface for calculating raw neighbourhood utility."""

    @abstractmethod
    def evaluate_change(
        self,
        *,
        agent: SchellingAgent,
        candidate: DestinationCandidate,
    ) -> NeighborhoodUtilityChange:
        raise NotImplementedError


class MeanResidentUtilityPolicy(
    NeighborhoodUtilityPolicy
):
    """Use the average raw utility of nearby residents.

    Current neighbourhood utility:

        U_n = mean(U_k for k in N_j).

    Updated neighbourhood utility:

        U_n,new
        =
        mean(U_k for k in N_j, U_ij).

    Existing residents contribute their currently stored raw utilities.
    The applicant contributes its raw candidate-location utility.
    """

    def __init__(
        self,
        *,
        empty_neighborhood_utility: float = 0.0,
    ) -> None:
        if not math.isfinite(
            empty_neighborhood_utility
        ):
            raise ValueError(
                "empty_neighborhood_utility must be finite."
            )

        self.empty_neighborhood_utility = (
            empty_neighborhood_utility
        )

    def evaluate_change(
        self,
        *,
        agent: SchellingAgent,
        candidate: DestinationCandidate,
    ) -> NeighborhoodUtilityChange:
        neighbors = (
            agent.model.neighborhood_definition
            .get_neighbors(candidate.cell)
        )

        residents = [
            resident
            for resident in neighbors
            if resident is not agent
        ]

        resident_utilities = [
            float(resident.current_utility)
            for resident in residents
            if math.isfinite(
                float(resident.current_utility)
            )
        ]

        if resident_utilities:
            current_utility = float(
                np.mean(resident_utilities)
            )
        else:
            current_utility = (
                self.empty_neighborhood_utility
            )

        applicant_utility = float(
            candidate.evaluation.utility
        )

        updated_utility = float(
            np.mean(
                [
                    *resident_utilities,
                    applicant_utility,
                ]
            )
        )

        return NeighborhoodUtilityChange(
            current_utility=current_utility,
            updated_utility=updated_utility,
            delta_utility=(
                updated_utility
                - current_utility
            ),
            resident_count=len(
                resident_utilities
            ),
            applicant_utility=applicant_utility,
        )


class PayoffStructure(ABC):
    """Replaceable rule for constructing game payoffs."""

    @abstractmethod
    def build(
        self,
        *,
        agent: SchellingAgent,
        candidate: DestinationCandidate,
        neighborhood_change: NeighborhoodUtilityChange,
        moving_cost: float,
        rejection_cost: float,
        neighborhood_risk_aversion: float,
    ) -> GamePayoffs:
        raise NotImplementedError


class TransferPayoffStructure(PayoffStructure):
    """Implement the currently specified payoff matrix.

    Raw utility matrix:

                         ACCEPT                       REJECT

    MOVE    (U_a+dU_a-c_m,              (U_a+dU_a-c_m-c_r,
             U_n+dU_n+c_m)               U_n+dU_n+c_m+c_r)

    STAY    (U_a, U_n)                   (U_a-c_r, U_n+c_r)

    Both players' complete raw outcome utilities are transformed only
    after c_m and c_r have been added or subtracted.
    """

    def build(
        self,
        *,
        agent: SchellingAgent,
        candidate: DestinationCandidate,
        neighborhood_change: NeighborhoodUtilityChange,
        moving_cost: float,
        rejection_cost: float,
        neighborhood_risk_aversion: float,
    ) -> GamePayoffs:
        current_agent_utility = float(
            agent.current_utility
        )

        destination_utility = float(
            candidate.evaluation.utility
        )

        current_neighborhood_utility = float(
            neighborhood_change.current_utility
        )

        updated_neighborhood_utility = float(
            neighborhood_change.updated_utility
        )

        household_utilities = np.array(
            [
                [
                    destination_utility
                    - moving_cost,

                    destination_utility
                    - moving_cost
                    - rejection_cost,
                ],
                [
                    current_agent_utility,

                    current_agent_utility
                    - rejection_cost,
                ],
            ],
            dtype=float,
        )

        neighborhood_utilities = np.array(
            [
                [
                    updated_neighborhood_utility
                    + moving_cost,

                    updated_neighborhood_utility
                    + moving_cost
                    + rejection_cost,
                ],
                [
                    current_neighborhood_utility,

                    current_neighborhood_utility
                    + rejection_cost,
                ],
            ],
            dtype=float,
        )

        household_values = self._transform_matrix(
            utilities=household_utilities,
            risk_aversion=agent.risk_aversion,
            transform=(
                agent.model.utility_policy
                .apply_signed_crra
            ),
        )

        neighborhood_values = self._transform_matrix(
            utilities=neighborhood_utilities,
            risk_aversion=(
                neighborhood_risk_aversion
            ),
            transform=(
                agent.model.utility_policy
                .apply_signed_crra
            ),
        )

        return GamePayoffs(
            household_utilities=(
                household_utilities
            ),
            neighborhood_utilities=(
                neighborhood_utilities
            ),
            household_values=household_values,
            neighborhood_values=(
                neighborhood_values
            ),
            delta_agent_utility=(
                destination_utility
                - current_agent_utility
            ),
            delta_agent_value=(
                candidate.evaluation.value
                - agent.current_value
            ),
            neighborhood_change=(
                neighborhood_change
            ),
        )

    @staticmethod
    def _transform_matrix(
        *,
        utilities: np.ndarray,
        risk_aversion: float,
        transform,
    ) -> np.ndarray:
        values = np.empty_like(
            utilities,
            dtype=float,
        )

        for row in range(2):
            for column in range(2):
                values[row, column] = transform(
                    utility=float(
                        utilities[row, column]
                    ),
                    risk_aversion=risk_aversion,
                )

        return values


class LogitQRESolver:
    """Solve simultaneous binary logit responses by fixed-point iteration."""

    def __init__(
        self,
        *,
        tolerance: float = 1e-10,
        maximum_iterations: int = 1000,
        damping: float = 0.5,
    ) -> None:
        if tolerance <= 0.0:
            raise ValueError(
                "tolerance must be strictly positive."
            )

        if maximum_iterations < 1:
            raise ValueError(
                "maximum_iterations must be positive."
            )

        if not 0.0 < damping <= 1.0:
            raise ValueError(
                "damping must satisfy 0 < damping <= 1."
            )

        self.tolerance = tolerance
        self.maximum_iterations = (
            maximum_iterations
        )
        self.damping = damping

    def solve(
        self,
        *,
        household_values: np.ndarray,
        neighborhood_values: np.ndarray,
        household_rationality: float,
        neighborhood_rationality: float,
    ) -> LogitQRESolution:
        p_move = 0.5
        p_accept = 0.5

        converged = False
        residual = math.inf
        iteration = 0

        for iteration in range(
            1,
            self.maximum_iterations + 1,
        ):
            move_response = self._household_response(
                household_values=household_values,
                p_accept=p_accept,
                rationality=household_rationality,
            )

            accept_response = (
                self._neighborhood_response(
                    neighborhood_values=(
                        neighborhood_values
                    ),
                    p_move=p_move,
                    rationality=(
                        neighborhood_rationality
                    ),
                )
            )

            next_p_move = (
                (1.0 - self.damping) * p_move
                + self.damping * move_response
            )

            next_p_accept = (
                (1.0 - self.damping) * p_accept
                + self.damping * accept_response
            )

            residual = max(
                abs(next_p_move - p_move),
                abs(next_p_accept - p_accept),
            )

            p_move = next_p_move
            p_accept = next_p_accept

            if residual <= self.tolerance:
                converged = True
                break

        return LogitQRESolution(
            p_move=float(p_move),
            p_accept=float(p_accept),
            p_move_accept=float(
                p_move * p_accept
            ),
            p_move_reject=float(
                p_move * (1.0 - p_accept)
            ),
            p_stay_accept=float(
                (1.0 - p_move) * p_accept
            ),
            p_stay_reject=float(
                (1.0 - p_move)
                * (1.0 - p_accept)
            ),
            converged=converged,
            iterations=iteration,
            residual=float(residual),
        )

    @classmethod
    def _household_response(
        cls,
        *,
        household_values: np.ndarray,
        p_accept: float,
        rationality: float,
    ) -> float:
        expected_move = (
            p_accept * household_values[0, 0]
            + (1.0 - p_accept)
            * household_values[0, 1]
        )

        expected_stay = (
            p_accept * household_values[1, 0]
            + (1.0 - p_accept)
            * household_values[1, 1]
        )

        return cls._binary_logit(
            first_value=expected_move,
            second_value=expected_stay,
            rationality=rationality,
        )

    @classmethod
    def _neighborhood_response(
        cls,
        *,
        neighborhood_values: np.ndarray,
        p_move: float,
        rationality: float,
    ) -> float:
        expected_accept = (
            p_move * neighborhood_values[0, 0]
            + (1.0 - p_move)
            * neighborhood_values[1, 0]
        )

        expected_reject = (
            p_move * neighborhood_values[0, 1]
            + (1.0 - p_move)
            * neighborhood_values[1, 1]
        )

        return cls._binary_logit(
            first_value=expected_accept,
            second_value=expected_reject,
            rationality=rationality,
        )

    @staticmethod
    def _binary_logit(
        *,
        first_value: float,
        second_value: float,
        rationality: float,
    ) -> float:
        if rationality < 0.0:
            raise ValueError(
                "rationality must be non-negative."
            )

        difference = rationality * (
            first_value - second_value
        )

        difference = max(
            min(difference, 700.0),
            -700.0,
        )

        return 1.0 / (
            1.0 + math.exp(-difference)
        )


class ApplicationGamePolicy:
    """Build, solve, sample, and record one application game."""

    def __init__(
        self,
        *,
        neighborhood_utility_policy: (
            NeighborhoodUtilityPolicy
        ),
        payoff_structure: PayoffStructure,
        qre_solver: LogitQRESolver,
        nash_solver: TwoByTwoNashSolver,
    ) -> None:
        self.neighborhood_utility_policy = (
            neighborhood_utility_policy
        )
        self.payoff_structure = payoff_structure
        self.qre_solver = qre_solver
        self.nash_solver = nash_solver

    def play(
        self,
        *,
        agent: SchellingAgent,
        candidate: DestinationCandidate,
    ) -> ApplicationGameRecord:
        neighborhood_change = (
            self.neighborhood_utility_policy
            .evaluate_change(
                agent=agent,
                candidate=candidate,
            )
        )

        payoffs = self.payoff_structure.build(
            agent=agent,
            candidate=candidate,
            neighborhood_change=(
                neighborhood_change
            ),
            moving_cost=agent.model.moving_cost,
            rejection_cost=(
                agent.model.rejection_cost
            ),
            neighborhood_risk_aversion=(
                agent.model
                .neighborhood_risk_aversion
            ),
        )

        qre = self.qre_solver.solve(
            household_values=(
                payoffs.household_values
            ),
            neighborhood_values=(
                payoffs.neighborhood_values
            ),
            household_rationality=(
                agent.rationality
            ),
            neighborhood_rationality=(
                agent.model
                .neighborhood_rationality
            ),
        )

        nash = self.nash_solver.analyze(
            household_values=(
                payoffs.household_values
            ),
            neighborhood_values=(
                payoffs.neighborhood_values
            ),
            comparison_move_probability=(
                qre.p_move
            ),
        )

        household_chose_move = (
            agent.model.random.random()
            < qre.p_move
        )

        neighborhood_chose_accept = (
            agent.model.random.random()
            < qre.p_accept
        )

        row = (
            0 if household_chose_move else 1
        )

        column = (
            0 if neighborhood_chose_accept else 1
        )

        outcome_names = {
            (0, 0): "move_accept",
            (0, 1): "move_reject",
            (1, 0): "stay_accept",
            (1, 1): "stay_reject",
        }

        return ApplicationGameRecord(
            step_number=int(
                getattr(agent.model, "steps", 0)
            ),
            agent_id=agent.unique_id,
            candidate_coordinate=(
                candidate.cell.coordinate
            ),
            payoffs=payoffs,
            qre=qre,
            nash=nash,
            household_chose_move=(
                household_chose_move
            ),
            neighborhood_chose_accept=(
                neighborhood_chose_accept
            ),
            realized_game_outcome=(
                outcome_names[row, column]
            ),
            realized_household_value=float(
                payoffs.household_values[
                    row,
                    column,
                ]
            ),
            realized_neighborhood_value=float(
                payoffs.neighborhood_values[
                    row,
                    column,
                ]
            ),
        )
