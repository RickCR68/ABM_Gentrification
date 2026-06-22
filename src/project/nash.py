from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class NashEquilibrium:
    """One Nash equilibrium of a 2x2 game."""

    kind: str

    p_move: float
    p_accept: float

    household_strategy: tuple[float, float]
    neighborhood_strategy: tuple[float, float]

    household_expected_value: float
    neighborhood_expected_value: float

    pure_action_profile: tuple[int, int] | None = None


@dataclass(frozen=True)
class NashAnalysis:
    """All enumerated Nash-equilibrium information for one game."""

    equilibria: tuple[NashEquilibrium, ...]

    move_probabilities: tuple[float, ...]
    accept_probabilities: tuple[float, ...]

    minimum_move_probability: float
    maximum_move_probability: float

    closest_move_probability: float
    move_probability_gap: float

    @property
    def number_of_equilibria(self) -> int:
        return len(self.equilibria)


class TwoByTwoNashSolver:
    """Enumerate pure equilibria and a strictly interior mixed equilibrium.

    Rows:
        0 = MOVE
        1 = STAY

    Columns:
        0 = ACCEPT
        1 = REJECT
    """

    def __init__(
        self,
        *,
        tolerance: float = 1e-10,
    ) -> None:
        if tolerance <= 0.0:
            raise ValueError(
                "tolerance must be strictly positive."
            )

        self.tolerance = tolerance

    def analyze(
        self,
        *,
        household_values: np.ndarray,
        neighborhood_values: np.ndarray,
        comparison_move_probability: float,
    ) -> NashAnalysis:
        household_values = self._validate_matrix(
            household_values,
            name="household_values",
        )

        neighborhood_values = self._validate_matrix(
            neighborhood_values,
            name="neighborhood_values",
        )

        equilibria: list[NashEquilibrium] = []

        equilibria.extend(
            self._find_pure_equilibria(
                household_values=household_values,
                neighborhood_values=(
                    neighborhood_values
                ),
            )
        )

        mixed = self._find_interior_mixed_equilibrium(
            household_values=household_values,
            neighborhood_values=neighborhood_values,
        )

        if mixed is not None:
            equilibria.append(mixed)

        equilibria = self._deduplicate_equilibria(
            equilibria
        )

        move_probabilities = tuple(
            equilibrium.p_move
            for equilibrium in equilibria
        )

        accept_probabilities = tuple(
            equilibrium.p_accept
            for equilibrium in equilibria
        )

        if move_probabilities:
            closest_move_probability = min(
                move_probabilities,
                key=lambda probability: abs(
                    probability
                    - comparison_move_probability
                ),
            )

            move_probability_gap = abs(
                comparison_move_probability
                - closest_move_probability
            )

            minimum_move_probability = min(
                move_probabilities
            )

            maximum_move_probability = max(
                move_probabilities
            )
        else:
            closest_move_probability = float("nan")
            move_probability_gap = float("nan")
            minimum_move_probability = float("nan")
            maximum_move_probability = float("nan")

        return NashAnalysis(
            equilibria=tuple(equilibria),
            move_probabilities=move_probabilities,
            accept_probabilities=accept_probabilities,
            minimum_move_probability=(
                minimum_move_probability
            ),
            maximum_move_probability=(
                maximum_move_probability
            ),
            closest_move_probability=(
                closest_move_probability
            ),
            move_probability_gap=move_probability_gap,
        )

    def _find_pure_equilibria(
        self,
        *,
        household_values: np.ndarray,
        neighborhood_values: np.ndarray,
    ) -> list[NashEquilibrium]:
        equilibria: list[NashEquilibrium] = []

        for row in range(2):
            for column in range(2):
                household_best_response = (
                    household_values[row, column]
                    >= household_values[
                        1 - row,
                        column,
                    ]
                    - self.tolerance
                )

                neighborhood_best_response = (
                    neighborhood_values[row, column]
                    >= neighborhood_values[
                        row,
                        1 - column,
                    ]
                    - self.tolerance
                )

                if not (
                    household_best_response
                    and neighborhood_best_response
                ):
                    continue

                p_move = 1.0 if row == 0 else 0.0
                p_accept = (
                    1.0 if column == 0 else 0.0
                )

                equilibria.append(
                    NashEquilibrium(
                        kind="pure",
                        p_move=p_move,
                        p_accept=p_accept,
                        household_strategy=(
                            p_move,
                            1.0 - p_move,
                        ),
                        neighborhood_strategy=(
                            p_accept,
                            1.0 - p_accept,
                        ),
                        household_expected_value=float(
                            household_values[row, column]
                        ),
                        neighborhood_expected_value=float(
                            neighborhood_values[
                                row,
                                column,
                            ]
                        ),
                        pure_action_profile=(
                            row,
                            column,
                        ),
                    )
                )

        return equilibria

    def _find_interior_mixed_equilibrium(
        self,
        *,
        household_values: np.ndarray,
        neighborhood_values: np.ndarray,
    ) -> NashEquilibrium | None:
        a = household_values
        b = neighborhood_values

        household_denominator = (
            a[0, 0]
            - a[0, 1]
            - a[1, 0]
            + a[1, 1]
        )

        neighborhood_denominator = (
            b[0, 0]
            - b[0, 1]
            - b[1, 0]
            + b[1, 1]
        )

        if (
            abs(household_denominator)
            <= self.tolerance
            or abs(neighborhood_denominator)
            <= self.tolerance
        ):
            return None

        p_accept = (
            a[1, 1] - a[0, 1]
        ) / household_denominator

        p_move = (
            b[1, 1] - b[1, 0]
        ) / neighborhood_denominator

        if not (
            self.tolerance
            < p_move
            < 1.0 - self.tolerance
        ):
            return None

        if not (
            self.tolerance
            < p_accept
            < 1.0 - self.tolerance
        ):
            return None

        household_strategy = np.array(
            [p_move, 1.0 - p_move],
            dtype=float,
        )

        neighborhood_strategy = np.array(
            [p_accept, 1.0 - p_accept],
            dtype=float,
        )

        household_expected_value = float(
            household_strategy
            @ household_values
            @ neighborhood_strategy
        )

        neighborhood_expected_value = float(
            household_strategy
            @ neighborhood_values
            @ neighborhood_strategy
        )

        return NashEquilibrium(
            kind="mixed",
            p_move=float(p_move),
            p_accept=float(p_accept),
            household_strategy=(
                float(p_move),
                float(1.0 - p_move),
            ),
            neighborhood_strategy=(
                float(p_accept),
                float(1.0 - p_accept),
            ),
            household_expected_value=(
                household_expected_value
            ),
            neighborhood_expected_value=(
                neighborhood_expected_value
            ),
            pure_action_profile=None,
        )

    def _deduplicate_equilibria(
        self,
        equilibria: list[NashEquilibrium],
    ) -> list[NashEquilibrium]:
        unique: list[NashEquilibrium] = []

        for candidate in equilibria:
            duplicate = any(
                abs(
                    candidate.p_move
                    - existing.p_move
                )
                <= self.tolerance
                and abs(
                    candidate.p_accept
                    - existing.p_accept
                )
                <= self.tolerance
                for existing in unique
            )

            if not duplicate:
                unique.append(candidate)

        return unique

    @staticmethod
    def _validate_matrix(
        matrix: np.ndarray,
        *,
        name: str,
    ) -> np.ndarray:
        matrix = np.asarray(
            matrix,
            dtype=float,
        )

        if matrix.shape != (2, 2):
            raise ValueError(
                f"{name} must have shape (2, 2)."
            )

        if not np.all(np.isfinite(matrix)):
            raise ValueError(
                f"{name} must contain only finite values."
            )

        return matrix.copy()

