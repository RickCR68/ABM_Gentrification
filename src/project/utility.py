from __future__ import annotations

import math
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from mesa.discrete_space import Cell

    from .agents import SchellingAgent


@dataclass(frozen=True)
class LocationEvaluation:
    """Utility evaluation of one household-location pair."""

    cell: Cell

    # Neighbourhood state
    rent: float
    mean_neighbor_income: float

    # Affordability
    maximum_affordable_rent: float
    affordable: bool

    # Utility components
    income_distance: float
    spillover: float

    rent_component: float
    similarity_component: float
    growth_component: float

    # Final utility and transformed value
    utility: float
    value: float


class UtilityPolicy(ABC):
    """Interface for interchangeable household utility models."""

    @abstractmethod
    def evaluate(
        self,
        agent: SchellingAgent,
        cell: Cell,
        *,
        is_current_location: bool = False,
    ) -> LocationEvaluation:
        """Evaluate a location for one household."""
        raise NotImplementedError


class IncomeNeighborhoodUtility(UtilityPolicy):
    """Utility based on rent, income similarity, and spillover potential.

    The utility of household i in neighbourhood j is

        U_ij
        =
        R_j
        - theta_i |y_i - mean_y_j|
        + beta_i eta S_ij,

    where

        S_ij
        =
        max(
            0,
            (mean_y_j - y_i) / mean_y_j
        ).

    A location is feasible only when

        R_j <= alpha y_i.

    Utility is transformed using signed CRRA:

        V(U)
        =
        sign(U)
        |U|^(1-rho_i)
        / (1-rho_i).
    """

    def __init__(
        self,
        *,
        affordability_share: float = 0.8,
        income_growth_scaling: float = 0.01,
        infeasible_utility: float = -math.inf,
    ) -> None:
        self._validate_policy_parameters(
            affordability_share=affordability_share,
            income_growth_scaling=income_growth_scaling,
        )

        self.affordability_share = affordability_share
        self.income_growth_scaling = income_growth_scaling
        self.infeasible_utility = infeasible_utility

    @staticmethod
    def _validate_policy_parameters(
        *,
        affordability_share: float,
        income_growth_scaling: float,
    ) -> None:
        """Validate model-level utility parameters."""
        if affordability_share < 0.0:
            raise ValueError(
                "affordability_share must be non-negative."
            )

        if income_growth_scaling < 0.0:
            raise ValueError(
                "income_growth_scaling must be non-negative."
            )

    def evaluate(
        self,
        agent: SchellingAgent,
        cell: Cell,
        *,
        is_current_location: bool = False,
    ) -> LocationEvaluation:
        """Evaluate one location for one household.

        The `is_current_location` parameter is retained for interface
        compatibility, but it currently does not alter utility because
        moving costs are excluded while the application game is ignored.
        """
        rent = self._layer_value(
            agent.model.grid.rent,
            cell,
        )

        mean_income = self._layer_value(
            agent.model.grid.mean_neighbor_income,
            cell,
        )

        maximum_affordable_rent = (
            self.maximum_affordable_rent(
                income=agent.income
            )
        )

        affordable = self.is_affordable(
            income=agent.income,
            rent=rent,
        )

        income_distance = abs(
            agent.income - mean_income
        )

        spillover = self.calculate_spillover(
            income=agent.income,
            mean_neighbor_income=mean_income,
        )

        rent_component = rent

        similarity_component = (
            -agent.income_similarity_preference
            * income_distance
        )

        growth_component = (
            agent.discount_factor
            * self.income_growth_scaling
            * spillover
        )

        if not affordable:
            return LocationEvaluation(
                cell=cell,
                rent=rent,
                mean_neighbor_income=mean_income,
                maximum_affordable_rent=(
                    maximum_affordable_rent
                ),
                affordable=False,
                income_distance=income_distance,
                spillover=spillover,
                rent_component=rent_component,
                similarity_component=similarity_component,
                growth_component=growth_component,
                utility=self.infeasible_utility,
                value=self.infeasible_utility,
            )

        utility = self.calculate_utility(
            rent_component=rent_component,
            similarity_component=similarity_component,
            growth_component=growth_component,
        )

        value = self.apply_signed_crra(
            utility=utility,
            risk_aversion=agent.risk_aversion,
        )

        return LocationEvaluation(
            cell=cell,
            rent=rent,
            mean_neighbor_income=mean_income,
            maximum_affordable_rent=(
                maximum_affordable_rent
            ),
            affordable=True,
            income_distance=income_distance,
            spillover=spillover,
            rent_component=rent_component,
            similarity_component=similarity_component,
            growth_component=growth_component,
            utility=utility,
            value=value,
        )

    @staticmethod
    def _layer_value(
        layer,
        cell: Cell,
    ) -> float:
        """Read a property-layer value at a cell coordinate."""
        return float(
            layer.data[cell.coordinate]
        )

    def maximum_affordable_rent(
        self,
        *,
        income: float,
    ) -> float:
        """Return the highest rent the household can afford."""
        return self.affordability_share * income

    def is_affordable(
        self,
        *,
        income: float,
        rent: float,
    ) -> bool:
        """Check whether R_j <= alpha y_i."""
        return rent <= self.maximum_affordable_rent(
            income=income
        )

    @staticmethod
    def calculate_spillover(
        *,
        income: float,
        mean_neighbor_income: float,
    ) -> float:
        """Calculate the positive neighbourhood-income spillover term."""
        if mean_neighbor_income <= 0.0:
            return 0.0

        return max(
            0.0,
            (
                mean_neighbor_income - income
            )
            / mean_neighbor_income,
        )

    @staticmethod
    def calculate_utility(
        *,
        rent_component: float,
        similarity_component: float,
        growth_component: float,
    ) -> float:
        """Calculate raw household utility."""
        return (
            rent_component
            + similarity_component
            + growth_component
        )

    @staticmethod
    def apply_signed_crra(
        *,
        utility: float,
        risk_aversion: float,
    ) -> float:
        """Apply the signed CRRA transformation.

        Standard CRRA is not defined for negative utility values. This
        signed extension preserves the utility sign while applying the
        CRRA curvature to its magnitude.
        """
        if not 0.0 <= risk_aversion < 5.0:
            raise ValueError(
                "risk_aversion must satisfy "
                "0 <= rho < 5."
            )

        if not math.isfinite(utility):
            return utility

        if utility == 0.0:
            return 0.0

        if risk_aversion == 0.0:
            return utility

        sign = (
            1.0
            if utility > 0.0
            else -1.0
        )

        magnitude = (
            abs(utility)
            ** (1.0 - risk_aversion)
            / (1.0 - risk_aversion)
        )

        return sign * magnitude
