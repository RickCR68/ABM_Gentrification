from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    from numpy.typing import NDArray

    from .model import GentrificationModel
    from .neighborhoods import NeighborhoodDefinition


class NeighborhoodStateManager:
    """Manage neighbourhood mean-income and rent property layers.

    This manager is responsible for:

    - calculating local mean household income around every cell;
    - refreshing the current neighbourhood-income layer;
    - adjusting rent toward current neighbourhood mean income.

    The model determines when these updates occur.
    """

    def __init__(
        self,
        *,
        neighborhood_definition: NeighborhoodDefinition,
        rent_adjustment_rate: float,
        empty_neighborhood_income: float = 0.0,
    ) -> None:
        """Create a neighbourhood-state manager.

        Args:
            neighborhood_definition:
                Defines which households belong to the neighbourhood
                surrounding each cell.
            rent_adjustment_rate:
                Delta in the rent-adjustment equation.
            empty_neighborhood_income:
                Value assigned to cells without neighbouring households.
        """
        if not 0.0 < rent_adjustment_rate < 1.0:
            raise ValueError(
                "rent_adjustment_rate must be strictly between 0 and 1."
            )

        if empty_neighborhood_income < 0.0:
            raise ValueError(
                "empty_neighborhood_income must be non-negative."
            )

        self.neighborhood_definition = neighborhood_definition
        self.rent_adjustment_rate = rent_adjustment_rate
        self.empty_neighborhood_income = (
            empty_neighborhood_income
        )

    def calculate_mean_income_and_variance(
        self,
        model: GentrificationModel,
    ) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
        """Calculate local mean income and variance around every grid cell.

        For each cell j, this calculates

            mean_y_j
            =
            mean income of households in j's neighbourhood.

            var_y_j
            = variance of household incomes in j's neighbourhood.

        If no neighbouring households are present, the configured
        `empty_neighborhood_income` value is used.
        """
        mean_incomes = np.full(
            model.grid.dimensions,
            fill_value=self.empty_neighborhood_income,
            dtype=float,
        )

        var_incomes = np.full(
            model.grid.dimensions,
            fill_value=0.0,
            dtype=float,
        )

        for cell in model.grid.all_cells:
            neighbors = (
                self.neighborhood_definition
                .get_neighbors(cell)
            )

            if not neighbors:
                continue

            incomes = np.fromiter(
                (
                    neighbor.income
                    for neighbor in neighbors
                ),
                dtype=float,
            )

            if incomes.size == 0:
                continue

            mean_incomes[cell.coordinate] = float(
                np.mean(incomes)
            )

            var_incomes[cell.coordinate] = float(
                np.var(incomes, ddof=0)
            )

        return mean_incomes, var_incomes

    def initialize_income_layer(
        self,
        model: GentrificationModel,
    ) -> None:
        """Initialize the current neighbourhood-income layer."""
        self.refresh_current_income(model)

    def refresh_current_income(
        self,
        model: GentrificationModel,
    ) -> None:
        """Recalculate the current neighbourhood-income layer.

        This method can be called more than once in a model step, such as:

        - after household incomes change;
        - after households relocate.
        """
        updated_mean_income, updated_mean_variance = self.calculate_mean_income_and_variance(
            model
        )
        #TODO: add variance layer to model.grid and update it here
        model.grid.mean_neighbor_income.data[:] = (
            updated_mean_income
        )
        model.grid.neighbor_income_variance.data[:] = updated_mean_variance

    def update_rents(
        self,
        model: GentrificationModel,
    ) -> None:
        """Adjust rent toward current neighbourhood mean income.

        This implements

            R_j^n
            =
            R_j^(n-1)
            + delta(
                mean_y_j - R_j^(n-1)
            ).
        """

        affordability_share = model.affordability_share

        rents = model.grid.rent.data

        mean_incomes = (
            model.grid.mean_neighbor_income.data
        )

        delta = self.rent_adjustment_rate


        rents[:] = rents + delta * (
            affordability_share*mean_incomes - rents
        )

    def mean_neighborhood_income(
        self,
        model: GentrificationModel,
    ) -> float:
        """Return the spatial mean of neighbourhood mean income."""
        return float(
            np.mean(
                model.grid.mean_neighbor_income.data
            )
        )

    def mean_rent(
        self,
        model: GentrificationModel,
    ) -> float:
        """Return mean rent across all grid cells."""
        return float(
            np.mean(model.grid.rent.data)
        )
