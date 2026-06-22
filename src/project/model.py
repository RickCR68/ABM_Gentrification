from __future__ import annotations

import math
from itertools import accumulate

import numpy as np

from mesa import Model
from mesa.datacollection import DataCollector
from mesa.discrete_space import OrthogonalMooreGrid

from .agents import SchellingAgent
from .destination_choice import RandomSatisficingChoice
from .neighbourhood_state import NeighborhoodStateManager
from .neighborhoods import MooreNeighborhood
from .utility import IncomeNeighborhoodUtility


class GentrificationModel(Model):
    """Agent-based model of income dynamics and residential relocation."""

    def __init__(
        self,
        *,
        width: int = 20,
        height: int = 20,
        density: float = 0.8,

        neighborhood_radius: int = 1,

        affordability_share: float = 0.8,
        rent_adjustment_rate: float = 0.1,

        initial_income_min: float = 0.2,
        initial_income_max: float = 1.0,

        discount_factor_min: float = 0.0,
        discount_factor_max: float = 1.0,

        risk_aversion_min: float = 0.0,
        risk_aversion_max: float = 0.8,

        rationality_min: float = 1.0,
        rationality_max: float = 10.0,

        income_similarity_min: float = 0.0,
        income_similarity_max: float = 2.0,

        satisficing_threshold: float = 0.1,
        minimum_absolute_improvement: float = 1e-6,

        vision_income_scale: float = 5.0,
        maximum_vision_radius: int = 10,

        steps_until_satisfied: int = 5,

        income_growth_scaling: float = 0.01,
        income_volatility: float = 0.05,

        rng=None,
    ) -> None:
        super().__init__(rng=rng)

        self._validate_parameters(
            width=width,
            height=height,
            density=density,
            neighborhood_radius=neighborhood_radius,
            affordability_share=affordability_share,
            rent_adjustment_rate=rent_adjustment_rate,
            initial_income_min=initial_income_min,
            initial_income_max=initial_income_max,
            discount_factor_min=discount_factor_min,
            discount_factor_max=discount_factor_max,
            risk_aversion_min=risk_aversion_min,
            risk_aversion_max=risk_aversion_max,
            rationality_min=rationality_min,
            rationality_max=rationality_max,
            income_similarity_min=income_similarity_min,
            income_similarity_max=income_similarity_max,
            satisficing_threshold=satisficing_threshold,
            minimum_absolute_improvement=(
                minimum_absolute_improvement
            ),
            vision_income_scale=vision_income_scale,
            maximum_vision_radius=maximum_vision_radius,
            steps_until_satisfied=steps_until_satisfied,
            income_growth_scaling=income_growth_scaling,
            income_volatility=income_volatility,
        )

        # Grid and population parameters
        self.width = width
        self.height = height
        self.density = density

        # Initial income bounds
        self.initial_income_min = initial_income_min
        self.initial_income_max = initial_income_max

        # Heterogeneous household parameter bounds
        self.discount_factor_min = discount_factor_min
        self.discount_factor_max = discount_factor_max

        self.risk_aversion_min = risk_aversion_min
        self.risk_aversion_max = risk_aversion_max

        self.rationality_min = rationality_min
        self.rationality_max = rationality_max

        self.income_similarity_min = income_similarity_min
        self.income_similarity_max = income_similarity_max

        # Global satisficing parameters
        self.satisficing_threshold = satisficing_threshold
        self.minimum_absolute_improvement = (
            minimum_absolute_improvement
        )

        # Vision parameters
        self.vision_income_scale = vision_income_scale
        self.maximum_vision_radius = maximum_vision_radius

        # Residence-duration satisfaction
        self.steps_until_satisfied = steps_until_satisfied

        # Income and rent dynamics
        self.income_growth_scaling = income_growth_scaling
        self.income_volatility = income_volatility
        self.rent_adjustment_rate = rent_adjustment_rate

        # Step-level counters
        self.satisfied_count = 0
        self.move_attempts = 0
        self.successful_moves = 0
        self.failed_searches = 0

        # Spatial environment
        self.grid = OrthogonalMooreGrid(
            dimensions=(width, height),
            torus=True,
            capacity=1,
            random=self.random,
        )

        self.neighborhood_definition = MooreNeighborhood(
            radius=neighborhood_radius,
        )

        self._create_property_layers()

        # Modular policies and state managers
        self.neighborhood_state = NeighborhoodStateManager(
            neighborhood_definition=(
                self.neighborhood_definition
            ),
            rent_adjustment_rate=rent_adjustment_rate,
        )

        self.utility_policy = IncomeNeighborhoodUtility(
            affordability_share=affordability_share,
            income_growth_scaling=income_growth_scaling,
        )

        self.destination_choice_policy = (
            RandomSatisficingChoice()
        )

        self.datacollector = self._create_datacollector()

        # Create households and initialise neighbourhood state
        self._create_agents()

        self.neighborhood_state.initialize_income_layer(
            self
        )

        self._update_agent_states()
        self._update_satisfaction()

        self.datacollector.collect(self)

    def _create_property_layers(self) -> None:
        """Create rent and current neighbourhood-income layers."""
        self.grid.create_property_layer(
            name="rent",
            default_value=0.0,
            dtype=float,
        )

        self.grid.rent.modify_cells(
            lambda cell: self.random.uniform(
                0.0,
                0.5,
            )
        )

        self.grid.create_property_layer(
            name="mean_neighbor_income",
            default_value=0.0,
            dtype=float,
        )

    def _create_datacollector(self) -> DataCollector:
        """Construct the Mesa data collector."""
        return DataCollector(
            model_reporters={
                "satisfied_count": "satisfied_count",
                "pct_satisfied": (
                    self.percentage_satisfied
                ),

                "move_attempts": "move_attempts",
                "successful_moves": "successful_moves",
                "failed_searches": "failed_searches",
                "movement_success_rate": (
                    self.movement_success_rate
                ),

                "city_mean_income": self.city_mean_income,
                "mean_neighbor_income": (
                    self.mean_neighbor_income
                ),
                "mean_rent": self.mean_rent,

                "mean_utility": self.mean_utility,
                "mean_value": self.mean_value,
                "mean_vision": self.mean_vision,

                "rent_income_timescale_ratio": (
                    self.rent_income_timescale_ratio
                ),
                "gini_coefficient": (
                    self.gini_coefficient
                ),
            },
            agent_reporters={
                "initial_income": "initial_income",
                "income": "income",

                "income_similarity_preference": (
                    "income_similarity_preference"
                ),
                "discount_factor": "discount_factor",
                "risk_aversion": "risk_aversion",
                "rationality": "rationality",

                "vision_radius": "vision_radius",

                "current_utility": "current_utility",
                "current_value": "current_value",

                "current_location_affordable": (
                    "current_location_affordable"
                ),

                "steps_since_move": "steps_since_move",
                "satisfied": "satisfied",
                "moved_this_step": "moved_this_step",

                "last_move_successful": (
                    "last_move_successful"
                ),
                "last_value_improvement": (
                    "last_value_improvement"
                ),
            },
        )

    @staticmethod
    def _validate_parameters(
        *,
        width: int,
        height: int,
        density: float,
        neighborhood_radius: int,
        affordability_share: float,
        rent_adjustment_rate: float,
        initial_income_min: float,
        initial_income_max: float,
        discount_factor_min: float,
        discount_factor_max: float,
        risk_aversion_min: float,
        risk_aversion_max: float,
        rationality_min: float,
        rationality_max: float,
        income_similarity_min: float,
        income_similarity_max: float,
        satisficing_threshold: float,
        minimum_absolute_improvement: float,
        vision_income_scale: float,
        maximum_vision_radius: int,
        steps_until_satisfied: int,
        income_growth_scaling: float,
        income_volatility: float,
    ) -> None:
        """Validate model parameters."""
        if width <= 0 or height <= 0:
            raise ValueError(
                "width and height must be positive."
            )

        if not 0.0 <= density <= 1.0:
            raise ValueError(
                "density must lie between 0 and 1."
            )

        if neighborhood_radius < 1:
            raise ValueError(
                "neighborhood_radius must be at least 1."
            )

        if affordability_share < 0.0:
            raise ValueError(
                "affordability_share must be non-negative."
            )

        if not 0.0 < rent_adjustment_rate < 1.0:
            raise ValueError(
                "rent_adjustment_rate must be strictly "
                "between 0 and 1."
            )

        if initial_income_min <= 0.0:
            raise ValueError(
                "initial_income_min must be positive."
            )

        if initial_income_min > initial_income_max:
            raise ValueError(
                "initial_income_min cannot exceed "
                "initial_income_max."
            )

        if not (
            0.0
            <= discount_factor_min
            <= discount_factor_max
            <= 1.0
        ):
            raise ValueError(
                "Discount-factor bounds must satisfy "
                "0 <= min <= max <= 1."
            )

        if not (
            0.0
            <= risk_aversion_min
            <= risk_aversion_max
            < 5.0
        ):
            raise ValueError(
                "Risk-aversion bounds must satisfy "
                "0 <= min <= max < 5."
            )

        if not (
            0.0
            <= rationality_min
            <= rationality_max
        ):
            raise ValueError(
                "Rationality bounds must satisfy "
                "0 <= min <= max."
            )

        if income_similarity_min < 0.0:
            raise ValueError(
                "income_similarity_min must be non-negative."
            )

        if income_similarity_min > income_similarity_max:
            raise ValueError(
                "income_similarity_min cannot exceed "
                "income_similarity_max."
            )

        if satisficing_threshold < 0.0:
            raise ValueError(
                "satisficing_threshold must be non-negative."
            )

        if minimum_absolute_improvement < 0.0:
            raise ValueError(
                "minimum_absolute_improvement must be "
                "non-negative."
            )

        if vision_income_scale < 0.0:
            raise ValueError(
                "vision_income_scale must be non-negative."
            )

        if maximum_vision_radius < 1:
            raise ValueError(
                "maximum_vision_radius must be at least 1."
            )

        if steps_until_satisfied < 1:
            raise ValueError(
                "steps_until_satisfied must be at least 1."
            )

        if income_growth_scaling < 0.0:
            raise ValueError(
                "income_growth_scaling must be non-negative."
            )

        if income_volatility < 0.0:
            raise ValueError(
                "income_volatility must be non-negative."
            )

    def _create_agents(self) -> None:
        """Populate cells with heterogeneous households."""
        for cell in self.grid.all_cells:
            if self.random.random() >= self.density:
                continue

            income = self.random.uniform(
                self.initial_income_min,
                self.initial_income_max,
            )

            income_similarity_preference = (
                self.random.uniform(
                    self.income_similarity_min,
                    self.income_similarity_max,
                )
            )

            discount_factor = self.random.uniform(
                self.discount_factor_min,
                self.discount_factor_max,
            )

            risk_aversion = self.random.uniform(
                self.risk_aversion_min,
                self.risk_aversion_max,
            )

            rationality = self.random.uniform(
                self.rationality_min,
                self.rationality_max,
            )

            SchellingAgent(
                model=self,
                cell=cell,
                income=income,
                income_similarity_preference=(
                    income_similarity_preference
                ),
                discount_factor=discount_factor,
                risk_aversion=risk_aversion,
                rationality=rationality,
            )

    def _update_agent_states(self) -> None:
        """Recalculate utility at each household's current location."""
        self.agents.do("assign_state")

    def _update_satisfaction(self) -> None:
        """Update duration-based household satisfaction."""
        self.satisfied_count = 0
        self.agents.do("update_satisfaction")

    def step(self) -> None:
        """Advance the model by one time step."""
        self.move_attempts = 0
        self.successful_moves = 0
        self.failed_searches = 0

        # 1. Household incomes evolve.
        self.agents.shuffle_do("change_income")

        # 2. Recalculate local mean income after income changes.
        self.neighborhood_state.refresh_current_income(
            self
        )

        # 3. Rents adjust toward current local income.
        self.neighborhood_state.update_rents(self)

        # 4. Evaluate agents at their current locations.
        self._update_agent_states()

        # 5. Every agent searches and may move.
        self.agents.shuffle_do("step")

        # 6. Movement changes neighbourhood composition.
        self.neighborhood_state.refresh_current_income(
            self
        )

        # 7. Recalculate utility at final locations.
        self._update_agent_states()

        # 8. Update steps-since-move satisfaction.
        self._update_satisfaction()

        # 9. Collect one observation.
        self.datacollector.collect(self)

    def percentage_satisfied(self) -> float:
        """Return the percentage of duration-satisfied households."""
        population = len(self.agents)

        if population == 0:
            return 0.0

        return (
            100.0
            * self.satisfied_count
            / population
        )

    def movement_success_rate(self) -> float:
        """Return the fraction of searches producing a move."""
        if self.move_attempts == 0:
            return 0.0

        return (
            self.successful_moves
            / self.move_attempts
        )

    def city_mean_income(self) -> float:
        """Return mean household income across the city."""
        if len(self.agents) == 0:
            return 0.0

        return float(
            np.mean(
                [
                    agent.income
                    for agent in self.agents
                ]
            )
        )

    def mean_rent(self) -> float:
        """Return mean rent across all grid cells."""
        return float(
            np.mean(self.grid.rent.data)
        )

    def mean_neighbor_income(self) -> float:
        """Return the spatial mean of local mean-income values."""
        return float(
            np.mean(
                self.grid.mean_neighbor_income.data
            )
        )

    def mean_utility(self) -> float:
        """Return mean finite raw utility."""
        values = [
            agent.current_utility
            for agent in self.agents
            if math.isfinite(agent.current_utility)
        ]

        return (
            float(np.mean(values))
            if values
            else 0.0
        )

    def mean_value(self) -> float:
        """Return mean finite signed-CRRA value."""
        values = [
            agent.current_value
            for agent in self.agents
            if math.isfinite(agent.current_value)
        ]

        return (
            float(np.mean(values))
            if values
            else 0.0
        )

    def mean_vision(self) -> float:
        """Return mean fixed household vision radius."""
        if len(self.agents) == 0:
            return 0.0

        return float(
            np.mean(
                [
                    agent.vision_radius
                    for agent in self.agents
                ]
            )
        )

    def rent_income_timescale_ratio(self) -> float:
        """Return tau = delta / eta."""
        if self.income_growth_scaling == 0.0:
            return math.inf

        return (
            self.rent_adjustment_rate
            / self.income_growth_scaling
        )

    def gini_coefficient(self):
        """Calculate the Gini coefficient for household incomes."""
        incomes = [agent.income for agent in self.agents]

        if not incomes:
            return 0.0

        sorted_incomes = sorted(incomes)
        n = len(incomes)
        cumulative_incomes = [0] + list(
            accumulate(sorted_incomes)
        )

        total_income = cumulative_incomes[-1]
        if total_income == 0:
            return 0.0

        gini_numerator = sum(
            (i + 1) * income
            for i, income in enumerate(sorted_incomes)
        )
        gini_denominator = n * total_income

        gini_coefficient = (
                (2 * gini_numerator) / gini_denominator
                - (n + 1) / n
        )

        return gini_coefficient
