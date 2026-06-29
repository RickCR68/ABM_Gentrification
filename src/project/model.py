from __future__ import annotations

import math
from collections import defaultdict
from itertools import accumulate
from threading import RLock

import numpy as np

from mesa import Model
from mesa.datacollection import DataCollector
from mesa.discrete_space import OrthogonalMooreGrid

from .agents import SchellingAgent
from .game import (
    ApplicationGamePolicy,
    LogitQRESolver,
    MeanResidentUtilityPolicy,
    TransferPayoffStructure,
)
from .destination_choice import (
    RandomSatisficingChoice,
)
from .metrics import homeless_fraction, theil_index, moran_i, spatial_entropy, neighborhood_heterogeneity, \
    income_mobility_indicator, segregation_index, gentrification_indicator
from .nash import TwoByTwoNashSolver
from .neighbourhood_state import (
    NeighborhoodStateManager,
)
from .neighborhoods import MooreNeighborhood
from .utility import IncomeNeighborhoodUtility


class ThreadSafeDataCollector(DataCollector):
    """DataCollector variant that serializes collection and dataframe reads."""

    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self._lock = RLock()

    def collect(self, model) -> None:
        with self._lock:
            super().collect(model)

    def get_model_vars_dataframe(self):
        with self._lock:
            return super().get_model_vars_dataframe()

    def get_agent_vars_dataframe(self):
        with self._lock:
            return super().get_agent_vars_dataframe()


class GentrificationModel(Model):
    """Agent-based model with relocation games and NE diagnostics."""

    def __init__(
        self,
        *,
        width: int = 11,
        height: int = 11,
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

        moving_cost: float = 0.05,
        rejection_cost: float = 0.05,

        neighborhood_risk_aversion: float = 0.5,
        neighborhood_rationality: float = 5.0,

        qre_tolerance: float = 1e-10,
        qre_maximum_iterations: int = 1000,
        qre_damping: float = 0.5,

        keep_game_history: bool = True,
        keep_agents: bool = True,

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
            minimum_absolute_improvement=minimum_absolute_improvement,
            vision_income_scale=vision_income_scale,
            maximum_vision_radius=maximum_vision_radius,
            steps_until_satisfied=steps_until_satisfied,
            income_growth_scaling=income_growth_scaling,
            income_volatility=income_volatility,
            moving_cost=moving_cost,
            rejection_cost=rejection_cost,
            neighborhood_risk_aversion=neighborhood_risk_aversion,
            neighborhood_rationality=neighborhood_rationality,
            qre_tolerance=qre_tolerance,
            qre_maximum_iterations=qre_maximum_iterations,
            qre_damping=qre_damping,
        )

        self.keep_agents = keep_agents

        self.affordability_share = affordability_share

        self.width = width
        self.height = height
        self.density = density

        self.initial_income_min = initial_income_min
        self.initial_income_max = initial_income_max

        self.discount_factor_min = discount_factor_min
        self.discount_factor_max = discount_factor_max

        self.risk_aversion_min = risk_aversion_min
        self.risk_aversion_max = risk_aversion_max

        self.rationality_min = rationality_min
        self.rationality_max = rationality_max

        self.income_similarity_min = income_similarity_min
        self.income_similarity_max = income_similarity_max

        self.satisficing_threshold = (
            satisficing_threshold
        )
        self.minimum_absolute_improvement = (
            minimum_absolute_improvement
        )

        # Vision parameters
        self.vision_income_scale = vision_income_scale
        self.maximum_vision_radius = (
            maximum_vision_radius
        )

        self.steps_until_satisfied = (
            steps_until_satisfied
        )

        self.income_growth_scaling = (
            income_growth_scaling
        )
        self.income_volatility = income_volatility
        self.rent_adjustment_rate = (
            rent_adjustment_rate
        )

        self.moving_cost = moving_cost
        self.rejection_cost = rejection_cost

        self.neighborhood_risk_aversion = (
            neighborhood_risk_aversion
        )
        self.neighborhood_rationality = (
            neighborhood_rationality
        )

        self.keep_game_history = keep_game_history
        self.game_history: list[dict[str, object]] = []
        self.game_records_this_step = []

        self.satisfied_count = 0
        self.move_attempts = 0
        self.successful_moves = 0
        self.failed_searches = 0
        self.voluntary_stays = 0
        self.destination_conflicts = 0

        self.move_accept_outcomes = 0
        self.move_reject_outcomes = 0
        self.stay_accept_outcomes = 0
        self.stay_reject_outcomes = 0

        self.qre_nonconvergence_count = 0

        self.grid = OrthogonalMooreGrid(
            dimensions=(width, height),
            torus=True,
            capacity=1,
            random=self.random,
        )

        self.neighborhood_definition = (
            MooreNeighborhood(
                radius=neighborhood_radius,
            )
        )

        self._create_property_layers()

        self.neighborhood_state = (
            NeighborhoodStateManager(
                neighborhood_definition=(
                    self.neighborhood_definition
                ),
                rent_adjustment_rate=(
                    rent_adjustment_rate
                ),
            )
        )

        self.utility_policy = (
            IncomeNeighborhoodUtility(
                affordability_share=(
                    self.affordability_share
                ),
                income_growth_scaling=(
                    income_growth_scaling
                ),
                infeasible_utility=-0,
            )
        )

        self.destination_choice_policy = (
            RandomSatisficingChoice()
        )

        self.application_game_policy = (
            ApplicationGamePolicy(
                neighborhood_utility_policy=(
                    MeanResidentUtilityPolicy()
                ),
                payoff_structure=(
                    TransferPayoffStructure()
                ),
                qre_solver=LogitQRESolver(
                    tolerance=qre_tolerance,
                    maximum_iterations=(
                        qre_maximum_iterations
                    ),
                    damping=qre_damping,
                ),
                nash_solver=TwoByTwoNashSolver(),
            )
        )

        self.datacollector = (
            self._create_datacollector()
        )

        self._create_agents()

        self.neighborhood_state.initialize_income_layer(
            self
        )

        self.neighborhood_state.initialize_rent(self)

        self._update_agent_states()
        self._update_percentiles()

        self.datacollector.collect(self)

    def _create_property_layers(self) -> None:
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

        self.grid.create_property_layer(
            name="neighbor_income_variance",
            default_value=0.0,
            dtype=float,
        )

    def _create_datacollector(self) -> DataCollector:
        """Construct the Mesa data collector."""
        return ThreadSafeDataCollector(
            model_reporters={
                "satisfied_count": "satisfied_count",
                "pct_satisfied": (
                    self.percentage_satisfied
                ),
                "move_attempts": "move_attempts",
                "successful_moves": "successful_moves",
                "failed_searches": "failed_searches",
                "voluntary_stays": "voluntary_stays",
                "destination_conflicts": (
                    "destination_conflicts"
                ),
                "movement_success_rate": (
                    self.movement_success_rate
                ),
                # "cells_rent": [cell.rent for cell in self.grid.all_cells],
                "city_mean_income": (
                    self.city_mean_income
                ),
                "mean_neighbor_income": (
                    self.mean_neighbor_income
                ),
                "mean_neighbor_income_variance": (
                    self.mean_neighbor_income_variance
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

                "mean_qre_move_probability": (
                    self.mean_qre_move_probability
                ),
                "mean_qre_accept_probability": (
                    self.mean_qre_accept_probability
                ),
                "mean_ne_move_probability": (
                    self.mean_ne_move_probability
                ),
                "mean_qre_ne_move_gap": (
                    self.mean_qre_ne_move_gap
                ),
                "mean_delta_neighborhood_utility": (
                    self.mean_delta_neighborhood_utility
                ),

                "move_accept_outcomes": (
                    "move_accept_outcomes"
                ),
                "move_reject_outcomes": (
                    "move_reject_outcomes"
                ),
                "stay_accept_outcomes": (
                    "stay_accept_outcomes"
                ),
                "stay_reject_outcomes": (
                    "stay_reject_outcomes"
                ),

                "qre_nonconvergence_count": (
                    "qre_nonconvergence_count"
                ),

                "ne_following_rate": (
                    self.ne_following_rate
                ),

                # Segregation and Gentrification Metrics
                "homeless_fraction": (
                    self.get_homeless_fraction
                ),
                "theil_index": (
                    self.get_theil_index
                ),
                "moran_i": (
                    self.get_moran_i
                ),
                "spatial_entropy": (
                    self.get_spatial_entropy
                ),
                "neighborhood_heterogeneity": (
                    self.get_neighborhood_heterogeneity
                ),
                "income_mobility_indicator": (
                    self.get_income_mobility_indicator
                ),
                "segregation_index": (
                    self.get_segregation_index
                ),
                "gentrification_indicator": (
                    self.get_gentrification_indicator
                ),
            },
            agent_reporters={
                "initial_income": "initial_income",
                "income": "income",
                "cell_coordinate": lambda a: a.cell.coordinate,
                "disposeable_income": (
                    'disposeable_income'
                ),
                "income_similarity_preference": (
                    "income_similarity_preference"
                ),
                "coordinates": lambda a: a.cell.coordinate,
                "rent": lambda a: a.cell.rent,
                "discount_factor": (
                    "discount_factor"
                ),
                "risk_aversion": "risk_aversion",
                "rationality": "rationality",

                "vision_radius": "vision_radius",

                "current_utility": "current_utility",
                "current_value": "current_value",

                "steps_since_move": (
                    "steps_since_move"
                ),
                "satisfied": "satisfied",
                "moved_this_step": (
                    "moved_this_step"
                ),

                "last_qre_move_probability": (
                    "last_qre_move_probability"
                ),
                "last_qre_accept_probability": (
                    "last_qre_accept_probability"
                ),
                "last_ne_move_probability": (
                    "last_ne_move_probability"
                ),
                "last_qre_ne_move_gap": (
                    "last_qre_ne_move_gap"
                ),
            } if self.keep_agents else None,
        )

    @staticmethod
    def _validate_parameters(
        **parameters,
    ) -> None:
        width = parameters["width"]
        height = parameters["height"]
        density = parameters["density"]

        if width <= 0 or height <= 0:
            raise ValueError(
                "width and height must be positive."
            )

        if not 0.0 <= density <= 1.0:
            raise ValueError(
                "density must lie between 0 and 1."
            )

        if parameters["neighborhood_radius"] < 1:
            raise ValueError(
                "neighborhood_radius must be at least 1."
            )

        if parameters["affordability_share"] < 0.0:
            raise ValueError(
                "affordability_share must be non-negative."
            )

        if not (
            0.0
            < parameters["rent_adjustment_rate"]
            < 1.0
        ):
            raise ValueError(
                "rent_adjustment_rate must lie strictly between 0 and 1."
            )

        if parameters["initial_income_min"] <= 0.0:
            raise ValueError(
                "initial_income_min must be positive."
            )

        if (
            parameters["initial_income_min"]
            > parameters["initial_income_max"]
        ):
            raise ValueError(
                "initial_income_min cannot exceed initial_income_max."
            )

        for prefix in (
            "discount_factor",
            "risk_aversion",
            "rationality",
            "income_similarity",
        ):
            minimum = parameters[f"{prefix}_min"]
            maximum = parameters[f"{prefix}_max"]

            if minimum > maximum:
                raise ValueError(
                    f"{prefix}_min cannot exceed {prefix}_max."
                )

        # if not (
        #     # 0.0
        #     # <= parameters["risk_aversion_min"]
        #     parameters["risk_aversion_max"]
        #     < 1.0
        # ):
        #     raise ValueError(
        #         "Household risk aversion must satisfy 0 <= rho < 999."
        #     )

        if parameters["rationality_min"] < 0.0:
            raise ValueError(
                "rationality_min must be non-negative."
            )

        if parameters["income_similarity_min"] < 0.0:
            raise ValueError(
                "income_similarity_min must be non-negative."
            )

        for name in (
            "satisficing_threshold",
            "minimum_absolute_improvement",
            "vision_income_scale",
            "income_growth_scaling",
            "income_volatility",
            "moving_cost",
            "rejection_cost",
            "neighborhood_rationality",
        ):
            if parameters[name] < 0.0:
                raise ValueError(
                    f"{name} must be non-negative."
                )

        if parameters["maximum_vision_radius"] < 1:
            raise ValueError(
                "maximum_vision_radius must be at least 1."
            )

        if parameters["steps_until_satisfied"] < 1:
            raise ValueError(
                "steps_until_satisfied must be at least 1."
            )

        if not (
            0.0
            <= parameters[
                "neighborhood_risk_aversion"
            ]
            < 1.0
        ):
            raise ValueError(
                "neighborhood_risk_aversion must satisfy 0 <= rho_n < 1."
            )

        if parameters["qre_tolerance"] <= 0.0:
            raise ValueError(
                "qre_tolerance must be positive."
            )

        if parameters["qre_maximum_iterations"] < 1:
            raise ValueError(
                "qre_maximum_iterations must be positive."
            )

        if not (
            0.0 < parameters["qre_damping"] <= 1.0
        ):
            raise ValueError(
                "qre_damping must satisfy 0 < damping <= 1."
            )

    def _create_agents(self) -> None:
        for cell in self.grid.all_cells:
            if self.random.random() >= self.density:
                continue

            SchellingAgent(
                model=self,
                cell=cell,
                income=self.random.uniform(
                    self.initial_income_min,
                    self.initial_income_max,
                ),
                income_similarity_preference=(
                    self.random.uniform(
                        self.income_similarity_min,
                        self.income_similarity_max,
                    )
                ),
                discount_factor=self.random.uniform(
                    self.discount_factor_min,
                    self.discount_factor_max,
                ),
                risk_aversion=self.random.uniform(
                    self.risk_aversion_min,
                    self.risk_aversion_max,
                ),
                rationality=self.random.uniform(
                    self.rationality_min,
                    self.rationality_max,
                ),
            )

    def _update_agent_states(self) -> None:
        self.agents.do("assign_state")

    def _update_satisfaction(self) -> None:
        self.satisfied_count = 0
        self.agents.do("update_satisfaction")

    def _reset_step_statistics(self) -> None:
        self.move_attempts = 0
        self.successful_moves = 0
        self.failed_searches = 0
        self.voluntary_stays = 0
        self.destination_conflicts = 0

        self.move_accept_outcomes = 0
        self.move_reject_outcomes = 0
        self.stay_accept_outcomes = 0
        self.stay_reject_outcomes = 0

        self.qre_nonconvergence_count = 0
        self.game_records_this_step = []

    def _execute_simultaneous_moves(self) -> None:
        """Resolve competing claims to initially empty destinations."""
        claims = defaultdict(list)

        for agent in self.agents:
            if agent.pending_destination is None:
                continue

            coordinate = (
                agent.pending_destination
                .cell
                .coordinate
            )

            claims[coordinate].append(agent)

        for claimants in claims.values():
            winner = self.random.choice(claimants)
            winner.execute_pending_move()

            for agent in claimants:
                if agent is winner:
                    continue

                agent.cancel_pending_move_due_to_conflict()

    def step(self) -> None:
        self._reset_step_statistics()

        # 1. Income changes.
        self.agents.shuffle_do("change_income")
        self.agents.shuffle_do("change_disposeable_income")

        # 2. Refresh local incomes and rents.
        self.neighborhood_state.refresh_current_income(
            self
        )
        self.neighborhood_state.update_rents(self)

        # 3. Evaluate the common pre-move state.
        self._update_agent_states()

        # 4. Every agent searches and plays its game without moving.
        self.agents.shuffle_do(
            "prepare_move_decision"
        )

        # 5. Resolve destination conflicts and execute moves together.
        self._execute_simultaneous_moves()

        # 6. Refresh the post-move spatial state.
        self.neighborhood_state.refresh_current_income(
            self
        )
        self._update_agent_states()
        self._update_satisfaction()
        self._update_percentiles()

        # 7. Record aggregate outputs.
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
            np.mean([
                agent.income
                for agent in self.agents
            ])
        )

    def mean_rent(self) -> float:
        """Return mean rent across all grid cells."""
        return float(
            np.mean(self.grid.rent.data)
        )

    def mean_neighbor_income(self) -> float:
        """Return the spatial mean of local mean-income values."""
        values = (
            self.grid.mean_neighbor_income.data
        )

        nonempty = values[values > 0.0]

        return (
            float(np.mean(nonempty))
            if nonempty.size > 0
            else 0.0
        )

    def mean_neighbor_income_variance(self) -> float:
        """Return the spatial variance of local income variance values."""
        values = (
            self.grid.neighbor_income_variance.data
        )

        nonempty = values[values > 0.0]

        return (
            float(np.mean(nonempty))
            if nonempty.size > 0
            else 0.0
        )


    def mean_utility(self) -> float:
        """Return mean finite raw utility."""
        values = [
            agent.current_utility
            for agent in self.agents
            if math.isfinite(
                agent.current_utility
            )
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
            np.mean([
                agent.vision_radius
                for agent in self.agents
            ])
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

    def record_game_outcome(
        self,
        game_record,
    ) -> None:
        """Store one application-game result."""

        # Full record for current-step QRE and Nash metrics.
        self.game_records_this_step.append(game_record)

        # Simplified cumulative history for the dashboard plots.
        if self.keep_game_history:
            self.game_history.append(
                {
                    "step": int(game_record.step_number),
                    "agent_id": game_record.agent_id,
                    "income": float(game_record.agent_income),
                    "income_group": int(game_record.income_group),
                    "outcome": game_record.realized_game_outcome,
                }
            )

    def mean_qre_move_probability(self) -> float:
        if not self.game_records_this_step:
            return 0.0

        return float(
            np.mean([
                record.qre.p_move
                for record in self.game_records_this_step
            ])
        )

    def mean_qre_accept_probability(self) -> float:
        if not self.game_records_this_step:
            return 0.0

        return float(
            np.mean([
                record.qre.p_accept
                for record in self.game_records_this_step
            ])
        )

    def mean_ne_move_probability(self) -> float:
        values = [
            record.nash.closest_move_probability
            for record in self.game_records_this_step
            if math.isfinite(
                record.nash.closest_move_probability
            )
        ]

        return (
            float(np.mean(values))
            if values
            else 0.0
        )

    def mean_qre_ne_move_gap(self) -> float:
        values = [
            record.nash.move_probability_gap
            for record in self.game_records_this_step
            if math.isfinite(
                record.nash.move_probability_gap
            )
        ]

        return (
            float(np.mean(values))
            if values
            else 0.0
        )

    def mean_delta_neighborhood_utility(
        self,
    ) -> float:
        if not self.game_records_this_step:
            return 0.0

        return float(
            np.mean([
                record.payoffs
                .neighborhood_change
                .delta_utility
                for record in self.game_records_this_step
            ])
        )

    def ne_following_rate(
        self,
        tolerance: float = 0.05,
    ) -> float:
        """Fraction of games whose QRE move probability is close to an NE.

        A game counts as NE-following when

            |P_QRE(MOVE) - P_NE(MOVE)| <= tolerance.
        """
        valid_records = [
            record
            for record in self.game_records_this_step
            if math.isfinite(
                record.nash.move_probability_gap
            )
        ]

        if not valid_records:
            return 0.0

        following_count = sum(
            record.nash.move_probability_gap
            <= tolerance
            for record in valid_records
        )

        return (
            following_count
            / len(valid_records)
        )

    def _update_percentiles(self):
        incomes = np.array([a.income for a in self.agents])

        self.income_thresholds = np.quantile(
            incomes,
            [0.15, 0.35, 0.55, 0.7, 0.85, 0.99]
        )
        self.agents.do("update_visuals")

    # =================================================================
    # Segregation and Gentrification Metrics
    # =================================================================

    def get_homeless_fraction(self) -> float:
        """Wrapper for homeless_fraction metric."""
        return homeless_fraction(self)

    def get_theil_index(self) -> float:
        """Wrapper for theil_index metric."""
        return theil_index(self)

    def get_moran_i(self) -> float:
        """Wrapper for moran_i metric."""
        return moran_i(self)

    def get_spatial_entropy(self) -> float:
        """Wrapper for spatial_entropy metric."""
        return spatial_entropy(self)

    def get_neighborhood_heterogeneity(self) -> float:
        """Wrapper for neighborhood_heterogeneity metric."""
        return neighborhood_heterogeneity(self)

    def get_income_mobility_indicator(self) -> float:
        """Wrapper for income_mobility_indicator metric."""
        return income_mobility_indicator(self)

    def get_segregation_index(self) -> float:
        """Wrapper for segregation_index metric."""
        return segregation_index(self)

    def get_gentrification_indicator(self) -> float:
        """Wrapper for gentrification_indicator metric."""
        return gentrification_indicator(self)


