from __future__ import annotations

import random
import numpy as np

from mesa import Model
from mesa.datacollection import DataCollector
from mesa.discrete_space import OrthogonalMooreGrid
from mesa.discrete_space import PropertyLayer

from .acceptance import SigmoidSimilarityAcceptance
from .agents import SchellingAgent
from .neighborhoods import MooreNeighborhood
from mesa.experimental.scenarios import Scenario


from .agents import SchellingAgent

class GentrificationModel(Model):
    def __init__(
        self,
        *,
        width: int = 20,
        height: int = 20,
        density: float = 0.8,
        minority_pc: float = 0.2,
        homophily_min: float = 0.2, # lower bound for unfirom dist
        homophily_max: float = 0.6, # upper bound
        alike_neighbors: int = 3,
        neighborhood_radius: int = 1,
        acceptance_midpoint: float = 0.5,
        acceptance_steepness: float = 10.0,
        empty_similarity: float = 0.5,
        rng=None,
    ) -> None:
        super().__init__(rng=rng)

        self._validate_parameters(
            width=width,
            height=height,
            density=density,
            minority_pc=minority_pc,
            homophily_min=homophily_min,
            homophily_max=homophily_max,
        )


        # Model parameters
        self.width = width
        self.height = height
        self.density = density
        self.alike_neighbors = alike_neighbors
        self.minority_pc = minority_pc

        self.homophily_min = homophily_min
        self.homophily_max = homophily_max

        self.grid = OrthogonalMooreGrid(
            dimensions=(width, height),
            torus=True,
            capacity=1,
            random=self.random,
        )

        # Replace either object later without changing the agents.
        self.neighborhood_definition = MooreNeighborhood(
            radius=neighborhood_radius
        )

        # Rent Layer
        self.grid.create_property_layer(
            name='rent',
            default_value=0.5,
            dtype=float,
        )
        self.grid.rent.modify_cells(lambda cell: self.random.uniform(0.0, 1.0))

        self.acceptance_policy = SigmoidSimilarityAcceptance(
            neighborhood=self.neighborhood_definition,
            midpoint=acceptance_midpoint,
            steepness=acceptance_steepness,
            empty_similarity=empty_similarity,
        )

        self.happy = 0
        self.move_attempts = 0
        self.successful_moves = 0
        self.rejected_moves = 0

        self.datacollector = DataCollector(
            model_reporters={
                "happy": "happy",
                "pct_happy": self._percentage_happy,
                "move_attempts": "move_attempts",
                "successful_moves": "successful_moves",
                "rejected_moves": "rejected_moves",
                "acceptance_rate": self._acceptance_rate,
                "mean_similarity": self._mean_similarity,
            },
            agent_reporters={
                "income": "income",
                "happy": "happy",
                # "homophily": "homophily",
                "similarity": "current_similarity",
                "last_acceptance_probability": (
                    "last_acceptance_probability"
                ),
            },
        )

        self._create_agents()
        for agent in self.agents[:5]:  # Print the first 5 agents for debugging
            print(f"Created agent with income {agent.income:.2f}")

        neighborhood_incomes = [[0.0] * self.grid.dimensions[1] for _ in range(self.grid.dimensions[0])]
        for cell in self.grid.all_cells:
            neighbors = self.neighborhood_definition.get_neighbors(cell)
            mean_neighbor_income = np.mean([neighbor.income for neighbor in neighbors]) if neighbors else 0.0
            print(f"Cell has {len(neighbors)} neighbors with mean income {mean_neighbor_income:.2f}")
            neighborhood_incomes[cell.coordinate[0]][cell.coordinate[1]] = mean_neighbor_income

        neighborhood_incomes = np.array(neighborhood_incomes)  # Convert to numpy array for easier handling
        self.grid.add_property_layer(
            PropertyLayer.from_data('mean_neighbor_income', neighborhood_incomes)
        )
        for income in self.grid.mean_neighbor_income.data:
            print(income)

        self._update_agent_states()
        self.datacollector.collect(self)

    @staticmethod
    def _validate_parameters(
        *,
        width: int,
        height: int,
        density: float,
        minority_pc: float,
        homophily_min: float,
        homophily_max: float,
    ) -> None:
        if width <= 0 or height <= 0:
            raise ValueError("width and height must be positive.")

        if not 0.0 <= density <= 1.0:
            raise ValueError("density must lie between 0 and 1.")

        if not 0.0 <= minority_pc <= 1.0:
            raise ValueError("minority_pc must lie between 0 and 1.")

        if not 0.0 <= homophily_min <= 1.0:
            raise ValueError("homophily_min must lie between 0 and 1.")

        if not 0.0 <= homophily_max <= 1.0:
            raise ValueError("homophily_max must lie between 0 and 1.")

        if homophily_min > homophily_max:
            raise ValueError(
                "homophily_min cannot be greater than homophily_max."
            )

    def _create_agents(self) -> None:
        """Populate cells and draw heterogeneous agent preferences."""
        for cell in self.grid.all_cells:
            if self.random.random() >= self.density:
                continue

            # agent_homophily = self.random.uniform(
            #     self.homophily_min,
            #     self.homophily_max,
            # )

            SchellingAgent(
                model=self,
                cell=cell,
                income=self.random.random(),
#                homophily=agent_homophily,
            )

    def _update_agent_states(self) -> None:
        # Collect initial state
        self.agents.do("assign_state")
        self.datacollector.collect(self)

    def step(self) -> None:
        """Advance the model by one time step."""
        self.move_attempts = 0
        self.successful_moves = 0
        self.rejected_moves = 0
        self.happy = 0

        # Agents act in a random order.
        self.agents.shuffle_do("change_income")  # Change all agents income in random order
        self.agents.shuffle_do("step")

        # Evaluate satisfaction after all movement attempts.
        self._update_agent_states()

        self.datacollector.collect(self)

    def _percentage_happy(self) -> float:
        population = len(self.agents)

        if population == 0:
            return 0.0

        return 100.0 * self.happy / population

    def _acceptance_rate(self) -> float:
        if self.move_attempts == 0:
            return 0.0

        return self.successful_moves / self.move_attempts

    def _mean_similarity(self) -> float:
        if len(self.agents) == 0:
            return 0.0

        return sum([agent.current_similarity for agent in self.agents]) / len(self.agents)
