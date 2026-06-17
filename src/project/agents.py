from __future__ import annotations

from typing import TYPE_CHECKING

from mesa.discrete_space import CellAgent

if TYPE_CHECKING:
    from mesa.discrete_space import Cell

class SchellingAgent(CellAgent):
    """Household agent in the residential segregation model."""

    def __init__(
        self,
        model,
        cell: Cell,
        income: float,
#        homophily: float = 0.4,
        alike_neighbors: int = 3,
        radius: int = 1
    ) -> None:
        """Create a new (customized) Schelling agent.
        Args:
            model: The model instance the agent belongs to
            income: The agent's income level
            alike_neighbors: Minimum number of similar neighbors needed for happiness
            radius: Search radius for checking neighbor similarity ('layers')
        """

        super().__init__(model)
        self.cell = cell
        self.income = income
        self.alike_neighbours = alike_neighbors
        self.radius = radius
        self.happy = False

    def get_bounds(self):
        """Calculate the lower and upper bounds for neighbor similarity."""
        self.low_bound = max(0, self.income - .25)
        self.high_bound = min(1, self.income + .25)

        # Diagnostic variables for later visualization and analysis.
        self.current_similarity = 0.0
        self.last_destination_similarity: float | None = None
        self.last_acceptance_probability: float | None = None
        self.last_move_accepted: bool | None = None

    def calculate_similarity(self, cell: Cell | None = None) -> float:
        """Calculate similarity around the current or supplied cell."""
        focal_cell = self.cell if cell is None else cell

        neighbors = self.get_neighbors(focal_cell)

        if not neighbors:
            return 0.0

        similar_count = sum(
            neighbor.income == self.income
            for neighbor in neighbors
        )

        return similar_count / len(neighbors)

    def change_income(self) -> None:
        """Change the agent's income based on its neighborhood."""
        if self.happy:
            neighbors = self.get_neighbors()

            reps = [self.income]
            for neighbor in neighbors:
                if neighbor.income > 0:
                    reps.append(neighbor.income)

            size = min(self.alike_neighbours, len(reps))
            reps.sort()
            self.income = sum(reps[:size]) / size
            print(f"Agent at {self.cell.coordinate} has new income {self.income:.2f}")

    def get_neighbors(self, cell: Cell | None = None) -> list[SchellingAgent]:
        """Get neighboring agents within the specified radius."""
        focal_cell = self.cell if cell is None else cell

        neighbors = self.model.neighborhood_definition.get_neighbors(focal_cell)
        neighbors = [neighbor for neighbor in neighbors if neighbor is not self]

        return neighbors

    def is_happy(self) -> bool:
        """Determine if the agent is happy based on its neighbors."""
        neighbors = self.get_neighbors()
        self.get_bounds()
        # Count similar neighbors
        similar_neighbors = len([n for n in neighbors if self.low_bound <= n.income <= self.high_bound])

        return similar_neighbors >= self.alike_neighbours

    def assign_state(self) -> None:
        """Determine if agent is happy and move if necessary."""
        if self.is_happy():
            self.happy = True
            self.model.happy += 1
        else:
            self.happy = False

    def choose_destination(self) -> Cell | None:
        """Select a candidate destination.

        For now, this is a random empty cell. This can later be replaced by
        bounded spatial search, utility maximisation, or a choice model.
        """
        if len(self.model.grid.empties) == 0:
            return None

        return self.model.grid.select_random_empty_cell()

    def attempt_move(self) -> bool:
        """Apply to a destination and move if the neighborhood accepts."""
        destination = self.choose_destination()

        if destination is None:
            self.last_move_accepted = False
            return False

        decision = self.model.acceptance_policy.evaluate(
            agent=self,
            destination=destination,
        )

        self.last_destination_similarity = decision.similarity
        self.last_acceptance_probability = decision.probability
        self.last_move_accepted = decision.accepted

        if decision.accepted:
            self.move_to(destination)
            self.model.successful_moves += 1
            return True

        self.model.rejected_moves += 1
        return False

    def step(self) -> None:
        # Move if unhappy
        if not self.happy:
            self.model.move_attempts += 1
            self.attempt_move()
