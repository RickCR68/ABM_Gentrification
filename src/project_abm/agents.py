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
        agent_type: int,
        homophily: float = 0.4,
    ) -> None:
        super().__init__(model)

        if agent_type not in (0, 1):
            raise ValueError("agent_type must be either 0 or 1.")

        if not 0.0 <= homophily <= 1.0:
            raise ValueError("homophily must lie between 0 and 1.")

        self.cell = cell
        self.type = agent_type
        self.homophily = homophily

        self.happy = False

        # Diagnostic variables for later visualization and analysis.
        self.current_similarity = 0.0
        self.last_destination_similarity: float | None = None
        self.last_acceptance_probability: float | None = None
        self.last_move_accepted: bool | None = None

    def calculate_similarity(self, cell: Cell | None = None) -> float:
        """Calculate similarity around the current or supplied cell."""
        focal_cell = self.cell if cell is None else cell

        neighbors = self.model.neighborhood_definition.get_neighbors(focal_cell)

        # Exclude the agent itself if it appears in the considered region.
        neighbors = [neighbor for neighbor in neighbors if neighbor is not self]

        if not neighbors:
            return 0.0

        similar_count = sum(
            neighbor.type == self.type
            for neighbor in neighbors
        )

        return similar_count / len(neighbors)

    def assign_state(self) -> None:
        """Update whether the agent is satisfied with its current location."""
        self.current_similarity = self.calculate_similarity()
        self.happy = self.current_similarity >= self.homophily

        if self.happy:
            self.model.happy += 1

    def choose_destination(self) -> Cell | None:
        """Select a candidate destination.

        For now, this is a random empty cell. This can later be replaced by
        bounded spatial search, utility maximisation, or a choice model.
        """
        if len(self.model.grid.empties) == 0:
            return None

        return self.model.grid.select_random_empty_cell()

    def attempt_move(self) -> bool:
        """Apply to a destination and move if the neighbourhood accepts."""
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
        """Attempt relocation when dissatisfied."""
        if not self.happy:
            self.model.move_attempts += 1
            self.attempt_move()

    
