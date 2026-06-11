from mesa.discrete_space import CellAgent

import random

class SchellingAgent(CellAgent):
    """Schelling segregation agent."""

    def __init__(
        self, model, cell, agent_type: int, alike_neighbors: int = 3, radius: int = 1
    ) -> None:
        """Create a new Schelling agent.
        Args:
            model: The model instance the agent belongs to
            agent_type: Indicator for the agent's type (minority=1, majority=0)
            alike_neighbors: Minimum number of similar neighbors needed for happiness
            radius: Search radius for checking neighbor similarity
        """
        super().__init__(model)
        self.cell = cell
        self.type = agent_type
        self.alike_neighbours = alike_neighbors
        self.radius = radius
        self.happy = False
        self.low_bound = max(0, self.type - .05)
        self.high_bound = min(1, self.type + .20)

    def change_reputation(self) -> None:
        """Change the agent's reputation based on its type."""
        if self.happy:
            neighbors = self.get_neighbors()

            reps = [self.type]
            for neighbor in neighbors:
                if neighbor.type > 0:
                    reps.append(neighbor.type)

            size = min(self.alike_neighbours, len(reps))
            reps.sort()
            self.type = sum(reps[:size]) / size
            print(f"Agent at {self.cell.coordinate} has new type {self.type:.2f}")

    def get_neighbors(self):
        """Get neighboring agents within the specified radius."""
        return list(self.cell.get_neighborhood(radius=self.radius).agents)

    def is_happy(self) -> bool:
        """Determine if the agent is happy based on its neighbors."""
        neighbors = self.get_neighbors()

        # Count similar neighbors
        similar_neighbors = len([n for n in neighbors if self.low_bound <= n.type <= self.high_bound])

        return similar_neighbors >= self.alike_neighbours

    def assign_state(self) -> None:
        """Determine if agent is happy and move if necessary."""
        if self.is_happy():
            self.happy = True
            self.model.happy += 1
        else:
            self.happy = False

    def step(self) -> None:
        # Move if unhappy
        if not self.happy:
            self.cell = self.model.grid.select_random_empty_cell()
