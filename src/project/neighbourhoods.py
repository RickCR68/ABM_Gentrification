from __future__ import annotations

from abc import ABC, abstractmethod
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from mesa.discrete_space import Cell


class NeighborhoodDefinition(ABC):
    """Determines which agents belong to a neighbourhood."""

    @abstractmethod
    def get_neighbors(self, cell: Cell) -> list:
        """Return the agents living in the given neighbourhood."""
        raise NotImplementedError


class MooreNeighborhood(NeighborhoodDefinition):
    """A Moore neighbourhood."""

    def __init__(self, radius: int = 1) -> None:
        if radius < 1:
            raise ValueError("Neighborhood radius must be at least 1.")

        self.radius = radius

    def get_neighbors(self, cell: Cell) -> list:
        neighborhood = cell.get_neighborhood(radius=self.radius)
        return list(neighborhood.agents)
    

# Possible other neighbourhoods here depending on what we want?