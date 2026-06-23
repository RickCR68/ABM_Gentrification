from __future__ import annotations

from abc import ABC, abstractmethod
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from mesa.discrete_space import Cell

class NeighborhoodDefinition(ABC):
    """Determines which agents belong to a neighborhood."""

    @abstractmethod
    def get_neighbors(self, cell: Cell, include_self=False) -> list:
        """Return the agents living in the given neighborhood.
        :param include_self:
        """
        raise NotImplementedError

class MooreNeighborhood(NeighborhoodDefinition):
    """A Moore neighborhood."""

    def __init__(self, radius: int = 1) -> None:
        if radius < 1:
            raise ValueError("Neighborhood radius must be at least 1.")

        self.radius = radius

    def get_neighbors(self, cell: Cell, include_self=False) -> list:
        neighborhood = cell.get_neighborhood(radius=self.radius, include_center=include_self)
        return list(neighborhood.agents)

# Possible other neighborhoods here depending on what we want?
