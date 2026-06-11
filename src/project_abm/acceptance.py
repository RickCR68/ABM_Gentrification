from __future__ import annotations

import math
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from mesa.discrete_space import Cell

    from agents import SchellingAgent
    from neighbourhoods import NeighborhoodDefinition


@dataclass(frozen=True)
class AcceptanceDecision:
    """Information produced by a neighbourhood acceptance decision."""

    accepted: bool
    # mostly for later extensions or if we want for graphs for some reason
    probability: float
    similarity: float
    neighbor_count: int


class AcceptancePolicy(ABC):
    """Interface for neighbourhood acceptance mechanisms."""

    @abstractmethod
    def evaluate(
        self,
        agent: SchellingAgent,
        destination: Cell,
    ) -> AcceptanceDecision:
        """Evaluate whether an agent is accepted by a destination."""
        raise NotImplementedError



# for now its just this. later we do game.
class SigmoidSimilarityAcceptance(AcceptancePolicy):
    """Accept agents probabilistically according to local similarity."""

    def __init__(
        self,
        neighborhood: NeighborhoodDefinition,
        midpoint: float = 0.5, # the similarity at which acceptance probability is 0.5
        steepness: float = 10.0, # how quickly probability changes around midpoint
        empty_similarity: float = 0.5, # prob for when neighbourhood is empty
    ) -> None:
        if not 0.0 <= midpoint <= 1.0:
            raise ValueError("midpoint must lie between 0 and 1.")

        if steepness <= 0:
            raise ValueError("steepness must be positive.")

        if not 0.0 <= empty_similarity <= 1.0:
            raise ValueError("empty_similarity must lie between 0 and 1.")

        self.neighborhood = neighborhood
        self.midpoint = midpoint
        self.steepness = steepness
        self.empty_similarity = empty_similarity

    def evaluate(
        self,
        agent: SchellingAgent,
        destination: Cell,
    ) -> AcceptanceDecision:
        residents = self.neighborhood.get_neighbors(destination)

        # The mover may currently live next to the prospective destination.
        # Exclude them so that they do not influence their own acceptance.
        residents = [resident for resident in residents if resident is not agent]

        similarity = self._calculate_similarity(agent, residents)
        probability = self._acceptance_probability(similarity)

        accepted = agent.model.random.random() < probability

        return AcceptanceDecision(
            accepted=accepted,
            probability=probability,
            similarity=similarity,
            neighbor_count=len(residents),
        )

    def _calculate_similarity(
        self,
        agent: SchellingAgent,
        residents: list,
    ) -> float:
        if not residents:
            return self.empty_similarity

        similar_count = sum(
            resident.type == agent.type
            for resident in residents
        )

        return similar_count / len(residents)

    def _acceptance_probability(self, similarity: float) -> float:
        exponent = -self.steepness * (similarity - self.midpoint)

        # Avoid numerical overflow if parameters later become very large.
        exponent = max(min(exponent, 700), -700)

        return 1.0 / (1.0 + math.exp(exponent))
    


#  Add our masgical game here later. 