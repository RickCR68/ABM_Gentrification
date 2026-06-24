"""
Segregation and Gentrification Metrics for Housing Market ABM

This module provides functions to compute various metrics for analyzing segregation,
gentrification, and inequality in the agent-based housing model. All functions are
pure functions that operate on agent/grid data, making them reusable for both
real-time simulation collection and post-hoc CSV analysis.

Mathematical references and interpretations:
- Moran's I: Spatial autocorrelation, ranges from -1 (dispersed) to 1 (clustered)
- Theil's Index: Entropy-based inequality measure, ranges from 0 (equal) to ln(n) (max inequality)
- Spatial Entropy: Normalized measure of spatial heterogeneity, ranges from 0 to 1
- Neighborhood Heterogeneity: Local income diversity, high values indicate mixed neighborhoods
"""

from __future__ import annotations

import math
from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    from .model import GentrificationModel


def homeless_fraction(model: GentrificationModel) -> float:
    """
    Calculate the fraction of agents who cannot afford their current rent.

    An agent is considered homeless if:
        income < rent * affordability_share

    This metric indicates how many agents are financially trapped in their
    current locations and likely to be forced to move.

    Args:
        model: The GentrificationModel instance

    Returns:
        Fraction of agents who are homeless (0 to 1)
    """
    if len(model.agents) == 0:
        return 0.0

    homeless_count = 0
    for agent in model.agents:
        rent = agent.cell.rent
        max_affordable_rent = agent.income * model.affordability_share
        if agent.income < rent * model.affordability_share:
            homeless_count += 1

    return homeless_count / len(model.agents)


def theil_index(model: GentrificationModel) -> float:
    """
    Calculate Theil's Index (entropy-based inequality measure).

    Theil's Index is based on information theory and measures relative inequality:
        T = (1/N) * Σ(y_i / mean_y) * ln(y_i / mean_y)

    Range: 0 (perfect equality) to ln(N) (maximum inequality)

    This metric is useful for tracking inequality deepening during gentrification,
    as richer and poorer households diverge.

    Args:
        model: The GentrificationModel instance

    Returns:
        Theil's Index value
    """
    incomes = np.array([agent.income for agent in model.agents])

    if len(incomes) == 0 or np.sum(incomes) == 0:
        return 0.0

    mean_income = np.mean(incomes)
    if mean_income == 0:
        return 0.0

    # Avoid division by zero and log(0)
    normalized_incomes = incomes / mean_income
    normalized_incomes = normalized_incomes[normalized_incomes > 0]

    theil = np.mean(normalized_incomes * np.log(normalized_incomes))
    return float(theil)


def moran_i(model: GentrificationModel, attribute: str = "income") -> float:
    """
    Calculate global Moran's I spatial autocorrelation.

    Moran's I measures whether similar values cluster spatially:
        I = (N / Σw_ij) * Σ_i Σ_j w_ij (z_i * z_j) / Σ_i (z_i)^2

    where:
        - w_ij is the spatial weight (1 if neighbors, 0 otherwise)
        - z_i is the deviation from mean of attribute
        - N is the number of agents

    Range: -1 (perfect dispersion) to 1 (perfect clustering)
    - I > 0: similar values cluster (segregation)
    - I < 0: dissimilar values cluster (integration)
    - I ≈ 0: random spatial distribution

    Uses Moore neighborhood (8-connected) based on model configuration.

    Args:
        model: The GentrificationModel instance
        attribute: Attribute to compute I on ('income' is recommended)

    Returns:
        Moran's I value
    """
    if len(model.agents) == 0:
        return 0.0

    # Build position to agent map
    position_to_agent = {}
    for agent in model.agents:
        pos = agent.cell.coordinate
        position_to_agent[pos] = agent

    if len(position_to_agent) < 2:
        return 0.0

    # Get attribute values
    positions = list(position_to_agent.keys())
    values = np.array([
        getattr(position_to_agent[pos], attribute)
        for pos in positions
    ])

    if np.std(values) == 0:
        return 0.0

    # Standardize values (deviations from mean)
    z = (values - np.mean(values)) / np.std(values)

    # Compute spatial weights using Moore neighborhood
    n = len(positions)
    pos_to_idx = {pos: i for i, pos in enumerate(positions)}

    width = model.width
    height = model.height

    # Count spatial relationships and compute I
    numerator = 0.0
    weight_sum = 0.0

    for i, pos_i in enumerate(positions):
        x_i, y_i = pos_i

        # Get Moore neighbors (8-connected with torus wrapping)
        for dx in [-1, 0, 1]:
            for dy in [-1, 0, 1]:
                if dx == 0 and dy == 0:
                    continue

                x_j = (x_i + dx) % width
                y_j = (y_i + dy) % height
                pos_j = (x_j, y_j)

                if pos_j in pos_to_idx:
                    j = pos_to_idx[pos_j]
                    w_ij = 1.0  # Moore neighbor
                    numerator += w_ij * z[i] * z[j]
                    weight_sum += w_ij

    if weight_sum == 0:
        return 0.0

    denominator = np.sum(z ** 2)
    if denominator == 0:
        return 0.0

    moran = (n / weight_sum) * (numerator / denominator)
    return float(moran)


def spatial_entropy(model: GentrificationModel) -> float:
    """
    Calculate normalized spatial entropy of income distribution.

    Entropy measures the heterogeneity/disorder in spatial distribution:
        H = -Σ p_i * log(p_i)

    Normalized by log(n) to give range [0, 1]:
        H_norm = H / log(n)

    Range: 0 (homogeneous/segregated) to 1 (heterogeneous/well-mixed)

    This metric captures whether neighborhoods have similar (low entropy, segregation)
    or diverse (high entropy, integration) income compositions.

    Implementation: Divides grid into equal-size regions and computes entropy
    of income quartile distribution within each region.

    Args:
        model: The GentrificationModel instance

    Returns:
        Normalized spatial entropy (0 to 1)
    """
    if len(model.agents) == 0:
        return 0.0

    # Get income quartiles for all agents
    incomes = np.array([agent.income for agent in model.agents])
    if len(incomes) < 4:
        return 0.0

    quartiles = np.quantile(incomes, [0.25, 0.5, 0.75])

    # Classify agents into income groups (4 groups: 0, 1, 2, 3).
    # `np.digitize` keeps this vectorized and preserves the same quartile split.
    income_groups = np.digitize(incomes, quartiles, right=True)

    # Divide grid into regions (e.g., 4x4 regions for computational efficiency)
    num_regions_per_side = max(2, model.width // 5)
    region_width = model.width / num_regions_per_side
    region_height = model.height / num_regions_per_side

    # Count income group distributions per region.
    region_distributions = {}

    for agent, group in zip(model.agents, income_groups):
        x, y = agent.cell.coordinate
        region_x = int(x / region_width)
        region_y = int(y / region_height)
        region_id = (min(region_x, num_regions_per_side - 1),
                     min(region_y, num_regions_per_side - 1))

        if region_id not in region_distributions:
            region_distributions[region_id] = [0, 0, 0, 0]
        region_distributions[region_id][group] += 1

    # Compute entropy for each region
    total_entropy = 0.0
    num_non_empty_regions = 0

    for counts in region_distributions.values():
        total_in_region = sum(counts)
        if total_in_region == 0:
            continue

        # Compute Shannon entropy
        entropy = 0.0
        for count in counts:
            if count > 0:
                p = count / total_in_region
                entropy -= p * np.log(p)

        # Normalize by log(4) for 4 income groups
        entropy /= np.log(4)
        total_entropy += entropy
        num_non_empty_regions += 1

    if num_non_empty_regions == 0:
        return 0.0

    # Average entropy across regions
    avg_entropy = total_entropy / num_non_empty_regions
    return float(avg_entropy)


def neighborhood_heterogeneity(model: GentrificationModel) -> float:
    """
    Calculate average neighborhood income heterogeneity (diversity).

    For each agent's neighborhood, compute the coefficient of variation (CV) of incomes:
        CV = std_dev / mean_income

    Then average across all neighborhoods.

    Range: 0 (all identical incomes, segregation) to ∞ (high variation)

    High values indicate well-mixed neighborhoods. This metric directly captures
    the degree of economic segregation at the local level.

    Args:
        model: The GentrificationModel instance

    Returns:
        Average neighborhood heterogeneity
    """
    if len(model.agents) == 0:
        return 0.0

    heterogeneities = []

    for agent in model.agents:
        neighbors = model.neighborhood_definition.get_neighbors(
            agent.cell
        )

        if not neighbors:
            continue

        neighbor_incomes = [
            neighbor.income
            for neighbor in neighbors
        ]
        neighbor_incomes.append(agent.income)

        mean_income = float(np.mean(neighbor_incomes))

        if mean_income > 0.0:
            cv = float(np.std(neighbor_incomes)) / mean_income
            heterogeneities.append(cv)

    if not heterogeneities:
        return 0.0

    return float(np.mean(heterogeneities))


def income_mobility_indicator(model: GentrificationModel) -> float:
    """
    Calculate income mobility indicator based on recent moves.

    This is a proxy measure: agents who have moved recently and whose income
    differs significantly from their previous neighborhood's mean income
    are considered mobile.

    The indicator is the fraction of agents who moved in the last step
    (may be extended to track multi-step history).

    Range: 0 to 1

    Note: This is a simplified implementation. For true income mobility tracking,
    agents would need historical income records.

    Args:
        model: The GentrificationModel instance

    Returns:
        Indicator of income mobility (fraction of agents who moved last step)
    """
    if len(model.agents) == 0:
        return 0.0

    moved_count = sum(
        1 for agent in model.agents
        if agent.moved_this_step
    )

    return moved_count / len(model.agents)


def segregation_index(model: GentrificationModel) -> float:
    """
    Calculate Dissimilarity Index (D) for residential segregation.

    The Dissimilarity Index measures the degree to which two groups are
    evenly distributed across neighborhoods:

        D = (1/2) * Σ |p_i - q_i|

    where p_i and q_i are the proportions of group members in area i.

    For income, we define two groups:
    - Low income: below median
    - High income: above median

    Range: 0 (perfect integration) to 1 (perfect segregation)

    Args:
        model: The GentrificationModel instance

    Returns:
        Dissimilarity Index value
    """
    if len(model.agents) < 2:
        return 0.0

    incomes = np.array([agent.income for agent in model.agents])
    median_income = np.median(incomes)

    # Classify agents as low or high income
    low_income_agents = []
    high_income_agents = []

    for agent in model.agents:
        if agent.income <= median_income:
            low_income_agents.append(agent)
        else:
            high_income_agents.append(agent)

    if len(low_income_agents) == 0 or len(high_income_agents) == 0:
        return 0.0

    n_low = len(low_income_agents)
    n_high = len(high_income_agents)

    # Compute neighborhoods and their compositions
    # Use coarse grid division for efficiency
    num_regions_per_side = max(2, model.width // 4)
    region_width = model.width / num_regions_per_side
    region_height = model.height / num_regions_per_side

    region_counts = {}  # (region_id, group) -> count

    for agent in low_income_agents:
        x, y = agent.cell.coordinate
        region_x = int(x / region_width)
        region_y = int(y / region_height)
        region_id = (min(region_x, num_regions_per_side - 1),
                     min(region_y, num_regions_per_side - 1))
        key = (region_id, 'low')
        region_counts[key] = region_counts.get(key, 0) + 1

    for agent in high_income_agents:
        x, y = agent.cell.coordinate
        region_x = int(x / region_width)
        region_y = int(y / region_height)
        region_id = (min(region_x, num_regions_per_side - 1),
                     min(region_y, num_regions_per_side - 1))
        key = (region_id, 'high')
        region_counts[key] = region_counts.get(key, 0) + 1

    # Compute D index
    dissimilarity = 0.0
    for region_id in set(r_id for r_id, _ in region_counts.keys()):
        low_in_region = region_counts.get((region_id, 'low'), 0)
        high_in_region = region_counts.get((region_id, 'high'), 0)

        p_i = low_in_region / n_low if n_low > 0 else 0
        q_i = high_in_region / n_high if n_high > 0 else 0

        dissimilarity += abs(p_i - q_i)

    d_index = dissimilarity / 2
    return float(d_index)


def gentrification_indicator(model: GentrificationModel) -> float:
    """
    Calculate a simple gentrification indicator.

    Gentrification is characterized by:
    1. Rising rents
    2. Rising neighborhood incomes
    3. Displacement of lower-income residents

    This indicator combines: (rent growth + income growth) / 2
    normalized by their recent rates.

    Note: This is a simplified proxy. True gentrification tracking would require
    long-term historical data on neighborhood composition changes.

    Range: Unbounded, but typically 0-1 during normal operation

    Args:
        model: The GentrificationModel instance

    Returns:
        Gentrification indicator value
    """
    mean_rent = model.mean_rent()
    mean_income = model.city_mean_income()

    if mean_rent == 0 or mean_income == 0:
        return 0.0

    # Normalize rent to income ratio
    rent_income_ratio = mean_rent / mean_income

    # Heuristic: gentrification active when ratio is rising
    # For MVP, we return a simple indicator based on rent adjustment dynamics
    # A proper implementation would track historical data

    gentrif_indicator = (
        model.rent_adjustment_rate * mean_rent +
        model.income_growth_scaling * mean_income
    ) / 2

    return float(gentrif_indicator)


