import numpy as np

from src.project.model import GentrificationModel

ITERATION_REPETITIONS = 4 # 10
SAMPLES_PER_CONTINUOUS_PARAMETER = 8
STEPS_PER_ITERATION = 100 # 200 at first then 2000

# Fixed parameters for the model
GRID_SIZE = 11
DENSITY = 0.95
KEEP_GAME_HISTORY = True
STEPS_UNTIL_SATISFACTION = 4 # Only used for visuals

def continuous_samples(parameter_min, parameter_max, non_negative: bool = True):
    """Generate a list of evenly spaced samples for a continuous parameter."""
    # check correct min and max ordering
    true_min, true_max = min(parameter_min, parameter_max), max(parameter_min, parameter_max)
    if non_negative:
        if true_max <= 0:
            raise ValueError("For non-negative parameters, the maximum value must be greater than 0.")
        if true_min <= 0:
            return np.linspace(0, true_max, SAMPLES_PER_CONTINUOUS_PARAMETER + 1)[1:]  # Exclude the first sample (0) to avoid non-positive values

    return np.linspace(true_min, true_max, SAMPLES_PER_CONTINUOUS_PARAMETER)

parameters = {
    # Model-Wide Parameters
    'neighborhood_radius': [1, 2, 3, 4, 5],
    'affordability_share': continuous_samples(0.0, 1.0, non_negative=True),
    'rent_adjustment_rate': continuous_samples(0.0, 0.2),
    'satisficing_threshold': continuous_samples(0.0, 1.0),
    'vision_income_scale': continuous_samples(0.0, 2.0, non_negative=True),

    # Agent-Specific Parameters - SOBOL
    'initial_income_max': continuous_samples(0.0, 1.0, non_negative=True),
    'discount_factor_max': continuous_samples(0.0, 2.0, non_negative=True),
    'risk_aversion_max': continuous_samples(-1.0, 1.0),
    'rationality_max': continuous_samples(0.0, 1.0),
    'income_similarity_min': continuous_samples(0.0, 2.0, non_negative=True),

    # income_growth_scaling: float = 0.01,
    # income_volatility: float = 0.05,

    # moving_cost: float = 0.05,
    # rejection_cost: float = 0.05,

    # neighborhood_risk_aversion: float = 0.5,
    # neighborhood_rationality: float = 5.0,

    # qre_tolerance: float = 1e-10,
    # qre_maximum_iterations: int = 1000,
    # qre_damping: float = 0.5,
}