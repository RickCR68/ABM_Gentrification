import numpy as np
from time import time
from pathlib import Path
from src.project.model import GentrificationModel

RUNS_PER_SAMPLE = 4 # 10
NUMBER_OF_CONTINUOUS_PARAMETER_VALUES = 8 # AKA N IN SOBOL
STEPS_PER_RUN = 100 # 200 at first then 2000

# Fixed parameters for the model
GRID_SIZE = 11
DENSITY = 0.95
KEEP_GAME_HISTORY = True
STEPS_UNTIL_SATISFIED = 4 # Only used for visuals
INCOME_GROWTH_SCALING = 0.02 # Follows IRL economy
INCOME_VOLATILITY = 0.05

def continuous_samples(parameter_min, parameter_max, non_negative: bool = True):
    """Generate a list of evenly spaced samples for a continuous parameter."""
    # check correct min and max ordering
    true_min, true_max = min(parameter_min, parameter_max), max(parameter_min, parameter_max)
    if non_negative:
        if true_max <= 0:
            raise ValueError("For non-negative parameters, the maximum value must be greater than 0.")
        if true_min <= 0:
            return np.linspace(0, true_max, NUMBER_OF_CONTINUOUS_PARAMETER_VALUES + 1)[1:]  # Exclude the first sample (0) to avoid non-positive values

    return np.linspace(true_min, true_max, NUMBER_OF_CONTINUOUS_PARAMETER_VALUES)

parameters = {
    # Agent-Specific Parameters
    #
    'neighborhood_radius': [1, 2, 3, 4, 5],
    'affordability_share': continuous_samples(0.0, 1.0, non_negative=True),
    'income_similarity_min': continuous_samples(0.0, 2.0, non_negative=True),
    'initial_income_max': continuous_samples(0.0, 1.0, non_negative=True),
    'discount_factor_max': continuous_samples(0.0, 2.0, non_negative=True),
    'risk_aversion_max': continuous_samples(-1.0, 1.0),
    'rationality_max': continuous_samples(0.0, 1.0),

    # Model-Wide Parameters - SOBOL - no need to test
    #
    # 'satisficing_threshold': continuous_samples(0.0, 1.0),
    # 'rent_adjustment_rate': continuous_samples(0.0, 0.2),
    # 'vision_income_scale': continuous_samples(0.0, 2.0, non_negative=True),
    # 'moving_cost': 0.05,
    # 'rejection_cost': 0.05,

    # Neighborhood/Game Theory Parameters ==> Remember to discuss if these stay
    #
    # neighborhood_risk_aversion: float = 0.5,
    # neighborhood_rationality: float = 5.0,
    # qre_tolerance: float = 1e-10,
    # qre_maximum_iterations: int = 1000,
    # qre_damping: float = 0.5,
}

for param_name, param_values in parameters.items():
    print(f"Parameter: {param_name}, Values: {param_values}")
    output_dir = Path(f"results/{param_name}")
    for param_value in param_values:
        print(f"  Running with {param_name} = {param_value}")
        params_dict = {
            f'{param_name}': param_value,
            'width': GRID_SIZE,
            'height': GRID_SIZE,
            'density': DENSITY,
            'steps_until_satisfied': STEPS_UNTIL_SATISFIED,
            'income_growth_scaling': INCOME_GROWTH_SCALING,
            'income_volatility': INCOME_VOLATILITY,
            'keep_game_history': KEEP_GAME_HISTORY
        }
        for i in range(RUNS_PER_SAMPLE):
            begining_time = time()
            gm = GentrificationModel(
                **params_dict
            )

            for _ in range(STEPS_PER_RUN):
                gm.step()
            gm.datacollector.get_model_vars_dataframe().to_csv(
                f"results/{param_name}/{param_value}_model_run_{i}.csv"
            )
            gm.datacollector.get_agent_vars_dataframe().to_csv(
                f"results/{param_name}/{param_value}_agents_run_{i}.csv"
            )
            print(f"    Run {i+1}/{RUNS_PER_SAMPLE} completed in {time() - begining_time:.2f} seconds.")
