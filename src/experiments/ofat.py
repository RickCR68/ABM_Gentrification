from mesa import batch_run
from SALib.sample import sobol

import numpy as np

from src.project.model import GentrificationModel

if __name__ == "__main__":
    # We define our variables and bounds
    params = {
        'income_similarity_max': [0.00, 3.00],  # income similatity     => Theta
        'risk_aversion_max': [0.00, 5.00],      # risk aversion         => Rho
        'discount_factor_max': [0.00, 1.00],    # Future Orientation    => Beta
        'rationality_max': [5.00, 20.00],       # Rationality           => Gamma
        'vision_income_scale': [0, 20],         # initial vision radius => v
    }

    problem = {
        'num_vars': 5,
        'names': list(params.keys()),
        'bounds': list(params.values())
    }

    max_steps = 10
    distinct_samples = 2

    data = {}

    for i, var in enumerate(problem['names']):
        # Get the bounds for this variable and get <distinct_samples> samples within this space (uniform)
        samples = np.linspace(*problem['bounds'][i], num=distinct_samples)

        # Keep in mind that wolf_gain_from_food should be integers. You will have to change
        # your code to acommodate for this or sample in such a way that you only get integers.

        batch = batch_run(
            GentrificationModel,
            parameters={ var: samples },
            number_processes=None,
            rng=4,
            max_steps=max_steps,
            display_progress=True
        )

        data[var] = batch.get_model_vars_dataframe()

    print(data)