import numpy as np
from time import time
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor, as_completed
from tqdm import tqdm
from src.project.model import GentrificationModel

RUNS_PER_SAMPLE = 10  # 10
NUMBER_OF_CONTINUOUS_PARAMETER_VALUES = 10  # AKA N IN SOBOL
STEPS_PER_RUN = 2000 # 200 at first then 2000

# Fixed parameters for the model
GRID_SIZE = 11
DENSITY = 0.95
KEEP_GAME_HISTORY = True
STEPS_UNTIL_SATISFIED = 4  # Only used for visuals
INCOME_GROWTH_SCALING = 0.02  # Follows IRL economy
INCOME_VOLATILITY = 0.05


def continuous_samples(parameter_min, parameter_max, non_negative: bool = True):
    """Generate a list of evenly spaced samples for a continuous parameter."""
    true_min, true_max = min(parameter_min, parameter_max), max(
        parameter_min, parameter_max
    )
    if non_negative:
        if true_max <= 0:
            raise ValueError(
                "For non-negative parameters, the maximum value must be greater than 0."
            )
        if true_min <= 0:
            return np.linspace(
                0, true_max, NUMBER_OF_CONTINUOUS_PARAMETER_VALUES + 1
            )[1:]

    return np.linspace(true_min, true_max, NUMBER_OF_CONTINUOUS_PARAMETER_VALUES)


parameters = {
    "rationality_max": continuous_samples(0.0, 1.0),
    "affordability_share": continuous_samples(0.0, 1.0, non_negative=True),
    "risk_aversion_max": continuous_samples(-1.0, 0.99),
    "discount_factor_max": continuous_samples(0.0, 2.0, non_negative=True),
    "initial_income_max": continuous_samples(0.0, 1.0, non_negative=True),
    "income_similarity_min": continuous_samples(0.0, 2.0, non_negative=True),
    "neighborhood_radius": [1, 2, 3, 4, 5],
}


def run_single_simulation(task_info):
    """Worker function to run a single standalone model simulation."""
    param_name = task_info["param_name"]
    param_value = task_info["param_value"]
    run_idx = task_info["run_idx"]

    # Incorporate baseline dictionary parameters
    params_dict = {
        f"{param_name}": param_value,
        "width": GRID_SIZE,
        "height": GRID_SIZE,
        "density": DENSITY,
        "steps_until_satisfied": STEPS_UNTIL_SATISFIED,
        "income_growth_scaling": INCOME_GROWTH_SCALING,
        "income_volatility": INCOME_VOLATILITY,
        "keep_game_history": KEEP_GAME_HISTORY,
        "initial_income_min": 0.1,
        "rationality_min": 0.0,
    }

    beginning_time = time()

    output_dir = Path(f"results/{param_name}/{param_value}/run_{run_idx}")
    output_dir.mkdir(parents=True, exist_ok=True)

    model_output_file = output_dir / f"model_run_{run_idx}.csv"
    agent_output_file = output_dir / f"agents_run_{run_idx}.csv"
    txt_output_file = output_dir / f"metadata_run_{run_idx}.txt"

    # Instantiate and step the model
    gm = GentrificationModel(**params_dict)
    for _ in range(STEPS_PER_RUN):
        gm.step()

    # Save CSV outputs
    gm.datacollector.get_model_vars_dataframe().to_csv(model_output_file)
    gm.datacollector.get_agent_vars_dataframe().to_csv(agent_output_file)

    execution_time_seconds = time() - beginning_time

    # Save text file with simple stats side-by-side
    with open(txt_output_file, "w", encoding="utf-8") as f:
        f.write(f"Parameter: {param_name}\n")
        f.write(f"Value: {param_value}\n")
        f.write(f"Run: {run_idx}\n")
        f.write(f"Execution Time: {execution_time_seconds:.2f} seconds\n")

    return param_name, param_value, run_idx


if __name__ == "__main__":
    # 1. Map out the full pipeline task configurations
    tasks = []
    for p_name, p_values in parameters.items():
        for p_value in p_values:
            for i in range(RUNS_PER_SAMPLE):
                tasks.append(
                    {"param_name": p_name, "param_value": p_value, "run_idx": i + 1}
                )

    total_tasks = len(tasks)
    print(f"Generated {total_tasks} total simulation tasks.")
    print("Spawning Process Pool... (Sit back, utilizing all CPU cores)")

    # 2. Distribute processes smoothly with an aggregated progress tracker
    with ProcessPoolExecutor() as executor:
        futures = [executor.submit(run_single_simulation, task) for task in tasks]

        with tqdm(
            total=total_tasks, desc="OFAT Iterations", unit="run", dynamic_ncols=True
        ) as pbar:
            for future in as_completed(futures):
                try:
                    p_name, p_val, run = future.result()
                    pbar.set_postfix_str(f"{p_name}={p_val:.2f} (R{run})")
                    pbar.update(1)
                except Exception as e:
                    print(f"\nA worker thread errored out: {e}")
                    pbar.update(1)