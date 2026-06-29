import numpy as np
import os
from time import time
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor
from tqdm import tqdm
from src.project.model import GentrificationModel

RUNS_PER_SAMPLE = 20 # 10
NUMBER_OF_CONTINUOUS_PARAMETER_VALUES = 16 # AKA N IN SOBOL
STEPS_PER_RUN = 2000 # 200 at first then 2000
MAX_WORKERS = os.cpu_count() or 1

# Fixed parameters for the model
GRID_SIZE = 11
DENSITY = 0.95
KEEP_GAME_HISTORY = False
KEEP_AGENTS = False
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
    "rationality_max": np.round(continuous_samples(0.0, 1.0), 5),
    "affordability_share": np.round(continuous_samples(0.0, 1.0, non_negative=True), 5),
    "risk_aversion_max": np.round(continuous_samples(-1.0, 0.99), 5),
    "discount_factor_max": np.round(continuous_samples(0.0, 2.0, non_negative=True), 5),
    "initial_income_max": np.round(continuous_samples(0.0, 1.0, non_negative=True), 5),
    "income_similarity_min": np.round(continuous_samples(0.0, 2.0, non_negative=True), 5),
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
        "keep_agents": KEEP_AGENTS,
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


def run_task_batch(task_batch):
    """Run a batch of simulation tasks inside one worker process."""
    batch_results = []

    for task_info in task_batch:
        try:
            batch_results.append(
                (
                    "ok",
                    run_single_simulation(task_info),
                )
            )
        except Exception as exc:
            batch_results.append(
                (
                    "error",
                    task_info,
                    repr(exc),
                )
            )

    return batch_results


def batch_tasks(tasks, batch_size):
    """Yield contiguous task batches of roughly equal size."""
    for start in range(0, len(tasks), batch_size):
        yield tasks[start : start + batch_size]


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
    print(
        f"Spawning Process Pool with {MAX_WORKERS} workers... (Sit back, utilizing all CPU cores)"
    )

    batch_size = max(1, total_tasks // (MAX_WORKERS * 4))
    task_batches = list(batch_tasks(tasks, batch_size))

    # 2. Distribute processes smoothly with an aggregated progress tracker
    with ProcessPoolExecutor(max_workers=MAX_WORKERS) as executor:
        with tqdm(
            total=total_tasks,
            desc="OFAT Iterations",
            unit="run",
            dynamic_ncols=True,
        ) as pbar:
            for batch_results in executor.map(run_task_batch, task_batches):
                for result in batch_results:
                    if result[0] == "ok":
                        p_name, p_val, run = result[1]
                        pbar.set_postfix_str(f"{p_name}={p_val:.2f} (R{run})")
                    else:
                        task_info = result[1]
                        error_message = result[2]
                        print(
                            f"\nA worker thread errored out for {task_info['param_name']}={task_info['param_value']} "
                            f"(R{task_info['run_idx']}): {error_message}"
                        )
                    pbar.update(1)