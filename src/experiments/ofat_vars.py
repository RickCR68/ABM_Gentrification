import os
import time
import numpy as np
import pandas as pd
from concurrent.futures import ProcessPoolExecutor, wait, FIRST_COMPLETED
from tqdm import tqdm

# Treat this import as your local project hook
from src.project.model import GentrificationModel

# --- 1. Global Configurations ---
MIN_RUNS = 15               # Baseline run floor before testing convergence
MAX_RUNS = 80               # Hard cutoff ceiling per parameter value
CONVERGENCE_THRESHOLD = 0.01 # 1% maximum variance shift allowed
STABILITY_WINDOW = 3        # Must pass the 1% test 3 times consecutively
STEPS_PER_RUN = 2000
MAX_WORKERS = os.cpu_count() or 1

TARGET_METRICS = [
    "city_mean_income", "mean_neighbor_income", "mean_neighbor_income_variance",
    "mean_rent", "mean_utility", "mean_value", "rent_income_timescale_ratio",
    "gini_coefficient", "theil_index", "moran_i", "segregation_index",
    "homeless_fraction", "spatial_entropy", "neighborhood_heterogeneity",
    "income_mobility_indicator", "gentrification_indicator"
]

BASELINE_PARAMS = {
    "width": 11, "height": 11, "density": 0.95, "neighborhood_radius": 2,
    "rationality_max": 0.5, "affordability_share": 0.3, "risk_aversion_max": 0.0,
    "discount_factor_max": 1.0, "initial_income_max": 0.5, "income_similarity_min": 1.0,
    "income_growth_scaling": 0.02, "income_volatility": 0.05, "steps_until_satisfied": 4,
    "keep_game_history": False, "keep_agents": False, "initial_income_min": 0.1, "rationality_min": 0.0,
}

# --- 2. Define the Parameters to Sweep (OFAT Design) ---
OFAT_SWEEP_SPACE = {
    "density": [0.95],
    "income_growth_scaling": [0.01]
}

# --- 3. Parallel Worker Function ---
def run_simulation_worker(param_setup, seed):
    """Executes a single model configuration and extracts final step metrics."""
    run_params = BASELINE_PARAMS.copy()
    run_params.update(param_setup)

    model = GentrificationModel(**run_params)
    for _ in range(STEPS_PER_RUN):
        model.step()

    df_vars = model.datacollector.get_model_vars_dataframe()

    result_row = {metric: df_vars[metric].iloc[-1] for metric in TARGET_METRICS}
    result_row["_seed"] = seed
    return result_row


# --- 4. Main Executive Routing Code ---
if __name__ == "__main__":
    print(f"=================================================================")
    print(f"LAUNCHING FLUID ADAPTIVE OFAT SWEEP")
    print(f"System Context: {MAX_WORKERS} cores | Targets: {len(OFAT_SWEEP_SPACE)} variables")
    print(f"Adaptive Range: Minimum {MIN_RUNS} runs -> Maximum {MAX_RUNS} runs")
    print(f"=================================================================\n")

    start_wall_time = time.time()
    ofat_final_summary_data = []

    sweep_coordinates = [(p, val) for p, vals in OFAT_SWEEP_SPACE.items() for val in vals]

    # Initialize one long-lived high-priority process pool
    with ProcessPoolExecutor(max_workers=MAX_WORKERS) as executor:

        for param_name, target_value in sweep_coordinates:
            current_coordinate_label = f"{param_name} @ {target_value}"
            current_setup = {param_name: target_value}

            collected_runs = []
            pending_futures = set()
            seed_counter = 0
            is_stable = False
            stable_counter = 0

            print(f"🧪 Evaluating Coordinate: {current_coordinate_label}")

            # Prime the pump: Fill all available cores immediately up to our minimum floor
            initial_batch_size = max(MIN_RUNS, MAX_WORKERS)
            for _ in range(initial_batch_size):
                if seed_counter < MAX_RUNS:
                    f = executor.submit(run_simulation_worker, current_setup, seed_counter)
                    pending_futures.add(f)
                    seed_counter += 1

            # Unified fluid execution loop
            with tqdm(
                total=MAX_RUNS,
                desc=f"  └─ Sampling Convergence",
                unit="run",
                leave=False
            ) as pbar:

                while pending_futures:
                    # Block ONLY until the single fastest worker finishes (event-driven)
                    done, pending_futures = wait(pending_futures, return_when=FIRST_COMPLETED)

                    for future in done:
                        collected_runs.append(future.result())
                        pbar.update(1)

                    # Evaluate stability metrics once the floor requirement is met
                    if len(collected_runs) >= MIN_RUNS and not is_stable:
                        df_current = pd.DataFrame(collected_runs)

                        # Compute variance shifts across expanding execution rows
                        expanding_std = df_current[TARGET_METRICS].expanding(min_periods=3).std()
                        current_std = expanding_std.iloc[-1]
                        previous_std = expanding_std.iloc[-2]

                        relative_variance_shift = (current_std - previous_std).abs() / (previous_std + 1e-8)
                        worst_metric_fluctuation = relative_variance_shift.max()

                        pbar.set_postfix_str(f"Worst Fluc: {worst_metric_fluctuation:.2%}")

                        if worst_metric_fluctuation <= CONVERGENCE_THRESHOLD:
                            stable_counter += 1
                        else:
                            stable_counter = 0

                        if stable_counter >= STABILITY_WINDOW:
                            is_stable = True
                            pbar.write(f"  ✨ Converged early at {len(collected_runs)} runs (Worst fluctuation dropped below {CONVERGENCE_THRESHOLD:.0%})")
                            # We stop feeding the loop here. The remaining active tasks in
                            # pending_futures will drain naturally without spawning replacements.

                    # Feed the Beast: Keep cores 100% saturated if we haven't hit stability or limits
                    while len(pending_futures) < MAX_WORKERS and seed_counter < MAX_RUNS and not is_stable:
                        f = executor.submit(run_simulation_worker, current_setup, seed_counter)
                        pending_futures.add(f)
                        seed_counter += 1

            # --- Compile Coordinate Metrics ---
            # Any mid-flight runs that completed during the drain phase are safely included
            df_coordinate_set = pd.DataFrame(collected_runs)[TARGET_METRICS]
            mean_metrics = df_coordinate_set.mean().to_dict()

            mean_metrics["parameter_name"] = param_name
            mean_metrics["parameter_value"] = target_value
            mean_metrics["runs_required"] = len(collected_runs)
            ofat_final_summary_data.append(mean_metrics)

    # --- 5. Output Packaging ---
    df_ofat_master = pd.DataFrame(ofat_final_summary_data)
    total_elapsed = time.time() - start_wall_time

    print("\n=================================================================")
    print(f"📊 SWEEP COMPLETE | Total Engine Wall-Time: {total_elapsed/60:.2f} minutes")
    print("=================================================================\n")
    print(df_ofat_master[["parameter_name", "parameter_value", "runs_required"]])