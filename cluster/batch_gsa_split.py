import os
import sys
import numpy as np
import pandas as pd
from concurrent.futures import ProcessPoolExecutor
from SALib.sample import sobol

# Ensure cluster can find the src path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from src.project.model import GentrificationModel

# Define absolute path for the output directory to avoid cluster sync issues
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
OUTPUT_DIR = os.path.join(BASE_DIR, "gsa_detailed_runs")

# 1. Define the GSA Problem Space
GSA_PROBLEM = {
    'num_vars': 5,
    'names': [
        'moving_cost', 
        'rejection_cost', 
        'vision_income_scale', 
        'satisficing_threshold', 
        'rent_adjustment_rate'
    ],
    'bounds': [
        [0.0, 2.0],   # moving_cost
        [0.0, 2.0],   # rejection_cost
        [0.0, 20.0],  # vision_income_scale
        [0.0, 1.0],   # satisficing_threshold
        [0.001, 0.1]  # rent_adjustment_rate
    ]
}

SIM_STEPS = 50  # Updated to requested steps

def run_single_simulation(args):
    """
    Worker function executed on an individual CPU core.
    """
    param_set, run_id = args
    (moving_cost, 
     rejection_cost, 
     vision_income_scale, 
     satisficing_threshold, 
     rent_adjustment_rate) = param_set
    
    model = GentrificationModel(
        width=11,
        height=11,
        density=0.9,
        neighborhood_radius=2,
        affordability_share=0.8,
        moving_cost=moving_cost,
        rejection_cost=rejection_cost,
        vision_income_scale=vision_income_scale,
        satisficing_threshold=satisficing_threshold,
        rent_adjustment_rate=rent_adjustment_rate,
        rng=42 + run_id
    )
    
    for _ in range(SIM_STEPS):
        model.step()
        
    df_model_vars = model.datacollector.get_model_vars_dataframe()
    df_agent_vars = model.datacollector.get_agent_vars_dataframe()
    
    # Graceful error handling for GPFS cluster file-system latency
    try:
        os.makedirs(OUTPUT_DIR, exist_ok=True)
    except FileExistsError:
        pass

    df_model_vars.to_csv(os.path.join(OUTPUT_DIR, f"model_run_{run_id}.csv"), index=True)
    df_agent_vars.to_csv(os.path.join(OUTPUT_DIR, f"agent_run_{run_id}.csv"), index=True)
    
    outputs = {
        'run_id': run_id,
        'moving_cost': moving_cost,
        'rejection_cost': rejection_cost,
        'vision_income_scale': vision_income_scale,
        'satisficing_threshold': satisficing_threshold,
        'rent_adjustment_rate': rent_adjustment_rate,
        'pct_satisfied': df_model_vars["pct_satisfied"].iloc[-1],
        'movement_success_rate': df_model_vars["movement_success_rate"].iloc[-1],
        'mean_rent': df_model_vars["mean_rent"].iloc[-1]
    }
    return outputs

if __name__ == "__main__":
    # 2. Configure target execution split configuration
    # Read from environment variable or default to part 1 (Acceptable values: 1, 2, 3, 4)
    CURRENT_PART = int(os.getenv("GSA_PART", 1))
    
    N = 512 
    param_values = sobol.sample(GSA_PROBLEM, N)
    total_tasks = len(param_values)
    print(f"Total global matrix contains {total_tasks} parameter combinations.")
    
    # Pre-create directory on the master thread
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    
    # Create all tasks matching their universal global matrix run_id indices
    all_tasks = [(param_values[i], i) for i in range(total_tasks)]
    
    # Slice the task list into 4 exact quadrants
    chunk_size = total_tasks // 4
    start_idx = (CURRENT_PART - 1) * chunk_size
    end_idx = start_idx + chunk_size if CURRENT_PART < 4 else total_tasks
    
    active_tasks = all_tasks[start_idx:end_idx]
    print(f"Executing Part {CURRENT_PART}/{4}. Processing tasks index {start_idx} to {end_idx} ({len(active_tasks)} simulations)...")
    
    # 3. Distributed Execution via Process Pool
    max_workers = int(os.getenv("SLURM_CPUS_PER_TASK", os.cpu_count()))
    print(f"Starting parallel processing using {max_workers} workers...")
    
    results = []
    with ProcessPoolExecutor(max_workers=max_workers) as executor:
        results = list(executor.map(run_single_simulation, active_tasks))
        
    # 4. Save Chunk Output
    df_results = pd.DataFrame(results)
    output_filename = f"gsa_raw_outputs_part_{CURRENT_PART}.csv"
    df_results.to_csv(output_filename, index=False)
    print(f"Successfully completed chunk! Part results saved to {output_filename}")