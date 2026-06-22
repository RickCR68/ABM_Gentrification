import os
import sys
import numpy as np
import pandas as pd
from concurrent.futures import ProcessPoolExecutor
from SALib.sample import sobol
from SALib.analyze import sobol as analyze_sobol

# Ensure cluster can find the src path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from src.project.model import GentrificationModel

# 1. Define the GSA Problem Space
GSA_PROBLEM = {
    'num_vars': 5,
    'names': ['theta', 'risk_aversion', 'discount_factor', 'rationality', 'vision_radius'],
    'bounds': [
        [0.0, 3.0],   # theta (similarity preference)
        [0.0, 5.0],   # risk_aversion (\rho)
        [0.0, 1.0],   # discount_factor (\beta)
        [1.0, 20.0],  # rationality (\lambda)
        [1, 10]       # vision_radius (v) - Treated as continuous for sampling, cast to int later
    ]
}

SIM_STEPS = 50  # Number of steps to run each model instance

def run_single_simulation(args):
    """
    Worker function executed on an individual CPU core.
    Unpacks parameter combinations, runs the model, and extracts key metrics.
    """
    param_set, run_id = args
    theta, risk_aversion, discount_factor, rationality, vision_radius = param_set
    
    # Initialize your model with the sampled parameters
    model = GentrificationModel(
        width=20,
        height=20,
        density=0.8,
        income_similarity_min=theta,  # Handle flat/heterogeneous distribution setup here
        income_similarity_max=theta,
        risk_aversion_min=risk_aversion,
        risk_aversion_max=risk_aversion,
        discount_factor_min=discount_factor,
        discount_factor_max=discount_factor,
        rationality_min=rationality,
        rationality_max=rationality,
        maximum_vision_radius=int(np.round(vision_radius)),
        # ... include other fixed operational parameters ...
    )
    
    # Execute the simulation run
    for _ in range(SIM_STEPS):
        model.step()
        
    # Extract targeted macroscopic system outputs
    outputs = {
        'run_id': run_id,
        'segregation_index': model.datacollector.get_model_vars_dataframe()["Segregation"].iloc[-1], # placeholder name
        'displacement_rate': model.datacollector.get_model_vars_dataframe()["Displacement"].iloc[-1], # placeholder name
        'mean_rent_growth': model.mean_rent()
    }
    return outputs

if __name__ == "__main__":
    # 2. Generate Parameter Samples (N must be a power of 2)
    # Total samples = N * (2 * num_vars + 2)
    N = 128 
    param_values = sobol.sample(GSA_PROBLEM, N)
    print(f"Generated {len(param_values)} parameter combinations for Sobol GSA.")
    
    # Package parameter values with identifiers for mapping
    tasks = [(param_values[i], i) for i in range(len(param_values))]
    
    # 3. Distributed Execution via Process Pool
    # Snellius thin nodes have up to 128 cores available per node
    max_workers = int(os.getenv("SLURM_CPUS_PER_TASK", os.cpu_count()))
    print(f"Starting parallel processing using {max_workers} workers...")
    
    results = []
    with ProcessPoolExecutor(max_workers=max_workers) as executor:
        results = list(executor.map(run_single_simulation, tasks))
        
    # 4. Process and Convert Data
    df_results = pd.DataFrame(results)
    df_results.to_csv("gsa_raw_outputs.csv", index=False)
    
    # 5. Calculate Sensitivity Indices
    print("\n--- Sobol Sensitivity Analysis Results ---")
    for metric in ['segregation_index', 'displacement_rate', 'mean_rent_growth']:
        Y = df_results[metric].values
        Si = analyze_sobol.analyze(GSA_PROBLEM, Y, print_to_console=False)
        
        print(f"\nTarget Output Metric: {metric}")
        # Convert to DataFrame for easier inspection
        df_si = pd.DataFrame({
            'Variable': GSA_PROBLEM['names'],
            'First-Order (S1)': Si['S1'],
            'Total-Order (ST)': Si['ST']
        })
        print(df_si.to_string(index=False))