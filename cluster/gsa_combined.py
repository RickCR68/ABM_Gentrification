import numpy as np
import pandas as pd
from SALib.analyze import sobol as analyze_sobol

# 1. Define the identical GSA Problem Space for SALib Analysis
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
        [0.0, 2.0],
        [0.0, 2.0],
        [0.0, 20.0],
        [0.0, 1.0],
        [0.001, 0.1]
    ]
}

if __name__ == "__main__":
    print("Combining part outputs and running Sobol analysis...")
    
    # 2. Read, combine, and re-sort by global run_id to restore matrix order
    try:
        parts = [pd.read_csv(f"gsa_raw_outputs_part_{i}.csv") for i in [1, 2, 3, 4]]
        df_results = pd.concat(parts)
        df_results = df_results.sort_values("run_id").reset_index(drop=True)
        
        # Save a master copy of complete data records for safe keeping
        df_results.to_csv("gsa_raw_outputs_combined.csv", index=False)
        print(f"Combined total of {len(df_results)} simulation runs successfully.")
        
    except FileNotFoundError as e:
        print(f"Error: Missing split chunk data file! Check your work directories. Details: {e}")
        exit(1)

    # 3. Calculate Sensitivity Indices
    print("\n--- Sobol Sensitivity Analysis Results ---")
    metrics_to_analyze = ['pct_satisfied', 'movement_success_rate', 'mean_rent']

    for metric in metrics_to_analyze:
        Y = df_results[metric].values
        
        # Check for edge cases where model output does not change across runs (e.g., zero variance)
        if np.all(Y == Y[0]):
            print(f"\nTarget Output Metric: {metric}")
            print("Warning: Output is constant across all runs. Variance is zero; skipping index computation.")
            continue
            
        # Execute global sensitivity analysis matrix math
        Si = analyze_sobol.analyze(GSA_PROBLEM, Y, print_to_console=False)
        
        print(f"\nTarget Output Metric: {metric}")
        # Convert to DataFrame for cleaner formatting
        df_si = pd.DataFrame({
            'Variable': GSA_PROBLEM['names'],
            'First-Order (S1)': Si['S1'],
            'Total-Order (ST)': Si['ST']
        })
        print(df_si.to_string(index=False))