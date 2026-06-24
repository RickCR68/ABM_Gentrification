import pandas as pd
import matplotlib.pyplot as plt
from src.project.model import GentrificationModel

def run_simulation(experiment_name: str, steps: int = 2000):
    print(f"🚀 Starting Experiment: {experiment_name}")
    
    # 1. Initialize the model with your custom flag
    model = GentrificationModel(
        width=20, 
        height=20, 
        density=0.8, 
        experiment_type=experiment_name
    )
    
    history = []
    
    # 2. Step through the simulation execution loop
    for step in range(steps):
        model.step()
        
        # 3. Pull metrics out of your datacollector or direct wrappers every step
        history.append({
            "Step": step,
            "Moran_I": model.get_moran_i(),
            "Segregation_Index": model.get_segregation_index(),
            "Homeless_Fraction": model.get_homeless_fraction(),
            "Mean_Rent": model.mean_rent(),
            "Success_Rate": model.movement_success_rate() if hasattr(model, 'movement_success_rate') else 0
        })
        
    print(f"✅ Finished {experiment_name}\n")
    return pd.DataFrame(history)

if __name__ == "__main__":
    # Run both experiments for 200 steps
    df_shock = run_simulation("poor_shock", steps=2000)
    df_bipolar = run_simulation("half_and_half", steps=2000)
    
    # 4. Generate comparison plots
    plt.figure(figsize=(12, 5))
    
    # Subplot 1: Spatial Clustering
    plt.subplot(1, 2, 1)
    plt.plot(df_shock["Step"], df_shock["Moran_I"], label="Poor Shock Model")
    plt.plot(df_bipolar["Step"], df_bipolar["Moran_I"], label="Half-and-Half Archetypes")
    plt.title("Spatial Segregation over Time (Moran's I)")
    plt.xlabel("Simulation Steps")
    plt.ylabel("Moran's I")
    plt.legend()
    
    # Subplot 2: Economic Displacement
    plt.subplot(1, 2, 2)
    plt.plot(df_shock["Step"], df_shock["Homeless_Fraction"], label="Poor Shock Model")
    plt.plot(df_bipolar["Step"], df_bipolar["Homeless_Fraction"], label="Half-and-Half Archetypes")
    plt.title("Displacement Over Time (Homeless Fraction)")
    plt.xlabel("Simulation Steps")
    plt.ylabel("Fraction of Population")
    plt.legend()
    
    plt.tight_layout()
    plt.savefig("experiment_results_2000.png")
    print("📊 Plot saved as 'experiment_results_2000.png'")