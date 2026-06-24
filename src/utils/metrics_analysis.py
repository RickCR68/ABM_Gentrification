"""
Analysis utilities for segregation and gentrification metrics.

This module provides functions to compute and analyze segregation metrics
from saved model outputs (CSV files). It reuses the pure metric functions
from src.project.metrics for consistency between real-time and post-hoc analysis.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pandas as pd
import numpy as np


def load_model_data(csv_path: str | Path) -> pd.DataFrame:
    """
    Load datacollector model reporter output from CSV.

    Args:
        csv_path: Path to the model reporter CSV file

    Returns:
        DataFrame with model metrics over time
    """
    return pd.read_csv(csv_path)


def load_agent_data(csv_path: str | Path) -> pd.DataFrame:
    """
    Load datacollector agent reporter output from CSV.

    Args:
        csv_path: Path to the agent reporter CSV file (typically
                 model_reporters.csv from mesa)

    Returns:
        DataFrame with agent-level data (Step, AgentID, and attributes)
    """
    return pd.read_csv(csv_path)


def compute_metrics_summary(model_df: pd.DataFrame) -> dict[str, Any]:
    """
    Compute summary statistics for segregation metrics from a model CSV.

    Assumes the CSV has columns for segregation metrics like:
    - homeless_fraction
    - theil_index
    - moran_i
    - spatial_entropy
    - neighborhood_heterogeneity
    - segregation_index
    - gentrification_indicator

    Args:
        model_df: DataFrame from load_model_data()

    Returns:
        Dictionary with summary statistics for each metric
    """
    metrics_to_summarize = [
        'homeless_fraction',
        'theil_index',
        'moran_i',
        'spatial_entropy',
        'neighborhood_heterogeneity',
        'segregation_index',
        'gentrification_indicator',
    ]

    summary = {}

    for metric in metrics_to_summarize:
        if metric in model_df.columns:
            values = model_df[metric].dropna()
            if len(values) > 0:
                summary[metric] = {
                    'mean': float(values.mean()),
                    'std': float(values.std()),
                    'min': float(values.min()),
                    'max': float(values.max()),
                    'initial': float(values.iloc[0]),
                    'final': float(values.iloc[-1]),
                    'change': float(values.iloc[-1] - values.iloc[0]),
                    'pct_change': float(
                        (values.iloc[-1] - values.iloc[0]) / values.iloc[0]
                        if values.iloc[0] != 0 else 0
                    ),
                }

    return summary


def analyze_segregation_trajectory(model_df: pd.DataFrame) -> dict[str, Any]:
    """
    Analyze whether segregation increased, decreased, or remained stable.

    Looks at multiple segregation indicators:
    - Moran's I (spatial clustering)
    - Segregation Index (dissimilarity)
    - Spatial Entropy (heterogeneity)

    Args:
        model_df: DataFrame from load_model_data()

    Returns:
        Dictionary with segregation trajectory analysis
    """
    analysis = {}

    segregation_indicators = ['moran_i', 'segregation_index']
    heterogeneity_indicators = ['spatial_entropy', 'neighborhood_heterogeneity']

    for indicator in segregation_indicators:
        if indicator in model_df.columns:
            values = model_df[indicator].dropna()
            if len(values) > 1:
                # Simple trend: compare final to initial
                initial = values.iloc[0]
                final = values.iloc[-1]
                change = final - initial

                if indicator == 'moran_i':
                    # High = clustered = segregated
                    trend = 'increased segregation' if change > 0.05 else (
                        'decreased segregation' if change < -0.05 else 'stable'
                    )
                else:  # segregation_index
                    # High = dissimilar = segregated
                    trend = 'increased segregation' if change > 0.05 else (
                        'decreased segregation' if change < -0.05 else 'stable'
                    )

                analysis[indicator] = {
                    'initial': float(initial),
                    'final': float(final),
                    'change': float(change),
                    'trend': trend,
                }

    for indicator in heterogeneity_indicators:
        if indicator in model_df.columns:
            values = model_df[indicator].dropna()
            if len(values) > 1:
                initial = values.iloc[0]
                final = values.iloc[-1]
                change = final - initial

                # High entropy/heterogeneity = less segregated (more integrated)
                trend = 'increased integration' if change > 0.05 else (
                    'decreased integration (segregation increasing)' if change < -0.05 else 'stable'
                )

                analysis[indicator] = {
                    'initial': float(initial),
                    'final': float(final),
                    'change': float(change),
                    'trend': trend,
                }

    return analysis


def analyze_gentrification(model_df: pd.DataFrame) -> dict[str, Any]:
    """
    Analyze gentrification indicators from model data.

    Examines:
    - Rising rents
    - Rising incomes
    - Homelessness rates
    - Income inequality (Gini, Theil)

    Args:
        model_df: DataFrame from load_model_data()

    Returns:
        Dictionary with gentrification analysis
    """
    analysis = {}

    # Rent trends
    if 'mean_rent' in model_df.columns:
        rent_values = model_df['mean_rent'].dropna()
        if len(rent_values) > 1:
            rent_change = rent_values.iloc[-1] - rent_values.iloc[0]
            rent_pct_change = (rent_change / rent_values.iloc[0]) if rent_values.iloc[0] != 0 else 0
            analysis['rent'] = {
                'initial': float(rent_values.iloc[0]),
                'final': float(rent_values.iloc[-1]),
                'change': float(rent_change),
                'pct_change': float(rent_pct_change),
                'trend': 'rising' if rent_change > 0 else 'falling' if rent_change < 0 else 'stable',
            }

    # Income trends
    if 'city_mean_income' in model_df.columns:
        income_values = model_df['city_mean_income'].dropna()
        if len(income_values) > 1:
            income_change = income_values.iloc[-1] - income_values.iloc[0]
            income_pct_change = (income_change / income_values.iloc[0]) if income_values.iloc[0] != 0 else 0
            analysis['income'] = {
                'initial': float(income_values.iloc[0]),
                'final': float(income_values.iloc[-1]),
                'change': float(income_change),
                'pct_change': float(income_pct_change),
                'trend': 'rising' if income_change > 0 else 'falling' if income_change < 0 else 'stable',
            }

    # Homelessness trends
    if 'homeless_fraction' in model_df.columns:
        homeless_values = model_df['homeless_fraction'].dropna()
        if len(homeless_values) > 1:
            homeless_change = homeless_values.iloc[-1] - homeless_values.iloc[0]
            analysis['homelessness'] = {
                'initial': float(homeless_values.iloc[0]),
                'final': float(homeless_values.iloc[-1]),
                'peak': float(homeless_values.max()),
                'change': float(homeless_change),
                'trend': 'worsening' if homeless_change > 0.05 else (
                    'improving' if homeless_change < -0.05 else 'stable'
                ),
            }

    # Inequality trends
    if 'gini_coefficient' in model_df.columns:
        gini_values = model_df['gini_coefficient'].dropna()
        if len(gini_values) > 1:
            gini_change = gini_values.iloc[-1] - gini_values.iloc[0]
            analysis['gini'] = {
                'initial': float(gini_values.iloc[0]),
                'final': float(gini_values.iloc[-1]),
                'change': float(gini_change),
                'trend': 'increasing inequality' if gini_change > 0.01 else (
                    'decreasing inequality' if gini_change < -0.01 else 'stable'
                ),
            }

    if 'theil_index' in model_df.columns:
        theil_values = model_df['theil_index'].dropna()
        if len(theil_values) > 1:
            theil_change = theil_values.iloc[-1] - theil_values.iloc[0]
            analysis['theil'] = {
                'initial': float(theil_values.iloc[0]),
                'final': float(theil_values.iloc[-1]),
                'change': float(theil_change),
                'trend': 'increasing inequality' if theil_change > 0.01 else (
                    'decreasing inequality' if theil_change < -0.01 else 'stable'
                ),
            }

    # Gentrification indicator
    if 'gentrification_indicator' in model_df.columns:
        gentrif_values = model_df['gentrification_indicator'].dropna()
        if len(gentrif_values) > 0:
            analysis['gentrification_indicator'] = {
                'mean': float(gentrif_values.mean()),
                'max': float(gentrif_values.max()),
                'final': float(gentrif_values.iloc[-1]),
            }

    return analysis


def compare_runs(
    model_dfs: list[pd.DataFrame],
    run_names: list[str] | None = None,
) -> pd.DataFrame:
    """
    Compare segregation metrics across multiple model runs.

    Args:
        model_dfs: List of DataFrames from load_model_data()
        run_names: Optional list of run names/descriptions

    Returns:
        DataFrame with comparison of final metric values across runs
    """
    if run_names is None:
        run_names = [f"Run {i}" for i in range(len(model_dfs))]

    comparison_data = []

    metrics_to_compare = [
        'homeless_fraction',
        'theil_index',
        'moran_i',
        'spatial_entropy',
        'neighborhood_heterogeneity',
        'segregation_index',
        'gentrification_indicator',
        'gini_coefficient',
    ]

    for df, run_name in zip(model_dfs, run_names):
        row = {'run': run_name}
        for metric in metrics_to_compare:
            if metric in df.columns:
                values = df[metric].dropna()
                if len(values) > 0:
                    row[f'{metric}_initial'] = float(values.iloc[0])
                    row[f'{metric}_final'] = float(values.iloc[-1])
                    row[f'{metric}_mean'] = float(values.mean())
                    row[f'{metric}_max'] = float(values.max())
        comparison_data.append(row)

    return pd.DataFrame(comparison_data)


def export_metrics_report(
    model_df: pd.DataFrame,
    output_path: str | Path,
) -> None:
    """
    Export a comprehensive metrics report to a text file.

    Args:
        model_df: DataFrame from load_model_data()
        output_path: Path to save the report
    """
    with open(output_path, 'w') as f:
        f.write("=" * 80 + "\n")
        f.write("SEGREGATION AND GENTRIFICATION METRICS REPORT\n")
        f.write("=" * 80 + "\n\n")

        # Summary statistics
        f.write("METRIC SUMMARY STATISTICS\n")
        f.write("-" * 80 + "\n")
        summary = compute_metrics_summary(model_df)
        for metric, stats in summary.items():
            f.write(f"\n{metric.upper()}\n")
            f.write(f"  Initial:    {stats['initial']:.6f}\n")
            f.write(f"  Final:      {stats['final']:.6f}\n")
            f.write(f"  Change:     {stats['change']:.6f}\n")
            f.write(f"  Pct change: {stats['pct_change']:.2%}\n")
            f.write(f"  Mean:       {stats['mean']:.6f}\n")
            f.write(f"  Std dev:    {stats['std']:.6f}\n")
            f.write(f"  Range:      [{stats['min']:.6f}, {stats['max']:.6f}]\n")

        # Segregation analysis
        f.write("\n" + "=" * 80 + "\n")
        f.write("SEGREGATION TRAJECTORY ANALYSIS\n")
        f.write("=" * 80 + "\n")
        seg_analysis = analyze_segregation_trajectory(model_df)
        for indicator, analysis in seg_analysis.items():
            f.write(f"\n{indicator.upper()}\n")
            f.write(f"  Trend:  {analysis['trend']}\n")
            f.write(f"  Change: {analysis['change']:.6f}\n")

        # Gentrification analysis
        f.write("\n" + "=" * 80 + "\n")
        f.write("GENTRIFICATION ANALYSIS\n")
        f.write("=" * 80 + "\n")
        gentrif_analysis = analyze_gentrification(model_df)
        for indicator, analysis in gentrif_analysis.items():
            f.write(f"\n{indicator.upper()}\n")
            for key, value in analysis.items():
                if isinstance(value, str):
                    f.write(f"  {key}: {value}\n")
                else:
                    f.write(f"  {key}: {value:.6f}\n")


def plot_segregation_metrics(
    model_df: pd.DataFrame,
    output_dir: str | Path | None = None,
) -> None:
    """
    Create plots of segregation metrics over time.

    Requires matplotlib. Creates plots for:
    - Moran's I and Segregation Index
    - Spatial Entropy and Neighborhood Heterogeneity
    - Homelessness rate
    - Income inequality

    Args:
        model_df: DataFrame from load_model_data()
        output_dir: Optional directory to save plots. If None, plots are displayed.
    """
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        print("matplotlib not available. Install with: pip install matplotlib")
        return

    output_dir = Path(output_dir) if output_dir else None
    if output_dir:
        output_dir.mkdir(parents=True, exist_ok=True)

    # Plot 1: Spatial clustering and segregation
    if 'moran_i' in model_df.columns or 'segregation_index' in model_df.columns:
        fig, ax = plt.subplots(figsize=(10, 6))
        if 'moran_i' in model_df.columns:
            ax.plot(model_df.index, model_df['moran_i'], label="Moran's I", linewidth=2)
        if 'segregation_index' in model_df.columns:
            ax.plot(model_df.index, model_df['segregation_index'], label="Segregation Index", linewidth=2)
        ax.set_xlabel("Step")
        ax.set_ylabel("Index Value")
        ax.set_title("Spatial Segregation Indicators")
        ax.legend()
        ax.grid(True, alpha=0.3)
        if output_dir:
            plt.savefig(output_dir / "segregation.png", dpi=150, bbox_inches='tight')
        else:
            plt.show()
        plt.close()

    # Plot 2: Spatial diversity
    if 'spatial_entropy' in model_df.columns or 'neighborhood_heterogeneity' in model_df.columns:
        fig, ax = plt.subplots(figsize=(10, 6))
        if 'spatial_entropy' in model_df.columns:
            ax.plot(model_df.index, model_df['spatial_entropy'], label="Spatial Entropy", linewidth=2)
        if 'neighborhood_heterogeneity' in model_df.columns:
            ax.plot(model_df.index, model_df['neighborhood_heterogeneity'],
                   label="Neighborhood Heterogeneity", linewidth=2)
        ax.set_xlabel("Step")
        ax.set_ylabel("Heterogeneity/Entropy")
        ax.set_title("Spatial Diversity Indicators")
        ax.legend()
        ax.grid(True, alpha=0.3)
        if output_dir:
            plt.savefig(output_dir / "diversity.png", dpi=150, bbox_inches='tight')
        else:
            plt.show()
        plt.close()

    # Plot 3: Homelessness and rent pressure
    if 'homeless_fraction' in model_df.columns:
        fig, ax = plt.subplots(figsize=(10, 6))
        ax.plot(model_df.index, model_df['homeless_fraction'], label="Homeless Fraction",
               linewidth=2, color='red')
        ax.set_xlabel("Step")
        ax.set_ylabel("Fraction of Homeless Agents")
        ax.set_title("Affordability Crisis: Homelessness Over Time")
        ax.legend()
        ax.grid(True, alpha=0.3)
        if output_dir:
            plt.savefig(output_dir / "homelessness.png", dpi=150, bbox_inches='tight')
        else:
            plt.show()
        plt.close()

    # Plot 4: Inequality
    if 'gini_coefficient' in model_df.columns or 'theil_index' in model_df.columns:
        fig, ax = plt.subplots(figsize=(10, 6))
        if 'gini_coefficient' in model_df.columns:
            ax.plot(model_df.index, model_df['gini_coefficient'], label="Gini Coefficient", linewidth=2)
        if 'theil_index' in model_df.columns:
            # Normalize Theil for visualization
            theil_normalized = model_df['theil_index'] / np.log(len(model_df))
            ax.plot(model_df.index, theil_normalized, label="Theil Index (normalized)", linewidth=2)
        ax.set_xlabel("Step")
        ax.set_ylabel("Inequality Index")
        ax.set_title("Income Inequality Over Time")
        ax.legend()
        ax.grid(True, alpha=0.3)
        if output_dir:
            plt.savefig(output_dir / "inequality.png", dpi=150, bbox_inches='tight')
        else:
            plt.show()
        plt.close()

