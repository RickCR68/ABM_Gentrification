#!/usr/bin/env python
"""Quick test to verify metrics implementation."""

import sys
sys.path.insert(0, '.')

try:
    print("Testing imports...")

    from src.project import metrics
    print("✓ Metrics module imported")

    from src.utils import metrics_analysis
    print("✓ Metrics analysis module imported")

    print("\nAvailable metric functions:")
    metric_functions = [
        'homeless_fraction',
        'theil_index',
        'moran_i',
        'spatial_entropy',
        'neighborhood_heterogeneity',
        'income_mobility_indicator',
        'segregation_index',
        'gentrification_indicator',
    ]

    for func_name in metric_functions:
        if hasattr(metrics, func_name):
            print(f"  ✓ {func_name}")
        else:
            print(f"  ✗ {func_name} NOT FOUND")

    print("\nAvailable analysis functions:")
    analysis_functions = [
        'load_model_data',
        'load_agent_data',
        'compute_metrics_summary',
        'analyze_segregation_trajectory',
        'analyze_gentrification',
        'compare_runs',
        'export_metrics_report',
        'plot_segregation_metrics',
    ]

    for func_name in analysis_functions:
        if hasattr(metrics_analysis, func_name):
            print(f"  ✓ {func_name}")
        else:
            print(f"  ✗ {func_name} NOT FOUND")

    print("\n✓ All imports successful!")

except Exception as e:
    print(f"✗ Error: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

