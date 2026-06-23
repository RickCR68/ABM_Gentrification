#!/usr/bin/env python
"""
Integration test for segregation and gentrification metrics.

Tests that:
1. Model initializes successfully
2. Metrics compute without errors
3. Data is collected and stored
4. Analysis functions work correctly
"""

import sys
import tempfile
from pathlib import Path

def test_model_initialization():
    """Test that model initializes with metrics."""
    print("Testing model initialization...", end=" ")
    try:
        from src.project.model import GentrificationModel

        model = GentrificationModel(
            width=10,
            height=10,
            density=0.5,
            neighborhood_radius=1,
            rng=42,
        )
        assert len(model.agents) > 0, "No agents created"
        print("✓")
        return model
    except Exception as e:
        print(f"✗ Error: {e}")
        raise

def test_metrics_computation(model):
    """Test that all metrics compute without errors."""
    print("Testing metrics computation...", end=" ")
    try:
        metrics = {
            "homeless_fraction": model.get_homeless_fraction(),
            "theil_index": model.get_theil_index(),
            "moran_i": model.get_moran_i(),
            "spatial_entropy": model.get_spatial_entropy(),
            "neighborhood_heterogeneity": model.get_neighborhood_heterogeneity(),
            "income_mobility_indicator": model.get_income_mobility_indicator(),
            "segregation_index": model.get_segregation_index(),
            "gentrification_indicator": model.get_gentrification_indicator(),
        }

        # Check all metrics are valid numbers
        for name, value in metrics.items():
            assert isinstance(value, (int, float)), f"{name} not a number"
            assert value >= 0 or name == "moran_i", f"{name} negative (only moran_i can be negative)"

        print("✓")
        return metrics
    except Exception as e:
        print(f"✗ Error: {e}")
        raise

def test_datacollection(model):
    """Test that metrics are collected in datacollector."""
    print("Testing data collection...", end=" ")
    try:
        # Run a few steps
        for _ in range(3):
            model.step()

        # Get collected data
        df = model.datacollector.get_model_vars_dataframe()

        # Check metrics are in the dataframe
        expected_columns = [
            "homeless_fraction",
            "theil_index",
            "moran_i",
            "spatial_entropy",
            "neighborhood_heterogeneity",
            "income_mobility_indicator",
            "segregation_index",
            "gentrification_indicator",
        ]

        for col in expected_columns:
            assert col in df.columns, f"Column {col} not in datacollector"
            assert len(df[col].dropna()) > 0, f"No data for {col}"

        print("✓")
        return df
    except Exception as e:
        print(f"✗ Error: {e}")
        raise

def test_csv_analysis(df):
    """Test that CSV analysis functions work."""
    print("Testing CSV analysis...", end=" ")
    try:
        from src.utils.metrics_analysis import (
            compute_metrics_summary,
            analyze_segregation_trajectory,
            analyze_gentrification,
        )

        # Test summary
        summary = compute_metrics_summary(df)
        assert len(summary) > 0, "No metrics in summary"

        # Test segregation trajectory
        seg_trends = analyze_segregation_trajectory(df)
        assert len(seg_trends) > 0, "No segregation trends"

        # Test gentrification analysis
        gentrif = analyze_gentrification(df)
        assert len(gentrif) > 0, "No gentrification analysis"

        print("✓")
    except Exception as e:
        print(f"✗ Error: {e}")
        raise

def test_export_functions(df):
    """Test that export functions work."""
    print("Testing export functions...", end=" ")
    try:
        from src.utils.metrics_analysis import (
            export_metrics_report,
            plot_segregation_metrics,
        )

        with tempfile.TemporaryDirectory() as tmpdir:
            # Test report export
            report_path = Path(tmpdir) / "report.txt"
            export_metrics_report(df, report_path)
            assert report_path.exists(), "Report file not created"
            assert report_path.stat().st_size > 0, "Report file is empty"

            # Test plotting
            plots_dir = Path(tmpdir) / "plots"
            plot_segregation_metrics(df, plots_dir)
            # Note: plotting might not create files if matplotlib display fails,
            # but function should complete without error

        print("✓")
    except Exception as e:
        print(f"✗ Error: {e}")
        raise

def main():
    """Run all integration tests."""
    print("=" * 60)
    print("METRICS INTEGRATION TEST SUITE")
    print("=" * 60)

    try:
        print("\n1. Model & Metrics Initialization")
        model = test_model_initialization()

        print("\n2. Metrics Computation")
        metrics = test_metrics_computation(model)

        print("\n3. Data Collection")
        df = test_datacollection(model)

        print("\n4. CSV Analysis")
        test_csv_analysis(df)

        print("\n5. Export Functions")
        test_export_functions(df)

        print("\n" + "=" * 60)
        print("✓ ALL TESTS PASSED!")
        print("=" * 60)
        print("\nIntegration Summary:")
        print("  • Model initializes successfully")
        print("  • All 8 metrics compute without errors")
        print("  • Data collection works correctly")
        print("  • CSV analysis functions operational")
        print("  • Export and visualization functions work")
        print("\nThe metrics system is ready for use! 🚀")
        return 0

    except Exception as e:
        print("\n" + "=" * 60)
        print("✗ TEST FAILED")
        print("=" * 60)
        import traceback
        traceback.print_exc()
        return 1

if __name__ == "__main__":
    sys.exit(main())

