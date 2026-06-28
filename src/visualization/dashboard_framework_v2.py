from __future__ import annotations

"""Config-driven Solara/Mesa dashboard framework.

Put this file and ``resize_observer.vue`` in ``src/visualization`` and run:

    solara run src/visualization/dashboard_framework.py

Edit only these main configuration objects in normal use:

* PLOT_SPECS: what every plot displays.
* PLOT_VIEWS: which plots appear in each dropdown view and their order.
* EXPERIMENTS: which parameters are adjustable on each URL/page.
"""

from dataclasses import dataclass, field
from typing import Any, Callable, Iterable, Mapping, Sequence

import math
import matplotlib.pyplot as plt
from matplotlib.figure import Figure
from matplotlib.lines import Line2D
import numpy as np
import solara

from mesa.visualization import Slider, SolaraViz, SpaceRenderer
from mesa.visualization.components import AgentPortrayalStyle, PropertyLayerStyle
from mesa.visualization.solara_viz import SpaceRendererComponent
from mesa.visualization.utils import update_counter

from src.utils.helpers import generate_vibrant_red_blue_gradient
from src.project.experimental_models import SchellingGridExperiment
from src.project.model import GentrificationModel

COLORS = [
    "#920E1D",
    "#A7380C",
    "#C55B2E",
    "#FFD68A",
    "#6BBE6C",
    "#107E22",
    "#145506",
]

GAME_OUTCOMES = (
    "move_accept",
    "move_reject",
    "stay_accept",
    "stay_reject",
)

GAME_OUTCOME_LABELS = {
    "move_accept": "Move–accept",
    "move_reject": "Move–reject",
    "stay_accept": "Stay–accept",
    "stay_reject": "Stay–reject",
}

INCOME_QUANTILE_LABELS = (
    "Bottom 15%",
    "15–35%",
    "35–55%",
    "55–70%",
    "70–85%",
    "85–99%",
    "Top 1%",
)


# -----------------------------------------------------------------------------
# Resize observer: resize_observer.vue must be next to this Python file.
# -----------------------------------------------------------------------------

@solara.component_vue("resize_observer.vue")
def ViewListener(
    view_data=None,
    on_view_data=None,
    children=[],
    style=None,
):
    pass


# -----------------------------------------------------------------------------
# Configuration types
# -----------------------------------------------------------------------------

@dataclass(frozen=True)
class SeriesSpec:
    metric: str
    label: str
    color: str | None = None
    linestyle: str = "-"


@dataclass(frozen=True)
class PlotSpec:
    key: str
    title: str
    series: tuple[SeriesSpec, ...]
    ylabel: str = ""
    xlabel: str = "Model step"
    legend: bool = True
    ylim: tuple[float, float] | None = None
    kind: str = "line"  # line, histogram, scatter
    height_px: int = 330


@dataclass(frozen=True)
class StatSpec:
    label: str
    getter: str | Callable[[Any], Any]
    fmt: str = ".3f"
    suffix: str = ""


@dataclass(frozen=True)
class StatSection:
    title: str
    items: tuple[StatSpec, ...]


@dataclass(frozen=True)
class ExperimentSpec:
    key: str
    label: str
    description: str
    adjustable_groups: tuple[str, ...]
    overrides: Mapping[str, Any] = field(default_factory=dict)
    default_view: str = "General"
    stats_view: str = "baseline"
    # spatial_title: str = "Spatial state"


# -----------------------------------------------------------------------------
# Complete baseline parameter values.
# Values that are not exposed on a page remain fixed.
# -----------------------------------------------------------------------------

BASE_VALUES: dict[str, Any] = {
    "rng": 42,
    "width": 20,
    "height": 20,
    "density": 0.8,
    "neighborhood_radius": 1,
    "initial_income_min": 0.2,
    "initial_income_max": 1.0,
    "affordability_share": 0.8,
    "income_similarity_min": 0.0,
    "income_similarity_max": 2.0,
    "discount_factor_min": 0.0,
    "discount_factor_max": 1.0,
    "risk_aversion_min": 0.0,
    "risk_aversion_max": 0.99,
    "rationality_min": 1.0,
    "rationality_max": 10.0,
    "satisficing_threshold": 0.1,
    "minimum_absolute_improvement": 1e-6,
    "vision_income_scale": 5.0,
    "maximum_vision_radius": 10,
    "steps_until_satisfied": 5,
    "rent_adjustment_rate": 0.01,
    "income_growth_scaling": 0.001,
    "income_volatility": 0.05,
    "moving_cost": 0.05,
    "rejection_cost": 0.05,
    "neighborhood_risk_aversion": 0.5,
    "neighborhood_rationality": 5.0,
    "qre_tolerance": 1e-10,
    "qre_maximum_iterations": 1000,
    "qre_damping": 0.5,
    "keep_game_history": True,
}


# -----------------------------------------------------------------------------
# Parameter controls. Use sliders only where continuous exploration is useful.
# Use InputText for exact/technical values and fixed raw values for advanced
# numerical settings that should not clutter the page.
# -----------------------------------------------------------------------------

CONTROL_FACTORIES: dict[str, Callable[[Any], Any]] = {
    "rng": lambda v: {
        "type": "InputText", "value": v, "label": "Random seed"
    },
    "density": lambda v: Slider(
        "Agent density", value=v, min=0.1, max=0.95, step=0.05
    ),
    "neighborhood_radius": lambda v: Slider(
        "Neighbourhood radius", value=v, min=1, max=4, step=1
    ),
    "initial_income_min": lambda v: {
        "type": "InputText", "value": v, "label": "Minimum initial income"
    },
    "initial_income_max": lambda v: {
        "type": "InputText", "value": v, "label": "Maximum initial income"
    },
    "affordability_share": lambda v: Slider(
        "Maximum affordable rent share", value=v, min=0.1, max=1.5, step=0.05
    ),
    "income_similarity_min": lambda v: {
        "type": "InputText", "value": v, "label": "Minimum similarity preference"
    },
    "income_similarity_max": lambda v: {
        "type": "InputText", "value": v, "label": "Maximum similarity preference"
    },
    "discount_factor_min": lambda v: {
        "type": "InputText", "value": v, "label": "Minimum discount factor"
    },
    "discount_factor_max": lambda v: {
        "type": "InputText", "value": v, "label": "Maximum discount factor"
    },
    "risk_aversion_min": lambda v: {
        "type": "InputText", "value": v, "label": "Minimum risk aversion"
    },
    "risk_aversion_max": lambda v: {
        "type": "InputText", "value": v, "label": "Maximum risk aversion"
    },
    "rationality_min": lambda v: {
        "type": "InputText", "value": v, "label": "Minimum household rationality"
    },
    "rationality_max": lambda v: {
        "type": "InputText", "value": v, "label": "Maximum household rationality"
    },
    "satisficing_threshold": lambda v: Slider(
        "Required proportional utility improvement",
        value=v, min=0.0, max=1.0, step=0.01,
    ),
    "vision_income_scale": lambda v: Slider(
        "Initial-income vision scaling", value=v, min=0.0, max=20.0, step=0.5
    ),
    "maximum_vision_radius": lambda v: Slider(
        "Maximum vision radius", value=v, min=1, max=20, step=1
    ),
    "steps_until_satisfied": lambda v: Slider(
        "Stationary steps until stable", value=v, min=1, max=30, step=1
    ),
    "rent_adjustment_rate": lambda v: Slider(
        "Rent adjustment rate", value=v, min=0.0, max=0.1, step=0.001
    ),
    "income_growth_scaling": lambda v: Slider(
        "Neighbourhood income-growth strength",
        value=v, min=0.0, max=0.1, step=0.001,
    ),
    "income_volatility": lambda v: Slider(
        "Income volatility", value=v, min=0.0, max=0.3, step=0.01
    ),
    "moving_cost": lambda v: {
        "type": "InputText", "value": v, "label": "Moving cost"
    },
    "rejection_cost": lambda v: {
        "type": "InputText", "value": v, "label": "Rejection cost"
    },
    "neighborhood_risk_aversion": lambda v: Slider(
        "Neighbourhood risk aversion", value=v, min=0.0, max=5.0, step=0.05
    ),
    "neighborhood_rationality": lambda v: Slider(
        "Neighbourhood rationality", value=v, min=0.0, max=20.0, step=0.5
    ),
}

PARAMETER_GROUPS: dict[str, tuple[str, ...]] = {
    "Basic": (
        "rng", "density", "neighborhood_radius",
        "initial_income_min", "initial_income_max",
    ),
    "Housing": (
        "affordability_share", "rent_adjustment_rate",
        "moving_cost", "rejection_cost",
    ),
    "Household behaviour": (
        "income_similarity_min", "income_similarity_max",
        "discount_factor_min", "discount_factor_max",
        "risk_aversion_min", "risk_aversion_max",
        "rationality_min", "rationality_max",
        "satisficing_threshold",
    ),
    "Mobility": (
        "vision_income_scale", "maximum_vision_radius",
        "steps_until_satisfied",
    ),
    "Income dynamics": (
        "income_growth_scaling", "income_volatility",
    ),
    "Neighbourhood game": (
        "neighborhood_risk_aversion", "neighborhood_rationality",
    ),
}


# -----------------------------------------------------------------------------
# Plot definitions. Add a plot once here, then place its key in any PLOT_VIEW.
# -----------------------------------------------------------------------------

PLOT_SPECS: dict[str, PlotSpec] = {
    "satisfaction": PlotSpec(
        key="satisfaction",
        title="Residential stability",
        ylabel="Stable households (%)",
        series=(SeriesSpec("pct_satisfied", "Stable households", "tab:green"),),
        ylim=(0, 100),
    ),
    "movement_counts": PlotSpec(
        key="movement_counts",
        title="Relocation outcomes",
        ylabel="Count",
        series=(
            SeriesSpec("successful_moves", "Successful moves", "tab:blue"),
            SeriesSpec("failed_searches", "Failed searches", "tab:red"),
        ),
    ),
    "movement_rate": PlotSpec(
        key="movement_rate",
        title="Realized movement rate",
        ylabel="Rate",
        series=(SeriesSpec("movement_success_rate", "Movement rate", "tab:purple"),),
        ylim=(0, 1),
    ),
    "income_rent": PlotSpec(
        key="income_rent",
        title="Income and rent dynamics",
        ylabel="Mean value",
        series=(
            SeriesSpec("city_mean_income", "Household income", "tab:blue"),
            SeriesSpec("mean_neighbor_income", "Neighbourhood income", "tab:orange"),
            SeriesSpec("mean_rent", "Rent", "tab:red"),
        ),
    ),
    "utility": PlotSpec(
        key="utility",
        title="Household utility and value",
        ylabel="Mean value",
        series=(
            SeriesSpec("mean_utility", "Raw utility", "tab:blue"),
            SeriesSpec("mean_value", "Signed CRRA value", "tab:green"),
        ),
    ),
    "inequality": PlotSpec(
        key="inequality",
        title="Income inequality",
        ylabel="Index",
        series=(
            SeriesSpec("gini_coefficient", "Gini coefficient", "tab:blue"),
            SeriesSpec("theil_index", "Theil index", "tab:purple"),
        ),
    ),
    "segregation": PlotSpec(
        key="segregation",
        title="Spatial segregation",
        ylabel="Index",
        series=(
            SeriesSpec("moran_i", "Moran's I", "tab:blue"),
            SeriesSpec("segregation_index", "Segregation index", "tab:red"),
        ),
    ),
    "diversity": PlotSpec(
        key="diversity",
        title="Spatial diversity",
        ylabel="Index",
        series=(
            SeriesSpec("spatial_entropy", "Spatial entropy", "tab:green"),
            SeriesSpec(
                "neighborhood_heterogeneity",
                "Neighbourhood heterogeneity",
                "tab:orange",
            ),
        ),
    ),
    "homelessness": PlotSpec(
        key="homelessness",
        title="Housing exclusion",
        ylabel="Fraction homeless",
        series=(SeriesSpec("homeless_fraction", "Homeless fraction", "tab:red"),),
        ylim=(0, 1),
    ),
    "mobility": PlotSpec(
        key="mobility",
        title="Income mobility",
        ylabel="Indicator",
        series=(
            SeriesSpec("income_mobility_indicator", "Income mobility", "tab:green"),
        ),
    ),
    "gentrification": PlotSpec(
        key="gentrification",
        title="Gentrification indicator",
        ylabel="Indicator",
        series=(
            SeriesSpec("gentrification_indicator", "Gentrification", "tab:orange"),
        ),
    ),
    "equilibrium_probability": PlotSpec(
        key="equilibrium_probability",
        title="Move probabilities: QRE and nearest Nash equilibrium",
        ylabel="Probability",
        series=(
            SeriesSpec("mean_qre_move_probability", "QRE move probability", "tab:blue"),
            SeriesSpec("mean_ne_move_probability", "Nearest-NE move probability", "tab:orange"),
        ),
        ylim=(0, 1),
    ),
    "equilibrium_gap": PlotSpec(
        key="equilibrium_gap",
        title="QRE–Nash deviation",
        ylabel="Mean absolute gap",
        series=(SeriesSpec("mean_qre_ne_move_gap", "QRE–NE gap", "tab:red"),),
    ),
    "ne_following": PlotSpec(
        key="ne_following",
        title="Nash-equilibrium following rate",
        ylabel="Rate",
        series=(SeriesSpec("ne_following_rate", "NE-following rate", "tab:green"),),
        ylim=(0, 1),
    ),
    "income_histogram": PlotSpec(
        key="income_histogram",
        title="Current household income distribution",
        ylabel="Households",
        xlabel="Income",
        series=(),
        kind="histogram",
    ),
    "rent_income_scatter": PlotSpec(
        key="rent_income_scatter",
        title="Current rent versus household income",
        ylabel="Rent",
        xlabel="Income",
        series=(),
        kind="scatter",
    ),
        "decision_heatmap": PlotSpec(
        key="decision_heatmap",
        title="Game outcomes by income quantile",
        ylabel="Income quantile",
        xlabel="Realized game outcome",
        series=(),
        kind="decision_heatmap",
        legend=False,
        height_px=430,
    ),

    # "decision_counts": PlotSpec(
    #     key="decision_counts",
    #     title="Cumulative game outcomes",
    #     ylabel="Number of games",
    #     xlabel="Realized game outcome",
    #     series=(),
    #     kind="decision_counts",
    #     height_px=430,
    # ),
        "decision_counts": PlotSpec(
        key="decision_counts",
        title=(
            "Overall game outcomes with "
            "income-adjusted composition"
        ),
        ylabel="Proportion of all games",
        xlabel="Realized game outcome",
        series=(),
        kind="decision_counts",
        height_px=430,
    ),
}

# Order in each tuple is the display order.
PLOT_VIEWS: dict[str, tuple[str, ...]] = {
    "General": (
        "income_histogram", "rent_income_scatter", "satisfaction", "movement_counts", "income_rent",
        "utility",
    ),
    "Gentrification metrics": (
        "gentrification", "segregation", "diversity",
        "inequality", "homelessness", "mobility",
    ),
    "Game metrics": (
        "decision_heatmap","decision_counts","equilibrium_probability", "equilibrium_gap",
        "ne_following", "movement_rate",
    ),
    "All": tuple(PLOT_SPECS),
}




# -----------------------------------------------------------------------------
# Page-specific text/statistics panels.
# Edit these independently for each experiment. A getter may be either:
#   * the name of a model attribute/method, or
#   * a callable taking the model.
# -----------------------------------------------------------------------------

STATS_VIEWS: dict[str, tuple[StatSection, ...]] = {
    "baseline": (
        StatSection(
            "Current model state",
            (
                StatSpec("Population", lambda m: len(m.agents), ".0f"),
                StatSpec("Residentially stable households", "satisfied_count", ".0f"),
                StatSpec("Percentage stable", "percentage_satisfied", ".1f", "%"),
                StatSpec("Mean fixed vision radius", "mean_vision", ".2f"),
            ),
        ),
        StatSection(
            "Economic state",
            (
                StatSpec("Mean household income", "city_mean_income"),
                StatSpec("Mean neighbourhood income", "mean_neighbor_income"),
                StatSpec("Mean rent", "mean_rent"),
                StatSpec("Mean raw household utility", "mean_utility"),
                StatSpec("Mean signed-CRRA household value", "mean_value"),
            ),
        ),
    ),
    "gentrification": (
        StatSection(
            "Gentrification experiment",
            (
                # Replace/add these with the exact model methods you use for m_i.
                StatSpec("Mean household income, m_i", "city_mean_income"),
                StatSpec("Mean neighbourhood income", "mean_neighbor_income"),
                StatSpec("Mean rent", "mean_rent"),
                StatSpec("Gentrification indicator", "gentrification_indicator"),
                StatSpec("Homeless fraction", "homeless_fraction", ".2%"),
            ),
        ),
        StatSection(
            "Spatial structure",
            (
                StatSpec("Moran's I", "moran_i", ".4f"),
                StatSpec("Segregation index", "segregation_index", ".4f"),
                StatSpec("Spatial entropy", "spatial_entropy", ".4f"),
                StatSpec("Neighbourhood heterogeneity", "neighborhood_heterogeneity", ".4f"),
                StatSpec("Gini coefficient", "gini_coefficient", ".4f"),
                StatSpec("Theil index", "theil_index", ".6f"),
            ),
        ),
    ),
    "game": (
        StatSection(
            "Relocation outcomes",
            (
                StatSpec("Successful moves", "successful_moves", ".0f"),
                StatSpec("Failed searches", "failed_searches", ".0f"),
                StatSpec("Voluntary stays", "voluntary_stays", ".0f"),
                StatSpec("Destination conflicts", "destination_conflicts", ".0f"),
                StatSpec("Realized movement rate", "movement_success_rate", ".1%"),
            ),
        ),
        StatSection(
            "QRE and Nash equilibrium",
            (
                StatSpec("Mean QRE move probability", "mean_qre_move_probability"),
                StatSpec("Mean QRE accept probability", "mean_qre_accept_probability"),
                StatSpec("Mean closest-NE move probability", "mean_ne_move_probability"),
                StatSpec("Mean absolute QRE-NE gap", "mean_qre_ne_move_gap"),
                StatSpec("NE-following rate", "ne_following_rate", ".1%"),
                StatSpec("QRE non-convergence count", "qre_nonconvergence_count", ".0f"),
            ),
        ),
    ),
}

# -----------------------------------------------------------------------------
# Experiment pages. Each page exposes only relevant controls; all remaining
# parameters use BASE_VALUES plus this experiment's overrides.
# -----------------------------------------------------------------------------

EXPERIMENTS: dict[str, ExperimentSpec] = {
    "baseline": ExperimentSpec(
        key="baseline",
        label="Baseline exploration",
        description="General model behaviour with a compact set of core controls.",
        adjustable_groups=("Basic", "Housing", "Income dynamics"),
        default_view="General",
        stats_view="baseline",
        # spatial_title="Baseline spatial state",
    ),
    "gentrification": ExperimentSpec(
        key="gentrification",
        label="Gentrification experiment",
        description="Explore rent adjustment, affordability, income dynamics, and spatial metrics.",
        adjustable_groups=("Basic", "Housing", "Income dynamics", "Mobility"),
        overrides={"rent_adjustment_rate": 0.03, "income_growth_scaling": 0.01},
        default_view="Gentrification metrics",
        stats_view="gentrification",
        # spatial_title="Gentrification spatial state",
    ),
    "game": ExperimentSpec(
        key="game",
        label="Relocation game experiment",
        description="Explore household and neighbourhood decision parameters.",
        adjustable_groups=("Basic", "Household behaviour", "Neighbourhood game", "Housing"),
        overrides={"keep_game_history": True},
        default_view="Game metrics",
        stats_view="game",
        # spatial_title="Relocation-game spatial state",
    ),
}


# -----------------------------------------------------------------------------
# Parameter construction
# -----------------------------------------------------------------------------

def build_model_params(experiment: ExperimentSpec) -> dict[str, Any]:
    values = dict(BASE_VALUES)
    values.update(experiment.overrides)

    adjustable_names = {
        name
        for group in experiment.adjustable_groups
        for name in PARAMETER_GROUPS[group]
    }

    params: dict[str, Any] = {}
    for name, value in values.items():
        if name in adjustable_names:
            factory = CONTROL_FACTORIES.get(name)
            if factory is None:
                raise KeyError(f"No control factory configured for adjustable parameter {name!r}")
            params[name] = factory(value)
        else:
            params[name] = value
    return params


def initial_model(experiment: ExperimentSpec) -> GentrificationModel:
    values = dict(BASE_VALUES)
    values.update(experiment.overrides)
    return GentrificationModel(**values)


# -----------------------------------------------------------------------------
# Data helpers
# -----------------------------------------------------------------------------

def model_history(model) -> Mapping[str, Sequence[float]]:
    """Return DataCollector model history as a column mapping."""
    collector = getattr(model, "datacollector", None)
    if collector is None:
        return {}

    try:
        frame = collector.get_model_vars_dataframe()
    except Exception:
        return {}

    if frame is None or frame.empty:
        return {}
    return {column: frame[column].tolist() for column in frame.columns}


def safe_current_metric(model, metric: str) -> float | None:
    value = getattr(model, metric, None)
    if callable(value):
        try:
            value = value()
        except Exception:
            return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None

def game_decision_matrix(
    model,
) -> tuple[np.ndarray, list[str], list[str]]:
    """Count game outcomes by recorded income quantile.

    Rows correspond to income groups.
    Columns correspond to realized game outcomes.
    """
    quantile_labels = list(
        INCOME_QUANTILE_LABELS
    )
    outcomes = list(GAME_OUTCOMES)

    matrix = np.zeros(
        (
            len(quantile_labels),
            len(outcomes),
        ),
        dtype=int,
    )

    outcome_columns = {
        outcome: index
        for index, outcome in enumerate(outcomes)
    }

    history = getattr(
        model,
        "game_history",
        [],
    )

    for record in history:
        if not isinstance(record, Mapping):
            continue

        outcome = record.get("outcome")

        if outcome not in outcome_columns:
            continue

        try:
            income_group = int(
                record["income_group"]
            )
        except (KeyError, TypeError, ValueError):
            continue

        if not 0 <= income_group < len(
            quantile_labels
        ):
            continue

        column = outcome_columns[outcome]

        matrix[
            income_group,
            column,
        ] += 1

    display_outcomes = [
        GAME_OUTCOME_LABELS[outcome]
        for outcome in outcomes
    ]

    return (
        matrix,
        quantile_labels,
        display_outcomes,
    )

# -----------------------------------------------------------------------------
# Responsive custom plots
# -----------------------------------------------------------------------------

@solara.component
def ResponsiveMetricPlot(model, spec: PlotSpec):
    update_counter.get()
    size = solara.use_reactive({"width": 600, "height": spec.height_px})
    width = max(int(size.value.get("width", 600)), 280)
    height = max(int(size.value.get("height", spec.height_px)), 230)
    dpi = 100

    fig = Figure(figsize=(width / dpi, height / dpi), dpi=dpi)
    ax = fig.subplots()

    if spec.kind == "histogram":
        housed_incomes = []
        homeless_incomes = []

        for agent in model.agents:
            income = float(agent.income)
            rent = float(agent.cell.rent)

            # Prefer a model/agent property if you already have one.
            is_homeless = (
                income
                < rent * float(model.affordability_share)
            )

            if is_homeless:
                homeless_incomes.append(income)
            else:
                housed_incomes.append(income)

        all_incomes = housed_incomes + homeless_incomes

        if all_incomes:
            bins = np.histogram_bin_edges(
                all_incomes,
                bins="auto",
            )

            ax.hist(
                [
                    housed_incomes,
                    homeless_incomes,
                ],
                bins=bins,
                stacked=True,
                label=[
                    "Housed households",
                    "Homeless households",
                ],
                color=[
                    "#4C78A8",
                    "#D62728",
                ],
                edgecolor="white",
                linewidth=0.6,
            )

            homeless_count = len(homeless_incomes)
            total_count = len(all_incomes)
            homeless_fraction = homeless_count / total_count

            ax.text(
                0.98,
                0.95,
                (
                    f"Homeless: {homeless_count}\n"
                    f"Share: {homeless_fraction:.1%}"
                ),
                transform=ax.transAxes,
                ha="right",
                va="top",
                fontsize=9,
                bbox={
                    "facecolor": "white",
                    "edgecolor": "0.8",
                    "alpha": 0.85,
                    "boxstyle": "round,pad=0.35",
                },
            )

    elif spec.kind == "decision_heatmap":
        (
            matrix,
            quantile_labels,
            outcome_labels,
        ) = game_decision_matrix(model)

        if matrix.sum() == 0:
            ax.text(
                0.5,
                0.5,
                "No game outcomes recorded yet.",
                transform=ax.transAxes,
                ha="center",
                va="center",
            )

            ax.set_xticks([])
            ax.set_yticks([])

        else:
            row_totals = matrix.sum(
                axis=1,
                keepdims=True,
            )

            proportions = np.divide(
                matrix,
                row_totals,
                out=np.zeros_like(
                    matrix,
                    dtype=float,
                ),
                where=row_totals != 0,
            )

            image = ax.imshow(
                proportions,
                aspect="auto",
                cmap="RdPu",
                vmin=0.0,
                vmax=1.0,
            )

            ax.set_xticks(
                np.arange(
                    len(outcome_labels)
                )
            )
            ax.set_xticklabels(
                outcome_labels,
                rotation=22,
                ha="right",
            )

            ax.set_yticks(
                np.arange(
                    len(quantile_labels)
                )
            )
            ax.set_yticklabels(
                quantile_labels
            )

            for row in range(
                matrix.shape[0]
            ):
                for column in range(
                    matrix.shape[1]
                ):
                    count = matrix[
                        row,
                        column,
                    ]

                    share = proportions[
                        row,
                        column,
                    ]

                    ax.text(
                        column,
                        row,
                        (
                            f"{count}\n"
                            f"{share:.0%}"
                        ),
                        ha="center",
                        va="center",
                        fontsize=8,
                        color=(
                            "white"
                            if share >= 0.55
                            else "black"
                        ),
                    )

            colorbar = fig.colorbar(
                image,
                ax=ax,
                fraction=0.045,
                pad=0.04,
            )

            colorbar.set_label(
                "Share within income quantile"
            )

    elif spec.kind == "decision_counts":
        (
            matrix,
            quantile_labels,
            outcome_labels,
        ) = game_decision_matrix(model)

        total_games = matrix.sum()

        if total_games == 0:
            ax.text(
                0.5,
                0.5,
                "No matching game outcomes recorded yet.",
                transform=ax.transAxes,
                ha="center",
                va="center",
                fontsize=11,
            )

            ax.set_xticks([])
            ax.set_yticks([])

        else:
            # -------------------------------------------------------------
            # 1. Overall proportion of every outcome.
            #
            # Shape: (number of outcomes,)
            # -------------------------------------------------------------
            outcome_totals = matrix.sum(axis=0)

            outcome_proportions = (
                outcome_totals / total_games
            )

            # -------------------------------------------------------------
            # 2. Correct for unequal numbers of games played by each
            #    income group.
            #
            # matrix[i, j] / group_totals[i] gives:
            #
            # P(outcome j | income group i)
            # -------------------------------------------------------------
            group_totals = matrix.sum(
                axis=1,
                keepdims=True,
            )

            outcome_rates_by_group = np.divide(
                matrix,
                group_totals,
                out=np.zeros_like(
                    matrix,
                    dtype=float,
                ),
                where=group_totals != 0,
            )

            # -------------------------------------------------------------
            # 3. For each outcome, normalize the income-group rates so
            #    their weights sum to one.
            #
            # These are relative income-group propensities, not raw shares
            # of observations.
            # -------------------------------------------------------------
            rate_totals_by_outcome = (
                outcome_rates_by_group.sum(
                    axis=0,
                    keepdims=True,
                )
            )

            income_weights_within_outcome = np.divide(
                outcome_rates_by_group,
                rate_totals_by_outcome,
                out=np.zeros_like(
                    outcome_rates_by_group,
                    dtype=float,
                ),
                where=rate_totals_by_outcome != 0,
            )

            # -------------------------------------------------------------
            # 4. Scale each income segment so that the full bar height
            #    remains equal to the overall outcome proportion.
            # -------------------------------------------------------------
            segment_heights = (
                income_weights_within_outcome
                * outcome_proportions[
                    np.newaxis,
                    :
                ]
            )

            x_positions = np.arange(
                len(outcome_labels)
            )

            bottoms = np.zeros(
                len(outcome_labels),
                dtype=float,
            )

            for group_index, (
                group_label,
                color,
            ) in enumerate(
                zip(
                    quantile_labels,
                    COLORS,
                    strict=False,
                )
            ):
                heights = segment_heights[
                    group_index,
                    :,
                ]

                bars = ax.bar(
                    x_positions,
                    heights,
                    bottom=bottoms,
                    color=color,
                    edgecolor="white",
                    linewidth=0.7,
                    label=group_label,
                )

                # Display the corrected income-group contribution inside
                # each outcome bar.
                for outcome_index, bar in enumerate(
                    bars
                ):
                    segment_height = heights[
                        outcome_index
                    ]

                    if segment_height <= 0:
                        continue

                    corrected_share = (
                        income_weights_within_outcome[
                            group_index,
                            outcome_index,
                        ]
                    )

                    # Avoid labels in very small segments.
                    if corrected_share < 0.06:
                        continue

                    ax.text(
                        bar.get_x()
                        + bar.get_width() / 2,
                        bottoms[outcome_index]
                        + segment_height / 2,
                        f"{corrected_share:.0%}",
                        ha="center",
                        va="center",
                        fontsize=8,
                        color=(
                            "white"
                            if corrected_share >= 0.25
                            else "black"
                        ),
                    )

                bottoms += heights

            ax.set_xticks(x_positions)

            ax.set_xticklabels(
                outcome_labels,
                rotation=20,
                ha="right",
            )

            ax.set_ylim(
                0.0,
                max(
                    0.05,
                    float(
                        outcome_proportions.max()
                    )
                    * 1.20,
                ),
            )

            ax.yaxis.set_major_formatter(
                plt.FuncFormatter(
                    lambda value, position: (
                        f"{value:.0%}"
                    )
                )
            )

            # Overall outcome proportion above each complete bar.
            for (
                position,
                outcome_proportion,
                outcome_total,
            ) in zip(
                x_positions,
                outcome_proportions,
                outcome_totals,
                strict=False,
            ):
                ax.text(
                    position,
                    outcome_proportion,
                    (
                        f"{outcome_proportion:.1%}\n"
                        f"n={int(outcome_total)}"
                    ),
                    ha="center",
                    va="bottom",
                    fontsize=8,
                    fontweight="bold",
                )

            ax.legend(
                title="Income quantile",
                bbox_to_anchor=(1.02, 1.0),
                loc="upper left",
                frameon=False,
                fontsize=8,
            )

    # elif spec.kind == "decision_counts":
    #     (
    #         matrix,
    #         quantile_labels,
    #         outcome_labels,
    #     ) = game_decision_matrix(model)

    #     if matrix.sum() == 0:
    #         ax.text(
    #             0.5,
    #             0.5,
    #             "No game outcomes recorded yet.",
    #             transform=ax.transAxes,
    #             ha="center",
    #             va="center",
    #         )

    #         ax.set_xticks([])
    #         ax.set_yticks([])

    #     else:
    #         x_positions = np.arange(
    #             len(outcome_labels)
    #         )

    #         bottoms = np.zeros(
    #             len(outcome_labels),
    #             dtype=float,
    #         )

    #         for group_index, (
    #             group_label,
    #             color,
    #         ) in enumerate(
    #             zip(
    #                 quantile_labels,
    #                 COLORS,
    #                 strict=False,
    #             )
    #         ):
    #             counts = matrix[
    #                 group_index,
    #                 :,
    #             ]

    #             ax.bar(
    #                 x_positions,
    #                 counts,
    #                 bottom=bottoms,
    #                 color=color,
    #                 edgecolor="white",
    #                 linewidth=0.6,
    #                 label=group_label,
    #             )

    #             bottoms += counts

    #         ax.set_xticks(x_positions)
    #         ax.set_xticklabels(
    #             outcome_labels,
    #             rotation=20,
    #             ha="right",
    #         )

    #         totals = matrix.sum(axis=0)

    #         for position, total in zip(
    #             x_positions,
    #             totals,
    #             strict=False,
    #         ):
    #             ax.text(
    #                 position,
    #                 total,
    #                 str(int(total)),
    #                 ha="center",
    #                 va="bottom",
    #                 fontsize=9,
    #                 fontweight="bold",
    #             )

    #         ax.legend(
    #             title="Income quantile",
    #             bbox_to_anchor=(
    #                 1.02,
    #                 1.0,
    #             ),
    #             loc="upper left",
    #             frameon=False,
    #             fontsize=8,
    #         )


    elif spec.kind == "scatter":
        incomes = np.asarray([float(agent.income) for agent in model.agents])
        rents = np.asarray([float(agent.cell.rent) for agent in model.agents])
        ax.scatter(incomes, rents, alpha=0.65, label="Households")
        if len(incomes):
            x = np.array([incomes.min(), incomes.max()])
            ax.plot(
                x,
                x * float(model.affordability_share),
                linestyle="--",
                label="Affordability threshold",
            )

    else:
        history = model_history(model)
        for series in spec.series:
            values = history.get(series.metric, [])
            if not values:
                current = safe_current_metric(model, series.metric)
                values = [] if current is None else [current]
            ax.plot(
                range(len(values)),
                values,
                label=series.label,
                color=series.color,
                linestyle=series.linestyle,
            )

    ax.set_title(spec.title, pad=10)
    ax.set_xlabel(spec.xlabel)
    ax.set_ylabel(spec.ylabel)
    if spec.ylim is not None:
        ax.set_ylim(*spec.ylim)
    ax.grid(alpha=0.2)
    if (
        spec.legend
        and spec.kind != "decision_counts"
    ):
        handles, labels = (
            ax.get_legend_handles_labels()
        )

        if handles:
            ax.legend(
                loc="best",
                frameon=False,
            )
        if handles:
            ax.legend(loc="best", frameon=False)
    fig.tight_layout()

    with ViewListener(
        view_data=size.value,
        on_view_data=size.set,
        style={
            "width": "100%",
            "height": f"{spec.height_px}px",
            "min-height": "230px",
            "overflow": "hidden",
        },
    ):
        solara.FigureMatplotlib(fig)


@solara.component
def PlotGallery(model, default_view: str = "General"):
    selected_view = solara.use_reactive(default_view)
    columns = solara.use_reactive(2)

    with solara.Card("Plots", style={"width": "100%"}):
        with solara.Row(gap="12px"):
            solara.Select(
                label="Plot view",
                value=selected_view,
                values=list(PLOT_VIEWS),
                style={"max-width": "320px"},
            )
            solara.Select(
                label="Columns",
                value=columns,
                values=[1, 2, 3],
                style={"max-width": "160px"},
            )

        plot_keys = PLOT_VIEWS[selected_view.value]
        width = 12 // int(columns.value)
        with solara.ColumnsResponsive(12, large=[width] * len(plot_keys)):
            for key in plot_keys:
                with solara.Card(style={"overflow": "hidden"}):
                    ResponsiveMetricPlot(model, PLOT_SPECS[key])


# -----------------------------------------------------------------------------
# Spatial component with property-layer selector
# -----------------------------------------------------------------------------

@solara.component
def SpatialView(model, title: str = "Spatial state"):
    update_counter.get()
    size = solara.use_reactive({"width": 620, "height": 560})
    selected_layer = solara.use_reactive("rent")

    width_px = max(int(size.value.get("width", 620)), 300)
    total_height_px = max(int(size.value.get("height", 560)), 330)
    plot_height_px = max(total_height_px - 95, 230)
    dpi = 100

    def agent_portrayal(agent) -> AgentPortrayalStyle:
        rent = float(agent.cell.rent)
        income = float(agent.income)
        homeless = income < rent * float(agent.model.affordability_share)
        return AgentPortrayalStyle(
            x=agent.cell.coordinate[0],
            y=agent.cell.coordinate[1],
            color=agent.colour,
            marker="s" if homeless else ("o" if agent.satisfied else "^"),
            size=80,
            zorder=3 if agent.satisfied else 2,
        )

    def property_layer_portrayal(layer):
        if selected_layer.value == "None" or layer.name != selected_layer.value:
            return PropertyLayerStyle(colormap="PuBu", alpha=0.0, colorbar=False)
        return PropertyLayerStyle(colormap="PuBu", alpha=0.8, colorbar=True)

    def post_process(ax):
        fig = ax.figure
        fig.set_size_inches(
            max(width_px - 20, 280) / dpi,
            max(plot_height_px - 10, 220) / dpi,
            forward=True,
        )
        # Do not use tight_layout here: Mesa may already have made a colorbar.
        has_colorbar = len(fig.axes) > 1
        fig.subplots_adjust(
            left=0.08,
            right=0.84 if has_colorbar else 0.97,
            bottom=0.18,
            top=0.90,
        )
        # ax.set_title(title, pad=10)

        # marker_handles = [
        #     Line2D([0], [0], marker="o", linestyle="None", color="black",
        #            markerfacecolor="white", markersize=8, label="Stable housed household"),
        #     Line2D([0], [0], marker="^", linestyle="None", color="black",
        #            markerfacecolor="white", markersize=8, label="Unstable housed household"),
        #     Line2D([0], [0], marker="s", linestyle="None", color="black",
        #            markerfacecolor="white", markersize=8, label="Housing-excluded household"),
        # ]
        # ax.legend(
        #     handles=marker_handles,
        #     loc="upper center",
        #     bbox_to_anchor=(0.5, -0.10),
        #     ncol=3,
        #     frameon=False,
        #     fontsize=8,
        # )

    renderer = SpaceRenderer(model, backend="matplotlib")
    renderer.setup_agents(agent_portrayal)
    renderer.setup_propertylayer(property_layer_portrayal)
    renderer.post_process = post_process
    renderer.render()

    with ViewListener(
        view_data=size.value,
        on_view_data=size.set,
        style={
            "width": "100%",
            "height": "100%",
            "min-height": "360px",
            "overflow": "hidden",
        },
    ):
        with solara.Column(gap="4px", style={"width": "100%", "height": "100%"}):
            solara.Select(
                label="Property layer",
                value=selected_layer,
                values=["rent", "mean_neighbor_income", "neighbor_income_variance", "None"],
                style={"max-width": "340px"},
            )
            with solara.Row(
                gap="12px",
                style={
                    "width": "100%",
                    "height": "100%",
                    "align-items": "stretch",
                    "overflow": "hidden",
                },
            ):
                # Main spatial plot
                with solara.Div(
                    style={
                        "width": "calc(100% - 230px)",
                        "height": "100%",
                        "min-width": "0",
                        "overflow": "hidden",
                    },
                ):
                    SpaceRendererComponent(
                        model,
                        renderer,
                    )

                # Legend panel
                with solara.Card(
                    style={
                        "width": "218px",
                        "height": "100%",
                        "overflow-y": "auto",
                        "padding": "10px",
                        "flex": "0 0 218px",
                    },
                ):
                    SpatialLegend(model)

@solara.component
def NonQuantileColoringSpatialView(model):
    update_counter.get()
    size = solara.use_reactive({"width": 620, "height": 560})
    selected_layer = solara.use_reactive("rent")

    width_px = max(int(size.value.get("width", 620)), 300)
    total_height_px = max(int(size.value.get("height", 560)), 330)
    plot_height_px = max(total_height_px - 95, 230)
    dpi = 100

    max_income = max((float(agent.income) for agent in model.agents), default=1.0)
    min_income = min((float(agent.income) for agent in model.agents), default=0.0)
    for agent in model.agents:
        agent.colour = generate_vibrant_red_blue_gradient(
            float(agent.income), min_income, max_income
        )

    def agent_portrayal(agent) -> AgentPortrayalStyle:
        rent = float(agent.cell.rent)
        income = float(agent.income)
        homeless = income < rent * float(agent.model.affordability_share)
        return AgentPortrayalStyle(
            x=agent.cell.coordinate[0],
            y=agent.cell.coordinate[1],
            color=agent.colour,
            marker="s" if homeless else ("o" if agent.satisfied else "^"),
            size=80,
            zorder=3 if agent.satisfied else 2,
        )

    def property_layer_portrayal(layer):
        if selected_layer.value == "None" or layer.name != selected_layer.value:
            return PropertyLayerStyle(colormap="PuBu", alpha=0.0, colorbar=False)
        return PropertyLayerStyle(colormap="PuBu", alpha=0.8, colorbar=True)

    def post_process(ax):
        fig = ax.figure
        fig.set_size_inches(
            max(width_px - 20, 280) / dpi,
            max(plot_height_px - 10, 220) / dpi,
            forward=True,
        )
        # Do not use tight_layout here: Mesa may already have made a colorbar.
        has_colorbar = len(fig.axes) > 1
        fig.subplots_adjust(
            left=0.08,
            right=0.84 if has_colorbar else 0.97,
            bottom=0.08,
            top=0.96,
        )

    renderer = SpaceRenderer(model, backend="matplotlib")
    renderer.setup_agents(agent_portrayal)
    renderer.setup_propertylayer(property_layer_portrayal)
    renderer.post_process = post_process
    renderer.render()

    with ViewListener(
        view_data=size.value,
        on_view_data=size.set,
        style={
            "width": "100%",
            "height": "100%",
            "min-height": "360px",
            "overflow": "hidden",
        },
    ):
        with solara.Column(gap="4px", style={"width": "100%", "height": "100%"}):
            solara.Select(
                label="Property layer",
                value=selected_layer,
                values=["rent", "mean_neighbor_income", "neighbor_income_variance", "None"],
                style={"max-width": "340px"},
            )
            with solara.Div(style={"overflow": "hidden", "min-height": "0"}):
                SpaceRendererComponent(model, renderer)


# -----------------------------------------------------------------------------
# Configurable page-specific statistics panel
# -----------------------------------------------------------------------------

def resolve_stat(model, getter):
    try:
        value = getter(model) if callable(getter) else getattr(model, getter)
        return value() if callable(value) else value
    except Exception:
        return None


def format_stat(value, fmt: str, suffix: str) -> str:
    if value is None:
        return "—"
    try:
        return f"{value:{fmt}}{suffix}"
    except (TypeError, ValueError):
        return f"{value}{suffix}"


@solara.component
def ExperimentStatistics(model, stats_view: str):
    update_counter.get()
    sections = STATS_VIEWS[stats_view]

    markdown_parts: list[str] = []
    for section in sections:
        markdown_parts.append(f"### {section.title}")
        for item in section.items:
            value = resolve_stat(model, item.getter)
            markdown_parts.append(
                f"**{item.label}:** {format_stat(value, item.fmt, item.suffix)}  "
            )
        markdown_parts.append("")

    solara.Markdown("\n".join(markdown_parts))



# -----------------------------------------------------------------------------
# Proper labeling
# -----------------------------------------------------------------------------

def income_group_labels(
    thresholds: np.ndarray,
) -> list[str]:
    if len(thresholds) == 6:
        return [
            "0.15",
            "0.35",
            "0.55",
            "0.70",
            "0.85",
            "0.99",
        ]



    t1, t2, t3, t4, t5, t6 = thresholds

    return [
        f"≤ {t1:.2f}",
        f"{t1:.2f} – {t2:.2f}",
        f"{t2:.2f} – {t3:.2f}",
        f"{t3:.2f} – {t4:.2f}",
        f"{t4:.2f} – {t5:.2f}",
        f"{t5:.2f} – {t6:.2f}",
        f"> {t6:.2f}",
    ]

@solara.component
def SpatialLegend(model):
    thresholds = getattr(
        model,
        "income_thresholds",
        [],
    )

    thresholds = np.asarray(
        thresholds,
        dtype=float,
    )

    # solara.Markdown("### Legend")

    solara.Markdown("**Household status**")

    marker_items = [
        ("●", "Stable household"),
        ("▲", "Unstable household"),
        ("■", "Homeless"),
    ]

    for symbol, label in marker_items:
        with solara.Row(
            gap="8px",
            style={
                "align-items": "center",
                "margin-bottom": "4px",
            },
        ):
            solara.Text(
                symbol,
                style={
                    "font-size": "22px",
                    "width": "24px",
                    "text-align": "center",
                },
            )
            solara.Text(label)

    solara.Markdown("**Relative income group, quantile thresholds**")

    labels = income_group_labels(thresholds)

    for color, label in zip(
        COLORS,
        labels,
        strict=False,
    ):
        with solara.Row(
            gap="8px",
            style={
                "align-items": "center",
                "margin-bottom": "3px",
            },
        ):
            solara.Div(
                style={
                    "width": "18px",
                    "height": "18px",
                    "background-color": color,
                    "border": "1px solid #555",
                    "border-radius": "3px",
                    "flex": "0 0 auto",
                },
            )

            solara.Text(
                label,
                style={
                    "font-size": "0.85rem",
                },
            )

# -----------------------------------------------------------------------------
# Page factory and routes
# -----------------------------------------------------------------------------

def make_experiment_page(experiment_key: str):
    experiment = EXPERIMENTS[experiment_key]

    @solara.component
    def ExperimentPage():
        solara.Title(experiment.label)
        solara.Markdown(f"## {experiment.label}\n{experiment.description}")

        model = initial_model(experiment)
        params = build_model_params(experiment)

        # Each item remains a Mesa dashboard tile. SpatialView and PlotGallery
        # are internally responsive and do not use make_plot_component.
        dashboard = SolaraViz(
            model,
            components=[
                lambda m: SpatialView(m, title=""),
                lambda m: PlotGallery(m, default_view=experiment.default_view),
                lambda m: ExperimentStatistics(m, stats_view=experiment.stats_view),
            ],
            model_params=params,
            name=experiment.label,
        )
        dashboard

    ExperimentPage.__name__ = f"{experiment_key.title()}ExperimentPage"
    return ExperimentPage

def make_schelling_exp_page():
    @solara.component
    def SchellingStablePage():
        solara.Title("Schelling Stable Layout Experiment")
        solara.Markdown(
            "This page runs a Schelling model to generate a stable layout and then "
            "populates the Gentrification model with agents matching that layout."
        )

        # Initialize the SchellingGridExperiment model
        model = SchellingGridExperiment(
            width=20,
            height=20,
            density=BASE_VALUES["density"],
            neighborhood_radius=BASE_VALUES["neighborhood_radius"],
            initial_income_min=BASE_VALUES["initial_income_min"],
            initial_income_max=BASE_VALUES["initial_income_max"],
            affordability_share=BASE_VALUES["affordability_share"],
            income_similarity_min=BASE_VALUES["income_similarity_min"],
            income_similarity_max=BASE_VALUES["income_similarity_max"],
            discount_factor_min=BASE_VALUES["discount_factor_min"],
            discount_factor_max=BASE_VALUES["discount_factor_max"],
            risk_aversion_min=BASE_VALUES["risk_aversion_min"],
            risk_aversion_max=BASE_VALUES["risk_aversion_max"],
            rationality_min=BASE_VALUES["rationality_min"],
            rationality_max=BASE_VALUES["rationality_max"],
            satisficing_threshold=BASE_VALUES["satisficing_threshold"],
            vision_income_scale=BASE_VALUES["vision_income_scale"],
            maximum_vision_radius=BASE_VALUES["maximum_vision_radius"],
            steps_until_satisfied=BASE_VALUES["steps_until_satisfied"],
            rent_adjustment_rate=BASE_VALUES["rent_adjustment_rate"],
            income_growth_scaling=BASE_VALUES["income_growth_scaling"],
            income_volatility=BASE_VALUES["income_volatility"],
            moving_cost=BASE_VALUES["moving_cost"],
            rejection_cost=BASE_VALUES["rejection_cost"],
            neighborhood_risk_aversion=BASE_VALUES["neighborhood_risk_aversion"],
            neighborhood_rationality=BASE_VALUES["neighborhood_rationality"],
            qre_tolerance=BASE_VALUES["qre_tolerance"],
            qre_maximum_iterations=BASE_VALUES["qre_maximum_iterations"],
            qre_damping=BASE_VALUES["qre_damping"],
            keep_game_history=BASE_VALUES["keep_game_history"]
        )

        # Build the parameter controls for this experiment
        params = build_model_params(EXPERIMENTS["baseline"])  # Use baseline params for now

        # Create the dashboard with the spatial view and plots
        dashboard = SolaraViz(
            model,
            components=[
                lambda m: NonQuantileColoringSpatialView(m),
                lambda m: PlotGallery(m, default_view="General"),
                lambda m: ExperimentStatistics(m, stats_view=EXPERIMENTS["baseline"].stats_view),
            ],
            model_params=params,
            name="Schelling Stable Layout Experiment",
        )
        dashboard

    return SchellingStablePage

HomePage = make_experiment_page("baseline")
GentrificationPage = make_experiment_page("gentrification")
GamePage = make_experiment_page("game")
ExpPage = make_schelling_exp_page()

routes = [
    solara.Route(path="/", component=HomePage, label="Baseline"),
    solara.Route(path="gentrification", component=GentrificationPage, label="Gentrification"),
    solara.Route(path="game", component=GamePage, label="Game"),
    solara.Route(path="schellingstable", component=ExpPage, label="Schelling Stable"),
]
