from __future__ import annotations


# """""
# * PLOT_SPECS: what every plot displays.
# * PLOT_VIEWS: which plots appear in each dropdown view and their order.
# * EXPERIMENTS: which parameters are adjustable on each URL/page.
# """

from dataclasses import dataclass, field
from typing import Any, Callable, Iterable, Mapping, Sequence

import math
import matplotlib.pyplot as plt
from matplotlib.figure import Figure
import numpy as np
import solara

from mesa.visualization import Slider, SolaraViz, SpaceRenderer
from mesa.visualization.components import AgentPortrayalStyle, PropertyLayerStyle
from mesa.visualization.solara_viz import SpaceRendererComponent
from mesa.visualization.utils import update_counter

from src.project.model import GentrificationModel


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
class ExperimentSpec:
    key: str
    label: str
    description: str
    adjustable_groups: tuple[str, ...]
    overrides: Mapping[str, Any] = field(default_factory=dict)
    default_view: str = "General"


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
}

# Order in each tuple is the display order.
PLOT_VIEWS: dict[str, tuple[str, ...]] = {
    "General": (
        "satisfaction", "movement_counts", "income_rent",
        "utility", "income_histogram", "rent_income_scatter",
    ),
    "Gentrification metrics": (
        "gentrification", "segregation", "diversity",
        "inequality", "homelessness", "mobility",
    ),
    "Game metrics": (
        "equilibrium_probability", "equilibrium_gap",
        "ne_following", "movement_rate",
    ),
    "All": tuple(PLOT_SPECS),
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
    ),
    "gentrification": ExperimentSpec(
        key="gentrification",
        label="Gentrification experiment",
        description="Explore rent adjustment, affordability, income dynamics, and spatial metrics.",
        adjustable_groups=("Basic", "Housing", "Income dynamics", "Mobility"),
        overrides={"rent_adjustment_rate": 0.03, "income_growth_scaling": 0.01},
        default_view="Gentrification metrics",
    ),
    "game": ExperimentSpec(
        key="game",
        label="Relocation game experiment",
        description="Explore household and neighbourhood decision parameters.",
        adjustable_groups=("Basic", "Household behaviour", "Neighbourhood game", "Housing"),
        overrides={"keep_game_history": True},
        default_view="Game metrics",
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
        values = [float(agent.income) for agent in model.agents]
        ax.hist(values, bins="auto", label="Households")

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
    if spec.legend:
        handles, labels = ax.get_legend_handles_labels()
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
def SpatialView(model):
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
# Compact current-state panel
# -----------------------------------------------------------------------------

@solara.component
def CurrentState(model):
    update_counter.get()
    ratio = model.rent_income_timescale_ratio()
    ratio_text = f"{ratio:.3f}" if math.isfinite(ratio) else "∞"
    solara.Markdown(
        f"""
### Current state

**Population:** {len(model.agents)}  
**Stable households:** {model.percentage_satisfied():.1f}%  
**Successful moves:** {model.successful_moves}  
**Failed searches:** {model.failed_searches}  
**Mean income:** {model.city_mean_income():.3f}  
**Mean rent:** {model.mean_rent():.3f}  
**Mean utility:** {model.mean_utility():.3f}  
**Rent/income adjustment ratio:** {ratio_text}
"""
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
                SpatialView,
                lambda m: PlotGallery(m, default_view=experiment.default_view),
                CurrentState,
            ],
            model_params=params,
            name=experiment.label,
        )
        dashboard

    ExperimentPage.__name__ = f"{experiment_key.title()}ExperimentPage"
    return ExperimentPage


HomePage = make_experiment_page("baseline")
GentrificationPage = make_experiment_page("gentrification")
GamePage = make_experiment_page("game")

routes = [
    solara.Route(path="/", component=HomePage, label="Baseline"),
    solara.Route(path="gentrification", component=GentrificationPage, label="Gentrification"),
    solara.Route(path="game", component=GamePage, label="Game"),
]
