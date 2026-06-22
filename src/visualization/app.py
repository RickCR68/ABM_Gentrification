from __future__ import annotations

import solara
import math
from mesa.visualization import (
    Slider,
    SolaraViz,
    SpaceRenderer,
    make_plot_component,
)
from mesa.visualization.components import AgentPortrayalStyle, PropertyLayerStyle
from matplotlib.figure import Figure
from mesa.visualization.utils import update_counter

from src.project.model import GentrificationModel
from src.utils.helpers import (
    generate_vibrant_red_blue_gradient,
)


def normalize_income(income: float, max_income: float) -> float:
    """Map non-negative income to the interval [0, 1)."""
    income = max(income/max_income, 0.0)

    return income


def agent_portrayal(agent) -> AgentPortrayalStyle:
    """Define how a household is displayed."""
    max_income = max(agent.model.agents, key=lambda a: a.income).income
    income_normalized = normalize_income(
        agent.income, max_income
    )

    hex_color = generate_vibrant_red_blue_gradient(
        income_normalized
    )

    return AgentPortrayalStyle(
        # Explicit coordinates are retained because the installed
        # Mesa renderer appears to require them.
        x=agent.cell.coordinate[0],
        y=agent.cell.coordinate[1],
        color=hex_color,
        marker="o" if agent.satisfied else "x",
        size=80,
        zorder=3 if agent.satisfied else 2,
    )


selected_layer = solara.reactive("rent")

def property_layer_portrayal(layer):
    if layer.name == selected_layer.value:
        return PropertyLayerStyle(
            color="blue", alpha=0.8, colorbar=True
        )
    # elif layer.name == "mean_neighbor_income":
    #     return PropertyLayerStyle(
    #         color="green", alpha=0.8, colorbar=True, vmin=0, vmax=10
    #     )
    return PropertyLayerStyle(
            color="blue", alpha=0.0, colorbar=False
        )

def layer_selector(layer):
    layers = ["rent", "mean_neighbor_income", "None"]  # Your actual layer names
    solara.Select(label="Select Property Layer",
                  value=selected_layer,
                  values=layers)


def get_model_statistics(model):
    """Display current model and game statistics."""
    ratio = model.rent_income_timescale_ratio()

    ratio_text = (
        f"{ratio:.3f}"
        if math.isfinite(ratio)
        else "∞"
    )

    return solara.Markdown(
        f"""
### Current model state

**Population:** {len(model.agents)}  
**Residentially stable households:** {model.satisfied_count}  
**Percentage stable:** {model.percentage_satisfied():.1f}%  
**Mean fixed vision radius:** {model.mean_vision():.2f}  

### Relocation

 
**Successful moves:** {model.successful_moves}  
**Failed searches:** {model.failed_searches}  
**Voluntary stays after game:** {model.voluntary_stays}  
**Destination conflicts:** {model.destination_conflicts}  
**Realized movement rate:** {model.movement_success_rate():.1%}  

### Game outcomes

**Move–accept:** {model.move_accept_outcomes}  
**Move–reject:** {model.move_reject_outcomes}  
**Stay–accept:** {model.stay_accept_outcomes}  
**Stay–reject:** {model.stay_reject_outcomes}  
**QRE non-convergence count:** {model.qre_nonconvergence_count}  

### QRE and Nash equilibrium

**Mean QRE move probability:** {model.mean_qre_move_probability():.3f}  
**Mean QRE accept probability:** {model.mean_qre_accept_probability():.3f}  
**Mean closest-NE move probability:** {model.mean_ne_move_probability():.3f}  
**Mean absolute QRE–NE gap:** {model.mean_qre_ne_move_gap():.3f}  
**NE-following rate (gap ≤ 0.05):** {model.ne_following_rate():.1%}  
**Mean change in neighbourhood utility:** {model.mean_delta_neighborhood_utility():.3f}  

### Economic state

**Mean household income:** {model.city_mean_income():.3f}  
**Mean nonempty neighbourhood income:** {model.mean_neighbor_income():.3f}  
**Mean rent:** {model.mean_rent():.3f}  
**Rent/income adjustment-rate ratio:** {ratio_text}  

### Utility and value

**Mean raw household utility:** {model.mean_utility():.3f}  
**Mean signed-CRRA household value:** {model.mean_value():.3f}
"""
    )



model_params = {
    "rng": {
        "type": "InputText",
        "value": 42,
        "label": "Random seed",
    },

    # ---------------------------------------------------------
    # Spatial parameters
    # ---------------------------------------------------------
    "width": 20,
    "height": 20,

    "density": Slider(
        "Agent density",
        value=0.8,
        min=0.1,
        max=0.95,
        step=0.05,
    ),

    "neighborhood_radius": Slider(
        "Neighbourhood radius",
        value=1,
        min=1,
        max=4,
        step=1,
    ),

    # ---------------------------------------------------------
    # Initial income distribution
    # ---------------------------------------------------------
    "initial_income_min": Slider(
        "Minimum initial income",
        value=0.2,
        min=0.05,
        max=2.0,
        step=0.05,
    ),

    "initial_income_max": Slider(
        "Maximum initial income",
        value=1.0,
        min=0.1,
        max=3.0,
        step=0.05,
    ),

    # ---------------------------------------------------------
    # Affordability
    # ---------------------------------------------------------
    "affordability_share": Slider(
        "Maximum affordable rent share",
        value=0.8,
        min=0.1,
        max=1.5,
        step=0.05,
    ),

    # ---------------------------------------------------------
    # Theta_i: heterogeneous income-similarity preference
    # ---------------------------------------------------------
    "income_similarity_min": Slider(
        "Minimum income-similarity preference",
        value=0.0,
        min=0.0,
        max=3.0,
        step=0.1,
    ),

    "income_similarity_max": Slider(
        "Maximum income-similarity preference",
        value=2.0,
        min=0.0,
        max=3.0,
        step=0.1,
    ),

    # ---------------------------------------------------------
    # Beta_i: heterogeneous discount factor
    # ---------------------------------------------------------
    "discount_factor_min": Slider(
        "Minimum discount factor",
        value=0.0,
        min=0.0,
        max=1.0,
        step=0.05,
    ),

    "discount_factor_max": Slider(
        "Maximum discount factor",
        value=1.0,
        min=0.0,
        max=1.0,
        step=0.05,
    ),

    # ---------------------------------------------------------
    # Rho_i: heterogeneous risk aversion
    # ---------------------------------------------------------
    "risk_aversion_min": Slider(
        "Minimum risk aversion",
        value=0.0,
        min=0.0,
        max=5.0,
        step=0.05,
    ),

    "risk_aversion_max": Slider(
        "Maximum risk aversion",
        value=1.0,
        min=0.0,
        max=5.0,
        step=0.05,
    ),

    # ---------------------------------------------------------
    # Lambda_i: heterogeneous rationality
    #
    # Stored for the later logit model but currently inactive.
    # ---------------------------------------------------------
    "rationality_min": Slider(
        "Minimum rationality",
        value=1.0,
        min=0.0,
        max=20.0,
        step=0.5,
    ),

    "rationality_max": Slider(
        "Maximum rationality",
        value=10.0,
        min=0.0,
        max=20.0,
        step=0.5,
    ),

    # ---------------------------------------------------------
    # Multiplicative satisficing
    # ---------------------------------------------------------
    "satisficing_threshold": Slider(
        "Required proportional utility improvement",
        value=0.1,
        min=0.0,
        max=1.0,
        step=0.01,
    ),

    # This is not exposed as a slider because it is mainly a
    # numerical safeguard for current utility equal to zero.
    "minimum_absolute_improvement": 1e-6,

    # ---------------------------------------------------------
    # Fixed vision based on initial income
    #
    # v_i = round(g * y_i(0) + 1)
    # ---------------------------------------------------------
    "vision_income_scale": Slider(
        "Initial-income vision scaling",
        value=5.0,
        min=0.0,
        max=20.0,
        step=0.5,
    ),

    "maximum_vision_radius": Slider(
        "Maximum vision radius",
        value=10,
        min=1,
        max=20,
        step=1,
    ),

    # ---------------------------------------------------------
    # Duration-based satisfaction
    # ---------------------------------------------------------
    "steps_until_satisfied": Slider(
        "Stationary steps until satisfied",
        value=5,
        min=1,
        max=30,
        step=1,
    ),

    # ---------------------------------------------------------
    # Rent dynamics
    # ---------------------------------------------------------
    "rent_adjustment_rate": Slider(
        "Rent adjustment rate",
        value=0.01,
        min=0.0,
        max=0.1,
        step=0.001,
    ),

    # ---------------------------------------------------------
    # Income dynamics
    # ---------------------------------------------------------
    "income_growth_scaling": Slider(
        "Neighbourhood income-growth strength",
        value=0.001,
        min=0.0,
        max=0.1,
        step=0.001,
    ),

    "income_volatility": Slider(
        "Income volatility",
        value=0.05,
        min=0.0,
        max=0.3,
        step=0.01,
    ),

    # ---------------------------------------------------------
    # Game costs
    # ---------------------------------------------------------
    "moving_cost": Slider(
        "Moving cost c_m",
        value=0.05,
        min=0.0,
        max=1.0,
        step=0.01,
    ),

    "rejection_cost": Slider(
        "Rejection cost c_r",
        value=0.05,
        min=0.0,
        max=1.0,
        step=0.01,
    ),

    # ---------------------------------------------------------
    # Neighbourhood preferences and decision noise
    # ---------------------------------------------------------
    "neighborhood_risk_aversion": Slider(
        "Neighbourhood risk aversion",
        value=0.5,
        min=0.0,
        max=5.0,
        step=0.05,
    ),

    "neighborhood_rationality": Slider(
        "Neighbourhood rationality",
        value=5.0,
        min=0.0,
        max=20.0,
        step=0.5,
    ),

    # ---------------------------------------------------------
    # Numerical QRE settings
    # ---------------------------------------------------------
    "qre_tolerance": 1e-10,
    "qre_maximum_iterations": 1000,
    "qre_damping": 0.5,

    "keep_game_history": True,

}


model = GentrificationModel(
    width=20,
    height=20,
    density=0.8,

    neighborhood_radius=1,

    affordability_share=0.8,
    rent_adjustment_rate=0.1,

    initial_income_min=0.2,
    initial_income_max=1.0,

    discount_factor_min=0.0,
    discount_factor_max=1.0,

    risk_aversion_min=0.0,
    risk_aversion_max=2.0,

    rationality_min=1.0,
    rationality_max=10.0,

    income_similarity_min=0.0,
    income_similarity_max=2.0,

    satisficing_threshold=0.1,
    minimum_absolute_improvement=1e-6,

    vision_income_scale=5.0,
    maximum_vision_radius=10,

    steps_until_satisfied=5,

    income_growth_scaling=0.01,
    income_volatility=0.05,

    moving_cost=0.05,
    rejection_cost=0.05,

    neighborhood_risk_aversion=0.5,
    neighborhood_rationality=5.0,

    qre_tolerance=1e-10,
    qre_maximum_iterations=1000,
    qre_damping=0.5,

    keep_game_history=True,

    rng=42,
)


renderer = SpaceRenderer(
    model,
    backend="matplotlib",
)

renderer.setup_agents(agent_portrayal)
renderer.setup_propertylayer(property_layer_portrayal)
renderer.render()

@solara.component
def IncomeHistogram(model):
    update_counter.get()  # Required to trigger updates
    fig = Figure(figsize=(6, 4))
    ax = fig.subplots()
    income_vals = [agent.income for agent in model.agents]
    ax.hist(income_vals)
    ax.set_title("Income Histogram")
    solara.FigureMatplotlib(fig)

@solara.component
def RentVsIncomeScatter(model):
    update_counter.get()  # Required to trigger updates
    fig = Figure(figsize=(6, 4))
    ax = fig.subplots()

    rent_vals = []
    income_vals = []

    for agent in model.agents:
        rent_vals.append(agent.cell.rent)
        income_vals.append(agent.income)
    ax.scatter(income_vals, rent_vals[:len(income_vals)])

    max_income = max(income_vals)
    min_income = min(income_vals)
    max_affordability_rent = max_income * model.affordability_share
    min_affordability_rent = min_income * model.affordability_share
    ax.plot([min_income, max_income], [min_affordability_rent, max_affordability_rent], color="red", linestyle="--", label="Affordability threshold")

    ax.set_xlabel("Income")
    ax.set_ylabel("Rent")
    ax.set_title("Rent vs Income")
    ax.legend()
    solara.FigureMatplotlib(fig)

SatisfactionPlot = make_plot_component(
    {
        "pct_satisfied": "tab:green",
    }
)


MovementPlot = make_plot_component(
    {
        # "move_attempts": "tab:gray",
        "successful_moves": "tab:blue",
        "failed_searches": "tab:red",
    }
)


MovementSuccessPlot = make_plot_component(
    {
        "movement_success_rate": "tab:purple",
    }
)


IncomeRentPlot = make_plot_component(
    {
        "city_mean_income": "tab:blue",
        "mean_neighbor_income": "tab:orange",
        "mean_rent": "tab:red",
    }
)


UtilityPlot = make_plot_component(
    {
        "mean_utility": "tab:blue",
        "mean_value": "tab:green",
    }
)
GiniPlot = make_plot_component(
    {
        "gini_coefficient": "tab:blue",
    }
)

EquilibriumProbabilityPlot = (
    make_plot_component(
        {
            "mean_qre_move_probability": (
                "tab:blue"
            ),
            "mean_ne_move_probability": (
                "tab:orange"
            ),
        }
    )
)
EquilibriumDeviationPlot = (
    make_plot_component(
        {
            "mean_qre_ne_move_gap": (
                "tab:red"
            ),
        }
    )
)
NEFollowingPlot = make_plot_component(
    {
        "ne_following_rate": "tab:green",
    }
)

page = SolaraViz(
    model,
    renderer,
    components=[
        layer_selector,
        IncomeHistogram,
        RentVsIncomeScatter,
        get_model_statistics,
        SatisfactionPlot,
        MovementPlot,
        MovementSuccessPlot,
        IncomeRentPlot,
        UtilityPlot,
        GiniPlot,
        EquilibriumProbabilityPlot,
        EquilibriumDeviationPlot,
        NEFollowingPlot,
    ],
    model_params=model_params,
    name="Gentrification model",
)


page


