from __future__ import annotations

import solara

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
    """Display current model statistics."""
    return solara.Markdown(
        f"""
### Current model state

**Population:** {len(model.agents)}  
**Satisfied households:** {model.satisfied_count}  
**Percentage satisfied:** {model.percentage_satisfied():.1f}%  
**Mean fixed vision radius:** {model.mean_vision():.2f}  

### Relocation

**Move attempts:** {model.move_attempts}  
**Successful moves:** {model.successful_moves}  
**Failed searches:** {model.failed_searches}  
**Movement success rate:** {model.movement_success_rate():.1%}  

### Economic state

**Mean household income:** {model.city_mean_income():.3f}  
**Mean neighbourhood income:** {model.mean_neighbor_income():.3f}  
**Mean rent:** {model.mean_rent():.3f}  
**Rent/income timescale ratio:** {model.rent_income_timescale_ratio():.3f}  

### Utility

**Mean raw utility:** {model.mean_utility():.3f}  
**Mean signed-CRRA value:** {model.mean_value():.3f}
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
        value=0.1,
        min=0.01,
        max=0.9,
        step=0.01,
    ),

    # ---------------------------------------------------------
    # Income dynamics
    # ---------------------------------------------------------
    "income_growth_scaling": Slider(
        "Neighbourhood income-growth strength",
        value=0.01,
        min=0.0,
        max=0.2,
        step=0.005,
    ),

    "income_volatility": Slider(
        "Income volatility",
        value=0.05,
        min=0.0,
        max=0.3,
        step=0.01,
    ),
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
    ax.set_xlabel("Income")
    ax.set_ylabel("Rent")
    ax.set_title("Rent vs Income")
    solara.FigureMatplotlib(fig)

SatisfactionPlot = make_plot_component(
    {
        "pct_satisfied": "tab:green",
    }
)


MovementPlot = make_plot_component(
    {
        "move_attempts": "tab:gray",
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

    ],
    model_params=model_params,
    name="Gentrification model",
)


page












# from __future__ import annotations
# import solara
# from mesa.visualization import SolaraViz, SpaceRenderer, Slider, make_plot_component
# from mesa.visualization.components import AgentPortrayalStyle

# from src.project.model import GentrificationModel
# from src.utils.helpers import generate_vibrant_red_blue_gradient

# def normalize_income(income: float) -> float:
#     income = max(float(income), 0.0)
#     return income / (1.0 + income)


# def agent_portrayal(agent):
#     """Define how a household is displayed."""
#     income_normalized = normalize_income(agent.income)

#     hex_color = generate_vibrant_red_blue_gradient(
#         income_normalized
#     )

#     return AgentPortrayalStyle(
#         color=hex_color,
#         marker="o" if agent.happy else "x",
#         size=80,
#         zorder=3 if agent.happy else 2,
#     )






# # def agent_portrayal(agent):
# #     return AgentPortrayalStyle(
# #         x=agent.cell.coordinate[0],
# #         y=agent.cell.coordinate[1],
# #         color="blue",
# #         size=100,
# #         zorder=100,
# #         marker="o",
# #     )



# def get_model_statistics(model):
#     """Display current movement, income, rent, and utility statistics."""
#     return solara.Markdown(
#         f"""
# ### Current model state

# **Population:** {len(model.agents)}  
# **Happy households:** {model.happy}  
# **Percentage happy:** {model.percentage_happy():.1f}%  

# ### Relocation

# **Move attempts:** {model.move_attempts}  
# **Successful moves:** {model.successful_moves}  
# **Failed searches:** {model.failed_searches}  
# **Movement success rate:** {model.movement_success_rate():.1%}  

# ### Economic state

# **Mean household income:** {model.city_mean_income():.3f}  
# **Mean rent:** {model.mean_rent():.3f}  

# ### Utility

# **Mean instantaneous utility:** {model.mean_current_utility():.3f}  
# **Mean expected utility:** {model.mean_expected_utility():.3f}  
# **Mean risk-adjusted utility:** {model.mean_risk_adjusted_utility():.3f}
# """
#     )


# model_params = {
#     "rng": {
#         "type": "InputText",
#         "value": 42,
#         "label": "Random seed",
#     },

#     # ---------------------------------------------------------
#     # Spatial parameters
#     # ---------------------------------------------------------
#     "width": 20,
#     "height": 20,

#     "density": Slider(
#         "Agent density",
#         value=0.8,
#         min=0.1,
#         max=0.95,
#         step=0.05,
#     ),

#     "neighborhood_radius": Slider(
#         "Neighbourhood radius",
#         value=1,
#         min=1,
#         max=4,
#         step=1,
#     ),

#     "search_radius": Slider(
#         "Household search radius",
#         value=4,
#         min=1,
#         max=10,
#         step=1,
#     ),

#     # ---------------------------------------------------------
#     # Utility and affordability
#     # ---------------------------------------------------------
#     "rent_weight": Slider(
#         "Rent utility weight",
#         value=-1.0,
#         min=-2.0,
#         max=1.0,
#         step=0.1,
#     ),

#     "affordability_share": Slider(
#         "Maximum rent share of income",
#         value=0.8,
#         min=0.1,
#         max=1.5,
#         step=0.1,
#     ),

#     "moving_cost": Slider(
#         "Moving cost",
#         value=0.05,
#         min=0.0,
#         max=0.5,
#         step=0.01,
#     ),

#     "capital_gain_weight": Slider(
#         "Expected neighbourhood-growth weight",
#         value=1.0,
#         min=0.0,
#         max=3.0,
#         step=0.1,
#     ),

#     # ---------------------------------------------------------
#     # Theta_i: income-similarity preference
#     # ---------------------------------------------------------
#     "income_similarity_min": Slider(
#         "Minimum income-similarity preference",
#         value=0.0,
#         min=0.0,
#         max=3.0,
#         step=0.1,
#     ),

#     "income_similarity_max": Slider(
#         "Maximum income-similarity preference",
#         value=2.0,
#         min=0.0,
#         max=3.0,
#         step=0.1,
#     ),

#     # ---------------------------------------------------------
#     # Current-location satisfaction
#     # ---------------------------------------------------------
#     "minimum_current_utility_min": Slider(
#         "Minimum current-utility threshold",
#         value=-1.0,
#         min=-5.0,
#         max=1.0,
#         step=0.1,
#     ),

#     "minimum_current_utility_max": Slider(
#         "Maximum current-utility threshold",
#         value=0.0,
#         min=-5.0,
#         max=1.0,
#         step=0.1,
#     ),

#     # ---------------------------------------------------------
#     # Satisficing destination threshold
#     # ---------------------------------------------------------
#     "satisficing_threshold_min": Slider(
#         "Minimum required utility improvement",
#         value=0.0,
#         min=0.0,
#         max=2.0,
#         step=0.05,
#     ),

#     "satisficing_threshold_max": Slider(
#         "Maximum required utility improvement",
#         value=0.2,
#         min=0.0,
#         max=2.0,
#         step=0.05,
#     ),

#     # ---------------------------------------------------------
#     # Rent dynamics
#     # ---------------------------------------------------------
#     "rent_adjustment_rate": Slider(
#         "Rent adjustment rate",
#         value=0.1,
#         min=0.01,
#         max=0.9,
#         step=0.01,
#     ),

#     # ---------------------------------------------------------
#     # Income dynamics
#     # ---------------------------------------------------------
#     "baseline_income_growth": Slider(
#         "Baseline income growth",
#         value=0.01,
#         min=-0.05,
#         max=0.1,
#         step=0.01,
#     ),

#     "opportunity_effect": Slider(
#         "Opportunity effect",
#         value=0.01,
#         min=0.0,
#         max=0.1,
#         step=0.01,
#     ),

#     "spillover_effect": Slider(
#         "Social spillover effect",
#         value=0.01,
#         min=0.0,
#         max=0.1,
#         step=0.01,
#     ),

#     "income_volatility": Slider(
#         "Income volatility",
#         value=0.05,
#         min=0.0,
#         max=0.3,
#         step=0.01,
#     ),
# }

# model = GentrificationModel(
#     width=20,
#     height=20,
#     density=0.8,

#     neighborhood_radius=1,
#     search_radius=4,

#     rent_weight=-1.0,
#     affordability_share=0.8,
#     moving_cost=0.05,
#     capital_gain_weight=1.0,
#     rent_adjustment_rate=0.1,

#     income_similarity_min=0.0,
#     income_similarity_max=2.0,

#     minimum_current_utility_min=-1.0,
#     minimum_current_utility_max=0.0,

#     satisficing_threshold_min=0.0,
#     satisficing_threshold_max=0.2,

#     baseline_income_growth=0.01,
#     opportunity_effect=0.01,
#     spillover_effect=0.01,
#     income_volatility=0.05,

#     rng=42,
# )


# renderer = SpaceRenderer(model)
# renderer.setup_agents(agent_portrayal)
# renderer.render()


# HappyPlot = make_plot_component(
#     {
#         "pct_happy": "tab:green",
#     }
# )


# MovementPlot = make_plot_component(
#     {
#         "move_attempts": "tab:gray",
#         "successful_moves": "tab:blue",
#         "failed_searches": "tab:red",
#     }
# )


# IncomeRentPlot = make_plot_component(
#     {
#         "city_mean_income": "tab:blue",
#         "mean_rent": "tab:red",
#     }
# )


# UtilityPlot = make_plot_component(
#     {
#         "mean_current_utility": "tab:blue",
#         "mean_expected_utility": "tab:orange",
#         "mean_risk_adjusted_utility": "tab:green",
#     }
# )


# MovementSuccessPlot = make_plot_component(
#     {
#         "movement_success_rate": "tab:purple",
#     }
# )



# page = SolaraViz(
#     model,
#     renderer,
#     components=[
#         get_model_statistics,
#         HappyPlot,
#         MovementPlot,
#         MovementSuccessPlot,
#         IncomeRentPlot,
#         UtilityPlot,
#     ],
#     model_params=model_params,
# )

# page