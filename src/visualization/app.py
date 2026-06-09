import os

import solara

from src.project_abm.model import Schelling, SchellingScenario

from mesa.visualization import (
    Slider,
    SolaraViz,
    SpaceRenderer,
    make_plot_component,
)
from mesa.visualization.components import AgentPortrayalStyle

from project_abm.model import GentrificationModel


def get_model_statistics(model):
    """Display current movement and happiness statistics."""
    acceptance_rate = (
        model.successful_moves / model.move_attempts
        if model.move_attempts > 0
        else 0.0
    )

    return solara.Markdown(
        f"""
        **Happy agents:** {model.happy}  
        **Move attempts:** {model.move_attempts}  
        **Accepted moves:** {model.successful_moves}  
        **Rejected moves:** {model.rejected_moves}  
        **Acceptance rate:** {acceptance_rate:.2%}
        """
    )


path = os.path.dirname(os.path.abspath(__file__))


def agent_portrayal(agent):
    """Define how an agent is displayed."""
    style = AgentPortrayalStyle(
        x=agent.cell.coordinate[0],
        y=agent.cell.coordinate[1],
        zorder=2,
        marker=os.path.join(path, "resources", "dog-solid.png"),
        size=80,
    )

    if agent.type == 0:
        marker = (
            "cat-solid.png"
            if agent.happy
            else "cat-solid-sad.png"
        )
    else:
        marker = (
            "dog-solid.png"
            if agent.happy
            else "dog-solid-sad.png"
        )

    style.update(
        (
            "marker",
            os.path.join(path, "resources", marker),
        )
    )

    return style


model_params = {
    "rng": {
        "type": "InputText",
        "value": 42,
        "label": "Random seed",
    },
    "density": Slider(
        "Agent density",
        value=0.8,
        min=0.1,
        max=0.95,
        step=0.05,
    ),
    "minority_pc": Slider(
        "Fraction minority",
        value=0.2,
        min=0.0,
        max=1.0,
        step=0.05,
    ),
    "homophily_min": Slider(
    "Minimum homophily",
    value=0.2,
    min=0.0,
    max=1.0,
    step=0.05,
    ),
    "homophily_max": Slider(
        "Maximum homophily",
        value=0.6,
        min=0.0,
        max=1.0,
        step=0.05,
    ),
    "neighborhood_radius": Slider(
        "Neighbourhood radius",
        value=1,
        min=1,
        max=4,
        step=1,
    ),
    "acceptance_midpoint": Slider(
        "Acceptance similarity midpoint",
        value=0.5,
        min=0.0,
        max=1.0,
        step=0.05,
    ),
    "acceptance_steepness": Slider(
        "Acceptance steepness",
        value=10.0,
        min=0.5,
        max=30.0,
        step=0.5,
    ),
    "empty_similarity": Slider(
        "Empty-area similarity",
        value=0.5,
        min=0.0,
        max=1.0,
        step=0.05,
    ),
    "width": 20,
    "height": 20,
}


model = GentrificationModel(
    width=20,
    height=20,
    density=0.8,
    minority_pc=0.2,
    homophily_min=0.2,
    homophily_max=0.6,
    neighborhood_radius=1,
    acceptance_midpoint=0.5,
    acceptance_steepness=10.0,
    empty_similarity=0.5,
    rng=42,
)

renderer = (
    SpaceRenderer(model, backend="matplotlib")
    .setup_agents(agent_portrayal)
)

renderer.render()

HappyPlot = make_plot_component(
    {
        "pct_happy": "tab:green",
    }
)

MovementPlot = make_plot_component(
    {
        "successful_moves": "tab:blue",
        "rejected_moves": "tab:red",
    }
)

page = SolaraViz(
    model,
    renderer,
    components=[
        HappyPlot,
        MovementPlot,
        get_model_statistics,
    ],
    model_params=model_params,
)

page