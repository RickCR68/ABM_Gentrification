import os
import mesa
import solara

from src.project.model import Schelling, SchellingScenario

from mesa.visualization import (
    Slider,
    SolaraViz,
    SpaceRenderer,
    make_plot_component,
)
from mesa.visualization.components import AgentPortrayalStyle


def get_happy_agents(model):
    """Display a text count of how many happy agents there are."""
    return solara.Markdown(f"**Happy agents: {model.happy}**")


path = os.path.dirname(os.path.abspath(__file__))


def agent_portrayal(agent):
    r = int((1 - agent.type) * 255)
    b = int(agent.type * 255)
    hex_color = f"#{r:02x}00{b:02x}"

    style = AgentPortrayalStyle(
        x=agent.cell.coordinate[0],
        y=agent.cell.coordinate[1],
        color=hex_color,
        marker='x',
        size=60,
    )

    if agent.happy:
        style.update(
            ("zorder", 3),
            ("marker", 'o'),
        )
    return style


model_params = {
    "rng": {
        "type": "InputText",
        "value": 67,
        "label": "Random Seed",
    },
    "density": Slider("Agent density", 0.4, 0.1, 1.0, 0.1),
    "minority_pc": Slider("Fraction minority", 0.5, 0.0, 1.0, 0.05),
    "alike_neighbors": Slider("Similar neighbors needed", 3, 0, 8, 1),
    "width": 25,
    "height": 25,
}

# Note: Models with images as markers are very performance intensive.
model1 = Schelling(scenario=SchellingScenario())
renderer = SpaceRenderer(model1, backend="matplotlib").setup_agents(agent_portrayal)
# Here we use renderer.render() to render the agents and grid in one go.
# This function always renders the grid and then renders the agents or
# property layers on top of it if specified.
renderer.render()

HappyPlot = make_plot_component({"happy": "tab:green"})

page = SolaraViz(
    model1,
    renderer,
    components=[
        HappyPlot,
        get_happy_agents,
    ],
    model_params=model_params,
)
page  # noqa
