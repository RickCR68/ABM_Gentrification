from src.project.model import GentrificationModel
from src.project.agents import SchellingAgent as GentrificationAgent # Assuming your agent class name
from mesa.examples.basic.schelling.model import Schelling, SchellingScenario

class SchellingGridExperiment(GentrificationModel):
    """A model that clears default agents and respawns them using a Schelling grid layout."""

    def __init__(
        self,
        *,
        neighborhood_radius: int = 1,
        **kwargs,
    ) -> None:
        # 1. Let the parent run its default setup (which creates its own default agents)
        super().__init__(
            neighborhood_radius=neighborhood_radius,
            **kwargs,
        )
        self.neighborhood_radius = neighborhood_radius

        # 2. CLEAR ALL DEFAULT AGENTS (Empty the SingleGrid and Model tracking)
        # We take a copy of the agents list to avoid modifying it while iterating
        self.remove_all_agents()

        # 3. Run the vanilla Schelling model to get your stable layout map
        shell_scenario = SchellingScenario(
            width=self.width,
            height=self.height,
            density=self.density,
            radius=self.neighborhood_radius,
        )
        schelling_model = Schelling(scenario=shell_scenario)
        schelling_model.run_model()

        # 4. Populate your model with fresh GentrificationAgents matching the layout
        for old_agent in schelling_model.agents:
            stable_cell = old_agent.cell.position
            stable_pos = (stable_cell[0], stable_cell[1])
            agent_income = old_agent.type + 0.1 # e.g., 0 or 1 representing groups

            # Create your actual custom GentrificationAgent
            # (Pass whatever arguments your custom agent __init__ expects)
            new_agent = GentrificationAgent(
                model=self,
                cell=self.grid._cells[stable_pos],
                income=agent_income,
                income_similarity_preference=(
                    self.random.uniform(
                        self.income_similarity_min,
                        self.income_similarity_max,
                    )
                ),
                discount_factor=self.random.uniform(
                    self.discount_factor_min,
                    self.discount_factor_max,
                ),
                risk_aversion=self.random.uniform(
                    self.risk_aversion_min,
                    self.risk_aversion_max,
                ),
                rationality=self.random.uniform(
                    self.rationality_min,
                    self.rationality_max,
                ),
            )

            # Safely place it on the completely wiped grid
            self.grid._cells[stable_pos].agents[0] = new_agent
