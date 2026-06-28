# Agent Based Modelling - Group 2

## The Idea

Makes [Schelling](https://mesa.readthedocs.io/stable/examples/basic/schelling.html) [Agents](https://mesa.readthedocs.io/latest/examples/basic/schelling.html#agents) have floating point values for their types (income in our case), and introduces more Decision-making steps for movement and satisfaction Built on top of the model and tools provided by [Mesa](https://mesa.readthedocs.io/latest/).

Before an agent's move, it tries to choose the best suited option within it's visible neighborhood, and upon choosing the best option, it applies for the Neighborhood, leading to a game being played where winning represents a reward when moving in (housewarming gift), and loosing, a 'fee' for not being well integrated.

## Running:

### Environment Config

Configure the environment from the provided `pyproject.toml`. [`uv`](https://docs.astral.sh/uv/) is recommended.


```shell
uv sync
```

If you'd prefer not using `uv`, make sure you have Python 3.14 installed, and a virtual environment created and active, then use `pip` to install:

```shell
# For the Model
mesa[rec,viz] # >= 3,

# For Analysis
pandas # >= 2.3.3,
salib # >= 1.5.2,
seaborn # >= 0.13.2,
```

### Web View (Solara)

```shell
# from root project folder
solara run src/visualization/dashboard_framework.py
# if it doesn't work, try:
python -m solara run src/visualization/dashboard_framework.py
# if it still doesn't work, try:
PYTHONPATH=. python -m solara run src/visualization/dashboard_framework.py

# Visualization options include:
# - .../visualization/app.py
# - .../visualization/dashboard_framework.py
# - .../visualization/dashboard_framework_v2.py
```

### Profiling

Use the profiling script to compare runs or understand time spent on each step of the process:

```shell
python scripts/profile_model.py --steps 25 --warmup-steps 5 --profile-output profiles/base.prof --text-output profiles/base.txt
```

The script prints a short run summary and a sorted cProfile table. It can also
write a raw `.prof` file.

## Known Issues:

- In some Tabs/Pages, reseting changes the grid size to 11x11;
    - This doesn't seem to affect `app.py`;
    - Reloading the page also seems to work, but your mileage may vary;
- Execution times heavily depend on hardware being used;
    - For a smoother experience, we used 100 steps at a time and the largest interval time (500ms usually);
    - clicking 'Step' instead of 'Play' also helps with analysis
- Solara is known to crash after long runs or some parameter chenges mid run, when that happens reloading the page is usually the best option;
