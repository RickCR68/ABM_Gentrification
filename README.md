# Agent Based Modelling - Group 2

## The Idea

Introduces more Decision-making steps into the regular Schelling [Agents](https://mesa.readthedocs.io/latest/examples/basic/schelling.html#agents) provided by [Mesa](https://mesa.readthedocs.io/latest/).

Before an agent's move, it tries to choose the best suited option within it's visible neighborhood, and upon choosing the best option, it applies for the Neighborhood, leading to a game being played where winning represents a reward when moving in (housewarming gift), and loosing, a 'fee' for not being accepted.

## Running:

### Environment Config

Configure the environment from the provided `pyproject.toml`. [`uv`](https://docs.astral.sh/uv/) is recommended.


```shell
uv sync
```

### Web View (Solara)

```shell
# from root project folder
solara run src/visualization/dashboard_framework.py
# if it doesn't work, try:
python -m solara run src/visualization/dashboard_framework.py
# if it still doesn't work, try:
PYTHONPATH=. python -m solara run src/visualization/dashboard_framework.py
```

### Profiling

Use the built-in profiler to compare runs before and after changes:

```shell
python scripts/profile_model.py --steps 25 --warmup-steps 5 --profile-output profiles/base.prof --text-output profiles/base.txt
```

The script prints a short run summary and a sorted cProfile table, and it also
writes a raw `.prof` file that you can compare across runs.

## Known Issues:

