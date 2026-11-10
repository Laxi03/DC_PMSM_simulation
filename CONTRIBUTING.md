# Contributing

Use Python 3.12, install with `python -m pip install -e '.[rl]'`, and run
`python -m unittest discover -s tests -v`. The classical baseline tests also
run without the learning extra.

When changing the simulator, state which equations, defaults, reference
profiles, or sampling assumptions change. When changing training, record the
seed, interaction budget, and task. Do not commit outputs, checkpoint binaries,
vendor datasheets, or machine-specific settings. Only load trusted checkpoints.
