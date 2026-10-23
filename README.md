# graph-opt-playground

Worked example for [graph-opt](https://github.com/khbalhandawi/graph-opt): the
synthetic toy benchmark from the paper on airline alliance design. It covers the
benchmark definition, the solver runs and the saved solutions. The IATA schedule
data used for the paper's main results are proprietary and are not included.

The benchmark partitions 6 airlines into alliances on a random route network of
20 airports. The objective trades off route-level competition (an HHI term)
against market penetration capacity (a log-MPC term) with weights 0.7 : 0.3.
With 6 airlines there are only Bell(6) = 203 partitions, so brute-force
enumeration gives the true optimum. That optimum is the reference for the
sampled greedy heuristic and the MIQP.

## Contents

| Path | What it is |
|---|---|
| `generate_toy_graph.py` | Builds the benchmark graph (seed 1234). It draws 2000 random (origin, destination, airline) records, drops the 102 where origin = destination, and keeps 1898 records on 375 of the 380 possible directed segments. |
| `data_toy/test_airline_nodes_E2000_V20_T6.{graphml,json}` | The generated graph and airline list, as used in the paper. |
| `solve_toy_benchmark.py` | Runs the brute-force oracle, then 10 sampled realizations each of the greedy heuristic and the MIQP (Gurobi). |
| `models_opt/E2000_V20_T6/low_samples_weighted/` | Saved outputs of `solve_toy_benchmark.py`: `brute_force/`, `greedy_optimizer_{0..9}/` and `exact_optimizer_{0..9}/`. |
| `toy_problems.ipynb` | Loads the saved runs and draws the paper's toy-benchmark figures into `images/optim_results_toy/low_samples_weighted/`. |

## Setup

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
```

The MIQP step needs a Gurobi licence (`gurobipy`). The notebook's graph-layout
figure also uses
[graph-community-detection](https://github.com/khbalhandawi/graph-community-detection)
(`show_graph`, `get_layout`). The other cells only need graph-opt.

## Reproduce

```bash
python generate_toy_graph.py      # rewrites data_toy/ (identical output with the fixed seed)
python solve_toy_benchmark.py     # rewrites models_opt/E2000_V20_T6/low_samples_weighted/
jupyter notebook toy_problems.ipynb
```

To make the figures from the saved solutions without a Gurobi licence, skip the
second step.

## Settings used in the paper

| Setting | Value |
|---|---|
| Samples per segment, `n_samples_edge` | 50 |
| Random walks per airport, `n_walks` | 10 |
| Walk length, `L` | 2 |
| Weights (β, γ) | (0.7, 0.3) |
| Sampled realizations | 10 |
| MIQP | `MIPGap=1e-3`, Gurobi's general log constraint (adaptive breakpoints, absolute error ≤ 1e-6) |
