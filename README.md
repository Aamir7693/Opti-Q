# Opti-Q: A Constraint-Based Optimization Framework for Multi-LLM Question Planning

Multi-objective optimization of LLM execution plans using DP, Hill Climbing and NSGA-II.

[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

## Overview
<img width="1500" height="1200" alt="Arch" src="https://github.com/user-attachments/assets/9e71f873-7033-430b-ac7c-8e1c22d74e62" />

This repository implements three algorithms for optimizing LLM execution plans represented as Directed Acyclic Graphs (DAGs):

1. **DP**: Fully Polynomial-Time Approximation Scheme using dynamic programming
2. **Hill Climbing**: Multi-Objective Query Optimization using hill climbing with Pareto dominance
3. **NSGA-II**: Multi-objective evolutionary algorithm using non-dominated sorting

The optimizer finds Pareto-optimal trade-offs between:
- **Cost**: Dollar cost per query
- **Latency**: Response time (milliseconds)
- **Energy**: Energy consumption (millijoules)
- **QoA**: Quality of Answer (0-1)

## Quick Start

### Installation

```bash
# Clone the repository
git clone https://github.com/Aamir7693/Opti-Q.git
cd Opti-Q

# Install dependencies
pip install -r requirements.txt

# Or install in development mode
pip install -e .
```

### Run DP

```bash
# Using default configuration
python -m experiments.run_dp

# Using custom configuration
python -m experiments.run_dp --config config/experiments/fptas_level2_no_pruning.yaml
```

### Python API

```python
import pandas as pd
from src.llm_dag_optimizer.core import FPTAS

# Load historical data
df = pd.read_csv("data/raw/levels/level_2_data.csv")

# Run FPTAS
solutions = FPTAS(
    query_type="Art",
    df_history=df,
    max_nodes=5,
    epsilon=0.05,
    verbose=True,
    disable_pruning=False,
)

# Process results
for sol in solutions[:5]:
    print(f"QoA={sol.qoa:.4f}, Cost={sol.cost:.6f}, "
          f"Latency={sol.latency:.2f}, Energy={sol.energy:.2f}")
```

## Repository Structure

```
llm-dag-optimizer/
├── config/                     # Configuration files
│   ├── default.yaml           # Default configuration
│   └── experiments/           # Pre-configured experiments
│
├── data/                       # Data files
│   └── raw/levels/            # Historical performance data
│
├── src/llm_dag_optimizer/     # Source code
│   ├── core/                  # Core algorithms (FPTAS, MOQO)
│   ├── structures/            # Data structures (Individual, DAG, Metrics)
│   ├── config/                # Configuration management
│   └── ...
│
├── experiments/               # Experiment scripts
│   └── run_fptas.py          # FPTAS runner
│
├── tests/                     # Unit and integration tests
│   ├── test_imports.py
│   └── test_integration.py
│
├── docs/                      # Documentation
│   ├── QUICK_START.md
│   ├── configuration.md
│   └── algorithms/
│
└── results/                   # Experiment outputs
```

## Configuration

All experiments are configured via YAML files. Example:

```yaml
algorithm: "fptas"

data:
  level: 2
  query_types: ["Art"]

problem:
  max_nodes: 5

fptas:
  epsilon: 0.05
  pruning:
    enabled: true

output:
  directory: "results"
  formats:
    csv: true
    json: true

random_seed: 42
```
## Requirements









