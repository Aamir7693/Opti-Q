# Opti-Q: A Constraint-Based Optimization Framework for Multi-LLM Question Planning

Multi-objective optimization of LLM execution plans using DP, Hill Climbing and NSGA-II.

[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

## Overview
<img width="800" height="600" alt="Arch" src="https://github.com/user-attachments/assets/9e71f873-7033-430b-ac7c-8e1c22d74e62" />

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
numpy>=1.20.0
pandas>=1.3.0
networkx>=2.6.0
pyyaml>=5.4.0
matplotlib>=3.4.0
tqdm>=4.62.0
psutil>=5.9.0
pygmo>=2.19.0
python-Levenshtein>=0.21.0
```

### Run DP

```bash
# Using default configuration
python -m experiments.run_dp

# Using custom configuration
python -m experiments.run_dp --config config/experiments/fptas_level2_no_pruning.yaml
```

### Run NSGA-II

```bash
# Using default configuration
python -m experiments.run_nsga

# Using custom configuration
python -m experiments.run_nsga --config config/experiments/nsga2_baseline.yaml
```
### Run Hill Climbing

```bash
# Using default configuration
python -m experiments.run_moqo

# Using custom configuration
python -m experiments.run_moqo --config config/experiments/moqo_baseline.yaml
```












