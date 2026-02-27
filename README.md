# Opti-Q: A Constraint-Based Optimization Framework for Multi-LLM Question Planning

Multi-objective optimization of LLM execution plans using FPTAS and MOQO (NSGA-II) algorithms.

[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

## Overview

This repository implements three algorithms for optimizing LLM execution plans represented as Directed Acyclic Graphs (DAGs):

1. **FPTAS (DP)**: Fully Polynomial-Time Approximation Scheme using dynamic programming
2. **MOQO (Hill Climbing)**: Multi-Objective Query Optimization using hill climbing with Pareto dominance
3. **NSGA-II**: Multi-objective evolutionary algorithm using non-dominated sorting

The optimizer finds Pareto-optimal trade-offs between:
- **Cost**: Dollar cost per query
- **Latency**: Response time (milliseconds)
- **Energy**: Energy consumption (millijoules)
- **QoA**: Quality of Answer (0-1)

## Key Features

✅ **Modular Architecture**: Clean separation of concerns
✅ **Configuration System**: 38+ toggles via YAML files
✅ **Reproducible**: Configurable random seeds and deterministic mode
✅ **Well-Tested**: Unit tests and integration tests included
✅ **Documented**: Comprehensive documentation and examples
✅ **Professional Structure**: Ready for paper submission

## Quick Start

### Installation

```bash
# Clone the repository
git clone https://github.com/yourusername/llm-dag-optimizer.git
cd llm-dag-optimizer

# Install dependencies
pip install -r requirements.txt

# Or install in development mode
pip install -e .
```

### Run FPTAS

```bash
# Using default configuration
python -m experiments.run_fptas

# Using custom configuration
python -m experiments.run_fptas --config config/experiments/fptas_level2_no_pruning.yaml
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

**38+ Configuration Toggles Available:**
- Algorithm selection (FPTAS/MOQO/NSGA-II)
- Data level (0-4)
- Pruning on/off (FPTAS)
- Timeout (MOQO)
- Population size, generations (NSGA-II)
- Fuzzy matching on/off
- Evaluation caching
- Output formats
- Debug options
- And more...

See [docs/configuration.md](docs/configuration.md) for complete reference.

## Algorithms

### 1. FPTAS (Dynamic Programming)

- **Type**: Exact approximation scheme
- **Approach**: Bottom-up dynamic programming with ε-dominance pruning
- **Guarantees**: (1+ε)-approximation of Pareto frontier
- **Best for**: Accuracy, complete enumeration

**Key Parameters:**
- `epsilon`: Approximation factor (e.g., 0.05 = 5%)
- `pruning.enabled`: Toggle ε-dominance pruning
- `max_nodes`: Maximum nodes in DAG

### 2. MOQO (Hill Climbing)

- **Type**: Hill climbing with Pareto dominance
- **Approach**: Random restarts + adaptive precision climb
- **Components**: ParetoClimb, ApproximateFrontiers
- **Best for**: Fast results, time-bounded optimization

**Key Parameters:**
- `timeout_seconds`: Time budget (e.g., 60s)
- `max_nodes`: Maximum nodes in DAG
- `alpha`: Precision parameter

### 3. NSGA-II (Evolutionary Algorithm)

- **Type**: Population-based multi-objective evolutionary
- **Approach**: Non-dominated sorting + crowding distance
- **Operators**: Crossover, mutation (add node, flip edge, change model)
- **Best for**: Diverse solutions, exploration

**Key Parameters:**
- `population_size`: Size of population (e.g., 200)
- `generations`: Number of generations (e.g., 200)
- `crossover_rate`: Crossover probability
- `mutation.*`: Mutation probabilities

**Comparison:**
| Algorithm | Speed | Quality | Diversity | Best Use Case |
|-----------|-------|---------|-----------|---------------|
| FPTAS (DP) | Slow | (1+ε)-optimal | High | Accuracy |
| MOQO (Hill) | Fast | Good | Low | Speed |
| NSGA-II | Medium | Good | High | Balance |

See [docs/algorithms/ALGORITHMS_OVERVIEW.md](docs/algorithms/ALGORITHMS_OVERVIEW.md) for detailed comparison.

## Example Experiments

### Experiment 1: Compare Pruning Impact

```bash
# Without pruning
python -m experiments.run_fptas --config config/experiments/fptas_level2_no_pruning.yaml

# With pruning
python -m experiments.run_fptas --config config/experiments/fptas_level2_with_pruning.yaml
```

### Experiment 2: Multi-Query Batch Processing

```yaml
data:
  query_types:
    - "Art"
    - "Science and technology"
    - "History"
```

### Experiment 3: High Precision

```yaml
fptas:
  epsilon: 0.01  # 1% approximation (slower but more accurate)
```

## Results

Results are saved in configured output directory:

```
results/
├── fptas_results_20260129_011103.csv
└── fptas_results_20260129_011103.json
```

### Sample Output

```csv
query_type,struct_id,assignment,cost,latency,energy,qoa
Art,661,"[4, 3, 0, 5, 0]",0.000097,80.36,134.38,0.7993
Art,667,"[4, 0, 5, 0, 5]",0.000124,82.15,140.22,0.7709
```

## Testing

```bash
# Run import tests
python tests/test_imports.py

# Run integration tests
python tests/test_integration.py

# Or use pytest (if installed)
pytest tests/
```

## Documentation

- [Quick Start Guide](docs/QUICK_START.md)
- [Configuration Guide](docs/configuration.md)
- [Algorithm Documentation](docs/algorithms/)
- [API Documentation](docs/api/)

## Project History

This repository was restructured on **January 29, 2026** for improved:
- Modularity and maintainability
- Configuration management
- Testing and validation
- Documentation
- Paper submission readiness

**Original backup**: `NSGA-II-backup-20260129-005834.zip` (4.3 MB)

## Citation

If you use this code in your research, please cite:

```bibtex
@article{yourpaper2026,
  title={Multi-Objective Optimization of LLM Execution Plans},
  author={Your Name},
  journal={Conference/Journal Name},
  year={2026}
}
```

## License

MIT License - see [LICENSE](LICENSE) file for details

## Contributing

Contributions welcome! Please:
1. Fork the repository
2. Create a feature branch
3. Add tests for new features
4. Submit a pull request

## Contact

- **Author**: Your Name
- **Email**: your.email@example.com
- **GitHub**: https://github.com/yourusername/llm-dag-optimizer

## Acknowledgments

- FPTAS algorithm inspired by classical approximation schemes
- MOQO based on NSGA-II evolutionary framework
- Historical performance data collection methodology

---

**Version**: 1.0.0
**Last Updated**: January 29, 2026
**Status**: Production-ready for paper submission
running---
python experiments/run_dp.py --queries Art --max-k 2 --delta 0.05 --only-delta --level 2
New test-set files (all scripts)

  Each script now produces a second CSV after the main solutions CSV:
  ┌─────────────┬────────────────────────────┬───────────────────────────┐
  │   Script    │       Solutions CSV        │       Test-set CSV        │
  ├─────────────┼────────────────────────────┼───────────────────────────┤
  │ run_nsga.py │ results/nsga_solutions.csv │ results/nsga_test_set.csv │
  ├─────────────┼────────────────────────────┼───────────────────────────┤
  │ run_dp.py   │ results/dp_solutions.csv   │ results/dp_test_set.csv   │
  ├─────────────┼────────────────────────────┼───────────────────────────┤
  │ run_moqo.py │ results/moqo_solutions.csv │ results/moqo_test_set.csv │
  └─────────────┴────────────────────────────┴───────────────────────────┘
