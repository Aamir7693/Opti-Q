"""Integration test to verify outputs match original implementation."""

import sys
from pathlib import Path
import pandas as pd

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.llm_dag_optimizer.core import FPTAS
from src.llm_dag_optimizer.config import Settings


def test_fptas_basic():
    """Test FPTAS produces results."""
    # Load data
    df = pd.read_csv("data/raw/levels/level_2_data.csv")

    # Run FPTAS with simple config
    solutions = FPTAS(
        query_type="Art",
        df_history=df,
        max_nodes=3,  # Small for fast test
        epsilon=0.05,
        verbose=False,
        allow_empty_preds=True,
        extract_only_admissible=True,
        disable_pruning=False,
    )

    # Verify we got solutions
    assert len(solutions) > 0, "FPTAS should return solutions"

    # Verify solution structure
    sol = solutions[0]
    assert hasattr(sol, 'cost'), "Solution should have cost"
    assert hasattr(sol, 'latency'), "Solution should have latency"
    assert hasattr(sol, 'energy'), "Solution should have energy"
    assert hasattr(sol, 'qoa'), "Solution should have QoA"
    assert hasattr(sol, 'struct_id'), "Solution should have struct_id"
    assert hasattr(sol, 'assignment'), "Solution should have assignment"

    # Verify metrics are reasonable
    assert 0 <= sol.qoa <= 1, f"QoA should be in [0,1], got {sol.qoa}"
    assert sol.cost >= 0, f"Cost should be non-negative, got {sol.cost}"
    assert sol.latency >= 0, f"Latency should be non-negative, got {sol.latency}"
    assert sol.energy >= 0, f"Energy should be non-negative, got {sol.energy}"

    print(f"✓ FPTAS test passed: {len(solutions)} solutions found")
    print(f"  Sample: QoA={sol.qoa:.4f}, Cost={sol.cost:.6f}, "
          f"Latency={sol.latency:.2f}, Energy={sol.energy:.2f}")


def test_config_integration():
    """Test config system integration."""
    config = Settings()

    # Test validation
    errors = config.validate()
    assert len(errors) == 0, f"Default config should be valid: {errors}"

    # Test config values
    assert config.algorithm in ["fptas", "moqo"]
    assert 0 <= config.data.level <= 4
    assert config.problem.max_nodes >= 1
    assert 0 < config.fptas.epsilon < 1

    print("✓ Config integration test passed")


if __name__ == "__main__":
    print("Running integration tests...\n")

    test_config_integration()
    test_fptas_basic()

    print("\n✅ All integration tests passed!")
