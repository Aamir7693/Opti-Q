"""Test that all imports work correctly."""

import sys
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))


def test_config_imports():
    """Test configuration module imports."""
    from src.llm_dag_optimizer.config import Settings, load_config, defaults
    assert Settings is not None
    assert load_config is not None


def test_structures_imports():
    """Test structures module imports."""
    from src.llm_dag_optimizer.structures import (
        Individual,
        LLMMetrics,
        Assignment,
        get_indegrees,
        mask_to_adjacency_matrix,
    )
    assert Individual is not None
    assert LLMMetrics is not None
    assert Assignment is not None


def test_core_imports():
    """Test core algorithm imports."""
    from src.llm_dag_optimizer.core import FPTAS
    assert FPTAS is not None


def test_package_imports():
    """Test main package imports."""
    from src.llm_dag_optimizer import (
        FPTAS,
        Settings,
        load_config,
        Individual,
        LLMMetrics,
    )
    assert FPTAS is not None
    assert Settings is not None


def test_config_loading():
    """Test configuration loading."""
    from src.llm_dag_optimizer.config import Settings

    # Test default config
    config = Settings()
    assert config.algorithm == "fptas"
    assert config.data.level == 2
    assert config.problem.max_nodes == 5

    # Test validation
    errors = config.validate()
    assert len(errors) == 0, f"Config validation failed: {errors}"


def test_metrics_dataclass():
    """Test metrics dataclass."""
    from src.llm_dag_optimizer.structures import LLMMetrics

    metrics = LLMMetrics(
        final_cost=0.001,
        final_latency=50.0,
        final_energy=100.0,
        quality_of_answer=0.85
    )

    assert metrics.final_cost == 0.001
    assert metrics.final_latency == 50.0
    assert metrics.final_energy == 100.0
    assert metrics.quality_of_answer == 0.85

    # Test to_tuple
    t = metrics.to_tuple()
    assert t == (0.001, 50.0, 100.0, 0.85)

    # Test from_tuple
    metrics2 = LLMMetrics.from_tuple(t)
    assert metrics2.final_cost == metrics.final_cost


if __name__ == "__main__":
    print("Running import tests...")

    test_config_imports()
    print("✓ Config imports")

    test_structures_imports()
    print("✓ Structure imports")

    test_core_imports()
    print("✓ Core imports")

    test_package_imports()
    print("✓ Package imports")

    test_config_loading()
    print("✓ Config loading")

    test_metrics_dataclass()
    print("✓ Metrics dataclass")

    print("\n✅ All import tests passed!")
