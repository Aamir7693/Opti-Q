"""Test that all three algorithms are accessible and work."""

import sys
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))


def test_algorithm_imports():
    """Test that all three algorithms can be imported."""
    from src.llm_dag_optimizer.core import FPTAS, RandomMOQO, run_nsga2

    assert FPTAS is not None, "FPTAS should be importable"
    assert RandomMOQO is not None, "RandomMOQO should be importable"
    assert run_nsga2 is not None, "run_nsga2 should be importable"

    print("✓ All three algorithms can be imported")


def test_evaluation_imports():
    """Test that evaluation module is accessible."""
    from src.llm_dag_optimizer.evaluation import (
        evaluate_dag_plan,
        get_single_model_metrics,
        get_pairwise_metrics,
    )

    assert evaluate_dag_plan is not None
    assert get_single_model_metrics is not None
    assert get_pairwise_metrics is not None

    print("✓ Evaluation module accessible")


def test_operators_imports():
    """Test that operators are accessible."""
    from src.llm_dag_optimizer.operators import (
        mutate_individual,
        crossover_individuals,
        tournament_selection,
    )

    assert mutate_individual is not None
    assert crossover_individuals is not None
    assert tournament_selection is not None

    print("✓ Operators module accessible")


def test_config_with_nsga2():
    """Test configuration system with NSGA-II."""
    from src.llm_dag_optimizer.config import Settings

    config = Settings()
    config.algorithm = "nsga2"
    config.moqo.population_size = 100
    config.moqo.generations = 50

    errors = config.validate()
    assert len(errors) == 0, f"Config should be valid: {errors}"

    print("✓ Configuration works with NSGA-II")


def test_fptas_still_works():
    """Quick test that FPTAS still works after refactoring."""
    import pandas as pd
    from src.llm_dag_optimizer.core import FPTAS

    df = pd.read_csv("data/raw/levels/level_2_data.csv")

    solutions = FPTAS(
        query_type="Art",
        df_history=df,
        max_nodes=3,
        epsilon=0.05,
        verbose=False,
        disable_pruning=False,
    )

    assert len(solutions) > 0, "FPTAS should return solutions"
    print(f"✓ FPTAS works: {len(solutions)} solutions")


if __name__ == "__main__":
    print("Running comprehensive algorithm tests...\n")

    test_algorithm_imports()
    test_evaluation_imports()
    test_operators_imports()
    test_config_with_nsga2()
    test_fptas_still_works()

    print("\n✅ All algorithm tests passed!")
    print("\n📋 Summary:")
    print("  ✓ FPTAS: Modular & working")
    print("  ✓ MOQO: Modular & accessible")
    print("  ✓ NSGA-II: Modular & accessible")
    print("  ✓ Evaluation: Modular")
    print("  ✓ Operators: Organized")
