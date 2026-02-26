"""Configuration settings management for LLM DAG Optimizer."""

import os
import yaml
from dataclasses import dataclass, field
from typing import List, Optional, Dict, Any
from pathlib import Path

from .defaults import *


@dataclass
class DataConfig:
    """Data-related configuration."""
    level: int = DEFAULT_LEVEL
    path: Optional[str] = None
    query_types: List[str] = field(default_factory=lambda: ["Art"])

    def __post_init__(self):
        """Set default path if not provided."""
        if self.path is None:
            self.path = f"data/raw/levels/level_{self.level}_data.csv"


@dataclass
class ProblemConfig:
    """Problem definition configuration."""
    max_nodes: int = DEFAULT_MAX_NODES
    models: List[int] = field(default_factory=lambda: DEFAULT_MODELS.copy())
    query_tokens: int = DEFAULT_QUERY_TOKENS
    blending_prompt_tokens: int = DEFAULT_BLENDING_PROMPT_TOKENS
    context_tokens: int = DEFAULT_CONTEXT_TOKENS


@dataclass
class FPTASPruningConfig:
    """FPTAS pruning configuration."""
    enabled: bool = DEFAULT_PRUNING_ENABLED
    method: str = "epsilon_dominance"


@dataclass
class FPTASConstraintsConfig:
    """FPTAS constraint configuration."""
    allow_empty_predecessors: bool = DEFAULT_ALLOW_EMPTY_PREDS
    extract_only_admissible: bool = DEFAULT_EXTRACT_ONLY_ADMISSIBLE


@dataclass
class FPTASConfig:
    """FPTAS algorithm configuration."""
    epsilon: float = DEFAULT_EPSILON
    pruning: FPTASPruningConfig = field(default_factory=FPTASPruningConfig)
    constraints: FPTASConstraintsConfig = field(default_factory=FPTASConstraintsConfig)


@dataclass
class MOQOMutationConfig:
    """MOQO mutation configuration."""
    add_node_prob: float = DEFAULT_ADD_NODE_PROB
    flip_edge_prob: float = DEFAULT_FLIP_EDGE_PROB
    model_mutation_prob: float = DEFAULT_MODEL_MUTATION_PROB


@dataclass
class MOQOOperatorsConfig:
    """MOQO genetic operators configuration."""
    crossover_rate: float = DEFAULT_CROSSOVER_RATE
    mutation: MOQOMutationConfig = field(default_factory=MOQOMutationConfig)


@dataclass
class MOQODeduplicationConfig:
    """MOQO deduplication configuration."""
    enabled: bool = DEFAULT_GENOTYPE_DEDUP


@dataclass
class MOQOConfig:
    """MOQO (NSGA-II) algorithm configuration."""
    population_size: int = DEFAULT_POPULATION_SIZE
    generations: int = DEFAULT_GENERATIONS
    operators: MOQOOperatorsConfig = field(default_factory=MOQOOperatorsConfig)
    deduplication: MOQODeduplicationConfig = field(default_factory=MOQODeduplicationConfig)


@dataclass
class EvaluationMatchingConfig:
    """Evaluation matching configuration."""
    fuzzy_enabled: bool = DEFAULT_FUZZY_ENABLED
    exact_enabled: bool = DEFAULT_EXACT_ENABLED
    fuzzy_threshold: float = DEFAULT_FUZZY_THRESHOLD


@dataclass
class EvaluationQoAConfig:
    """QoA estimation configuration."""
    default_value: float = DEFAULT_QOA_VALUE
    estimation_method: str = DEFAULT_QOA_METHOD


@dataclass
class EvaluationCachingConfig:
    """Evaluation caching configuration."""
    enabled: bool = DEFAULT_EVAL_CACHING


@dataclass
class EvaluationConfig:
    """Evaluation configuration."""
    matching: EvaluationMatchingConfig = field(default_factory=EvaluationMatchingConfig)
    qoa: EvaluationQoAConfig = field(default_factory=EvaluationQoAConfig)
    caching: EvaluationCachingConfig = field(default_factory=EvaluationCachingConfig)


@dataclass
class OptimizationConfig:
    """Optimization features configuration."""
    canonicalization: bool = DEFAULT_CANONICALIZATION
    history_indexing: bool = DEFAULT_HISTORY_INDEXING
    parallel: bool = DEFAULT_PARALLEL_ENABLED
    workers: int = DEFAULT_PARALLEL_WORKERS


@dataclass
class ValidationConfig:
    """Validation configuration."""
    blend_constraints: bool = DEFAULT_BLEND_CONSTRAINTS
    dag_validity: bool = DEFAULT_DAG_VALIDITY
    strict_mode: bool = DEFAULT_STRICT_MODE


@dataclass
class OutputFormatsConfig:
    """Output formats configuration."""
    csv: bool = DEFAULT_CSV_OUTPUT
    json: bool = DEFAULT_JSON_OUTPUT
    pickle: bool = DEFAULT_PICKLE_OUTPUT


@dataclass
class OutputVisualizationConfig:
    """Visualization configuration."""
    enabled: bool = DEFAULT_VIZ_ENABLED
    format: str = DEFAULT_VIZ_FORMAT


@dataclass
class OutputConfig:
    """Output configuration."""
    directory: str = DEFAULT_OUTPUT_DIR
    formats: OutputFormatsConfig = field(default_factory=OutputFormatsConfig)
    visualization: OutputVisualizationConfig = field(default_factory=OutputVisualizationConfig)
    save_intermediate: bool = DEFAULT_SAVE_INTERMEDIATE


@dataclass
class DebugConfig:
    """Debug configuration."""
    verbose: bool = DEFAULT_VERBOSE
    log_level: str = DEFAULT_LOG_LEVEL
    profiling: bool = DEFAULT_PROFILING
    trace_evaluation: bool = DEFAULT_TRACE_EVAL


@dataclass
class Settings:
    """Main configuration settings class."""
    algorithm: str = DEFAULT_ALGORITHM
    data: DataConfig = field(default_factory=DataConfig)
    problem: ProblemConfig = field(default_factory=ProblemConfig)
    fptas: FPTASConfig = field(default_factory=FPTASConfig)
    moqo: MOQOConfig = field(default_factory=MOQOConfig)
    evaluation: EvaluationConfig = field(default_factory=EvaluationConfig)
    optimization: OptimizationConfig = field(default_factory=OptimizationConfig)
    validation: ValidationConfig = field(default_factory=ValidationConfig)
    output: OutputConfig = field(default_factory=OutputConfig)
    debug: DebugConfig = field(default_factory=DebugConfig)
    random_seed: Optional[int] = DEFAULT_RANDOM_SEED
    deterministic: bool = DEFAULT_DETERMINISTIC

    @classmethod
    def from_yaml(cls, yaml_path: str) -> 'Settings':
        """Load settings from YAML file.

        Args:
            yaml_path: Path to YAML configuration file

        Returns:
            Settings object
        """
        with open(yaml_path, 'r') as f:
            config_dict = yaml.safe_load(f)

        return cls.from_dict(config_dict)

    @classmethod
    def from_dict(cls, config_dict: Dict[str, Any]) -> 'Settings':
        """Create settings from dictionary.

        Args:
            config_dict: Configuration dictionary

        Returns:
            Settings object
        """
        # Recursively convert nested dicts to config objects
        def convert_nested(data, config_class):
            if data is None:
                return config_class()
            if isinstance(data, dict):
                kwargs = {}
                for field_name, field_type in config_class.__annotations__.items():
                    if field_name in data:
                        value = data[field_name]
                        # Check if field type is also a dataclass
                        if hasattr(field_type, '__annotations__'):
                            kwargs[field_name] = convert_nested(value, field_type)
                        else:
                            kwargs[field_name] = value
                return config_class(**kwargs)
            return data

        return convert_nested(config_dict, cls)

    def to_dict(self) -> Dict[str, Any]:
        """Convert settings to dictionary.

        Returns:
            Configuration dictionary
        """
        def convert_to_dict(obj):
            if hasattr(obj, '__dataclass_fields__'):
                return {
                    field: convert_to_dict(getattr(obj, field))
                    for field in obj.__dataclass_fields__
                }
            elif isinstance(obj, list):
                return [convert_to_dict(item) for item in obj]
            return obj

        return convert_to_dict(self)

    def to_yaml(self, output_path: str):
        """Save settings to YAML file.

        Args:
            output_path: Path to save YAML file
        """
        with open(output_path, 'w') as f:
            yaml.dump(self.to_dict(), f, default_flow_style=False, sort_keys=False)

    def validate(self) -> List[str]:
        """Validate configuration settings.

        Returns:
            List of validation error messages (empty if valid)
        """
        errors = []

        # Validate algorithm
        valid_algorithms = ["fptas", "moqo", "nsga2", "nsga-ii", "nsga"]
        if self.algorithm not in valid_algorithms:
            errors.append(f"Invalid algorithm: {self.algorithm}. Must be one of {valid_algorithms}")

        # Validate data level
        if not 0 <= self.data.level <= 4:
            errors.append(f"Invalid data level: {self.data.level}. Must be 0-4")

        # Validate max_nodes
        if self.problem.max_nodes < 1:
            errors.append(f"Invalid max_nodes: {self.problem.max_nodes}. Must be >= 1")

        # Validate epsilon
        if not 0 < self.fptas.epsilon < 1:
            errors.append(f"Invalid epsilon: {self.fptas.epsilon}. Must be between 0 and 1")

        # Validate MOQO parameters
        if self.moqo.population_size < 1:
            errors.append(f"Invalid population_size: {self.moqo.population_size}. Must be >= 1")

        if self.moqo.generations < 1:
            errors.append(f"Invalid generations: {self.moqo.generations}. Must be >= 1")

        # Validate probabilities
        probs = [
            ("crossover_rate", self.moqo.operators.crossover_rate),
            ("add_node_prob", self.moqo.operators.mutation.add_node_prob),
            ("flip_edge_prob", self.moqo.operators.mutation.flip_edge_prob),
            ("model_mutation_prob", self.moqo.operators.mutation.model_mutation_prob),
            ("fuzzy_threshold", self.evaluation.matching.fuzzy_threshold),
        ]

        for name, value in probs:
            if not 0 <= value <= 1:
                errors.append(f"Invalid {name}: {value}. Must be between 0 and 1")

        # Validate log level
        valid_log_levels = ["DEBUG", "INFO", "WARNING", "ERROR"]
        if self.debug.log_level not in valid_log_levels:
            errors.append(f"Invalid log_level: {self.debug.log_level}. "
                         f"Must be one of {valid_log_levels}")

        return errors


def load_config(config_path: Optional[str] = None) -> Settings:
    """Load configuration from file or use defaults.

    Args:
        config_path: Optional path to YAML config file

    Returns:
        Settings object
    """
    if config_path is None or not os.path.exists(config_path):
        return Settings()

    return Settings.from_yaml(config_path)
