"""
Configuration Module

Contains all global configuration parameters and constants.
"""

import json

# =============================================================================
# Algorithm Parameters
# =============================================================================

# Default values
REPETITIONS = 1
RESULTS_FILE = "nsga_results.csv"
POP_SIZE = 200
GENERATIONS = 200
MAX_NODES = 5

# Genetic operator probabilities
ADD_NODE_PROB = 0.1
FLIP_EDGE_PROB = 0.3
MODEL_MUTATION_PROB = 0.2

# Query types to optimize
QUERY_TYPES = ['Art']
# Full list (commented out for quick testing):
# QUERY_TYPES = [
#     'Art', 'Geography', 'History', 'Music', 'Other', 'Politics',
#     'Science and technology', 'Sports', 'TV shows', 'Video games',
#     'biology_mmlu', 'business_mmlu', 'chemistry_mmlu', 'computer science_mmlu',
#     'economics_mmlu', 'engineering_mmlu', 'health_mmlu', 'history_mmlu',
#     'law_mmlu', 'math_mmlu', 'other_mmlu', 'philosophy_mmlu',
#     'physics_mmlu', 'psychology_mmlu'
# ]

# =============================================================================
# Evaluation Parameters
# =============================================================================

# Token counts for evaluation
QUERY_TOKENS = 215
BLENDING_PROMPT_TOKENS = 26
CTX_TOKENS = 39

# Historical metrics file
FILEPATH_TO_HISTORY_FILE = "updated_final_metrics_l4_with_tok.csv"

# Evaluation defaults
DEFAULT_QOA = 0.5
EPS = 1e-9

# =============================================================================
# Fuzzy Matching Parameters
# =============================================================================

FUZZY_THRESHOLD = 0.70

# =============================================================================
# Optimization Flags
# =============================================================================

NSGA_DEBUG = True
DISABLE_TQDM = True
DAG_DEBUG_PRINT = False


# =============================================================================
# Configuration Loading
# =============================================================================

def load_config_from_file(config_path: str):
    """
    Load configuration from JSON file.

    Args:
        config_path: Path to JSON configuration file

    Returns:
        None (updates global variables)
    """
    global REPETITIONS, RESULTS_FILE, POP_SIZE, GENERATIONS, MAX_NODES
    global ADD_NODE_PROB, FLIP_EDGE_PROB, MODEL_MUTATION_PROB, QUERY_TYPES

    try:
        with open(config_path) as f:
            cfg = json.load(f)

        # Update globals from config
        REPETITIONS = cfg.get("repetitions", REPETITIONS)
        RESULTS_FILE = cfg.get("results_file_name", RESULTS_FILE)
        POP_SIZE = cfg.get("pop_size", POP_SIZE)
        GENERATIONS = cfg.get("generations", GENERATIONS)
        MAX_NODES = cfg.get("max_nodes", MAX_NODES)
        ADD_NODE_PROB = cfg.get("prob_add_node", ADD_NODE_PROB)
        FLIP_EDGE_PROB = cfg.get("prob_flip_edge", FLIP_EDGE_PROB)
        MODEL_MUTATION_PROB = cfg.get("prob_model_mutation", MODEL_MUTATION_PROB)
        QUERY_TYPES = list(cfg.get("query_types", QUERY_TYPES))

        print(f"Configuration loaded from {config_path}")

    except FileNotFoundError:
        print(f"Error: Config file '{config_path}' not found.")
    except json.JSONDecodeError as e:
        print(f"Error: Invalid JSON in config file '{config_path}': {e}")
    except Exception as e:
        print(f"Error loading config file '{config_path}': {e}")


def print_config():
    """Print current configuration."""
    print("=" * 80)
    print("CURRENT CONFIGURATION")
    print("=" * 80)
    print(f"REPETITIONS:           {REPETITIONS}")
    print(f"RESULTS_FILE:          {RESULTS_FILE}")
    print(f"POP_SIZE:              {POP_SIZE}")
    print(f"GENERATIONS:           {GENERATIONS}")
    print(f"MAX_NODES:             {MAX_NODES}")
    print(f"ADD_NODE_PROB:         {ADD_NODE_PROB}")
    print(f"FLIP_EDGE_PROB:        {FLIP_EDGE_PROB}")
    print(f"MODEL_MUTATION_PROB:   {MODEL_MUTATION_PROB}")
    print(f"QUERY_TYPES:           {QUERY_TYPES}")
    print(f"HISTORY_FILE:          {FILEPATH_TO_HISTORY_FILE}")
    print("=" * 80)
