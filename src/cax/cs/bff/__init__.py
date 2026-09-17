"""BFF module.

References:
    [1] Computational Life: How Well-formed, Self-replicating Programs Emerge from
        Simple Interaction, Agüera y Arcas et al. 2024. https://arxiv.org/abs/2406.19108
    [2] BFF: Simple explanations for complex phenomena, Knierim et al. 2026.
        https://arxiv.org/abs/2607.01483

"""

from .assay import assays, mutational_scan, run_pairs
from .cs import BFF, local_pairing, skeleton_hash
from .detector import (
    CHAIN_LENGTH,
    NUM_CHAINS,
    replication_score,
    sample_partners,
    score_from_tapes,
)
from .grid import BFFGrid, GridState
from .interpreter import (
    Control,
    ThreadState,
    initial_thread_state,
    match_bracket,
    match_bracket_cyclic,
    run,
    step,
)
from .language import (
    COMMAND_CHARS,
    OP_CHARS,
    Op,
    is_instruction,
    opcode_table_from_bytes,
    opcode_table_from_string,
    opcode_table_permuted,
    opcode_table_swap_heads,
    parse,
    unparse,
)
from .metrics import byte_entropy, compressed_bits_per_byte, high_order_entropy

__all__ = [
    "BFF",
    "CHAIN_LENGTH",
    "COMMAND_CHARS",
    "NUM_CHAINS",
    "OP_CHARS",
    "BFFGrid",
    "Control",
    "GridState",
    "Op",
    "ThreadState",
    "assays",
    "byte_entropy",
    "compressed_bits_per_byte",
    "high_order_entropy",
    "initial_thread_state",
    "is_instruction",
    "local_pairing",
    "match_bracket",
    "match_bracket_cyclic",
    "mutational_scan",
    "opcode_table_from_bytes",
    "opcode_table_from_string",
    "opcode_table_permuted",
    "opcode_table_swap_heads",
    "parse",
    "replication_score",
    "run",
    "run_pairs",
    "sample_partners",
    "score_from_tapes",
    "skeleton_hash",
    "step",
    "unparse",
]
