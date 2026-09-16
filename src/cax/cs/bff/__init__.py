"""BFF module.

References:
    [1] Computational Life: How Well-formed, Self-replicating Programs Emerge from
        Simple Interaction, Agüera y Arcas et al. 2024. https://arxiv.org/abs/2406.19108
    [2] BFF: Simple explanations for complex phenomena, Knierim et al. 2026.
        https://arxiv.org/abs/2607.01483

"""

from .cs import BFF
from .detector import (
    CHAIN_LENGTH,
    NUM_CHAINS,
    replication_score,
    sample_partners,
    score_from_tapes,
)
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
    "Control",
    "Op",
    "ThreadState",
    "byte_entropy",
    "compressed_bits_per_byte",
    "high_order_entropy",
    "initial_thread_state",
    "is_instruction",
    "match_bracket",
    "match_bracket_cyclic",
    "opcode_table_from_bytes",
    "opcode_table_from_string",
    "opcode_table_permuted",
    "opcode_table_swap_heads",
    "parse",
    "replication_score",
    "run",
    "sample_partners",
    "score_from_tapes",
    "step",
    "unparse",
]
