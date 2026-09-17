"""BFF language module.

This module defines the BFF instruction set as data: an opcode enumeration and a
256-entry table mapping every byte value to an opcode. The table is the parameter that
distinguishes BFF dialects sharing the same semantics, such as the ASCII encoding of
the reference implementation and the permuted encoding that packs the ten instructions
into the byte values 1 through 10.

References:
    [1] Computational Life: How Well-formed, Self-replicating Programs Emerge from
        Simple Interaction, Agüera y Arcas et al. 2024. https://arxiv.org/abs/2406.19108
    [2] cubff reference implementation. https://github.com/paradigms-of-intelligence/cubff

"""

from enum import IntEnum

import jax.numpy as jnp
from jax import Array


class Op(IntEnum):
    """BFF opcodes, in the order of the reference implementation's `BffOp` enum.

    The two data-dependent control-flow instructions come first, then the four
    tape-modifying instructions, then the four head moves. `NULL` is the zero byte, the
    only value that reads as "false" in a loop test; `NOOP` is every other byte with no
    instruction assigned. Both are skipped by the interpreter but still cost a step.
    """

    LOOP_START = 0  # [  if tape[head0] == 0: jump forward past the matching ]
    LOOP_END = 1  # ]  if tape[head0] != 0: jump backward past the matching [
    PLUS = 2  # +  tape[head0] += 1
    MINUS = 3  # -  tape[head0] -= 1
    COPY01 = 4  # .  tape[head1] = tape[head0]
    COPY10 = 5  # ,  tape[head0] = tape[head1]
    DEC0 = 6  # <  head0 -= 1
    INC0 = 7  # >  head0 += 1
    DEC1 = 8  # {  head1 -= 1
    INC1 = 9  # }  head1 += 1
    NULL = 10  # the zero byte
    NOOP = 11  # any other byte
    SWAP = 12  # ~  exchange head0 and head1 (single-copy dialect, not in the reference)


COMMAND_CHARS = "[]+-.,<>{}"
"""Instruction characters in opcode order, as the reference prints them."""

OP_CHARS = {**{Op(i): char for i, char in enumerate(COMMAND_CHARS)}, Op.SWAP: "~"}
"""Canonical character of every instruction opcode."""


def is_instruction(op: Array) -> Array:
    """Whether an opcode is an instruction rather than the zero byte or a no-op.

    Args:
        op: Integer array of `Op` values.

    Returns:
        Boolean array of the same shape.

    """
    return (op < Op.NULL) | (op == Op.SWAP)


def opcode_table_from_bytes(command_bytes: list[int]) -> Array:
    """Create the byte-to-opcode table from the byte values of the ten instructions.

    Args:
        command_bytes: Ten distinct byte values, one per instruction in the order of
            `COMMAND_CHARS`, i.e. `[ ] + - . , < > { }`. The zero byte is always
            `Op.NULL` and may not be used as an instruction; every other unlisted byte
            is `Op.NOOP`.

    Returns:
        Integer array of shape (256,) mapping each byte value to an `Op`.

    """
    if len(command_bytes) != 10:
        raise ValueError(f"Expected 10 command bytes, got {len(command_bytes)}")
    if len(set(command_bytes)) != 10:
        raise ValueError(f"Command bytes must be distinct, got {command_bytes!r}")
    for byte in command_bytes:
        if not 1 <= byte <= 255:
            raise ValueError(f"Command bytes must be in [1, 255], got {byte!r}")
    table = [int(Op.NOOP)] * 256
    table[0] = int(Op.NULL)
    for op, byte in enumerate(command_bytes):
        table[byte] = op
    return jnp.array(table, dtype=jnp.int32)


def opcode_table_from_string(commands: str = COMMAND_CHARS) -> Array:
    """Create the byte-to-opcode table from the ASCII characters of the instructions.

    The default, `"[]+-.,<>{}"`, is the encoding of the reference implementation's
    `bff` and `bff_noheads` languages, where instructions sit at their ASCII codes and
    ten of the 256 byte values are instructions.

    Args:
        commands: Ten distinct characters, one per instruction in the order of
            `COMMAND_CHARS`. Their code points are the instruction bytes.

    Returns:
        Integer array of shape (256,) mapping each byte value to an `Op`.

    """
    return opcode_table_from_bytes([ord(char) for char in commands])


def opcode_table_swap_heads(commands: str = "[]+-.<>~") -> Array:
    """Create the byte-to-opcode table of the single-copy dialect with a head swap.

    The dialect keeps `[ ] + - . < >` and replaces the three head1 instructions
    `{ } ,` by one, `~`, which exchanges the two heads; `,` is then `~.~` and a head1
    move is `~>~`. Eight instructions, so 8 of 256 byte values execute.

    Args:
        commands: Eight distinct characters for `[ ] + - . < > ~`, in that order.

    Returns:
        Integer array of shape (256,) mapping each byte value to an `Op`.

    """
    if len(commands) != 8 or len(set(commands)) != 8:
        raise ValueError(f"Expected 8 distinct command characters, got {commands!r}")
    ops = [
        Op.LOOP_START,
        Op.LOOP_END,
        Op.PLUS,
        Op.MINUS,
        Op.COPY01,
        Op.DEC0,
        Op.INC0,
        Op.SWAP,
    ]
    table = [int(Op.NOOP)] * 256
    table[0] = int(Op.NULL)
    for op, char in zip(ops, commands, strict=True):
        if ord(char) == 0:
            raise ValueError("The zero byte cannot be an instruction")
        table[ord(char)] = int(op)
    return jnp.array(table, dtype=jnp.int32)


def opcode_table_permuted() -> Array:
    """Create the packed byte-to-opcode table of the reference `bff_perm` language.

    The ten instructions occupy byte values 1 through 10 in the order
    `< > { } + - . , [ ]`, so the instructions are adjacent to the zero byte and a
    single increment or decrement turns one instruction into another.

    Returns:
        Integer array of shape (256,) mapping each byte value to an `Op`.

    """
    # bff_perm.cu: 1 -> DEC0, 2 -> INC0, 3 -> DEC1, 4 -> INC1, 5 -> PLUS, 6 -> MINUS,
    # 7 -> COPY01, 8 -> COPY10, 9 -> LOOP_START, 10 -> LOOP_END.
    by_op = {
        Op.LOOP_START: 9,
        Op.LOOP_END: 10,
        Op.PLUS: 5,
        Op.MINUS: 6,
        Op.COPY01: 7,
        Op.COPY10: 8,
        Op.DEC0: 1,
        Op.INC0: 2,
        Op.DEC1: 3,
        Op.INC1: 4,
    }
    return opcode_table_from_bytes([by_op[Op(op)] for op in range(10)])


def parse(program: str, *, opcode_table: Array | None = None) -> Array:
    """Parse a program written with the instruction characters into bytes.

    Each instruction character (`[ ] + - . , < > { } ~`) maps to the byte that
    `opcode_table` assigns to that instruction, the character `0` maps to the zero
    byte, and any other character maps to its own code point, which is a no-op unless
    the table says otherwise. This mirrors the reference parser for the printable
    subset it accepts.

    Args:
        program: Program text, e.g. `"[[{.>]-]]-]>.{[["`.
        opcode_table: Byte-to-opcode table; the default ASCII encoding if None.

    Returns:
        Unsigned 8-bit array of shape (len(program),).

    """
    if opcode_table is None:
        opcode_table = opcode_table_from_string()
    table = [int(value) for value in opcode_table]
    byte_of_op = {table[byte]: byte for byte in range(256) if table[byte] != Op.NOOP}
    op_of_char = {char: int(op) for op, char in OP_CHARS.items()}
    out = []
    for char in program:
        if char == "0":
            out.append(0)
        elif char in op_of_char and op_of_char[char] in byte_of_op:
            out.append(byte_of_op[op_of_char[char]])
        else:
            out.append(ord(char) % 256)
    return jnp.array(out, dtype=jnp.uint8)


def unparse(tape: Array, *, opcode_table: Array | None = None) -> str:
    """Render a tape as text: instruction characters, `0` for zero, others as is.

    Bytes that are neither instructions nor zero are rendered as a space when they are
    not printable ASCII, so a tape reads as its executable skeleton.

    Args:
        tape: Unsigned 8-bit array of shape (length,).
        opcode_table: Byte-to-opcode table; the default ASCII encoding if None.

    Returns:
        A string of `len(tape)` characters.

    """
    if opcode_table is None:
        opcode_table = opcode_table_from_string()
    table = [int(value) for value in opcode_table]
    chars = []
    for byte in [int(value) for value in tape]:
        op = Op(table[byte])
        if op in OP_CHARS:
            chars.append(OP_CHARS[op])
        elif op == Op.NULL:
            chars.append("0")
        elif 33 <= byte <= 126:
            chars.append(chr(byte))
        else:
            chars.append(" ")
    return "".join(chars)
