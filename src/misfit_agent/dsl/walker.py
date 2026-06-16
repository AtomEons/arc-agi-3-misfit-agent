"""Generic AST walker — used by synthesis, MDL bit counting, and analysis.

Walking a Program tree is a fundamental operation. The synthesis engine
walks to enumerate holes. The MDL scorer walks to sum per-node bit costs.
The verifier walks to certify type correctness post-hoc. All share this
single traversal API.
"""

from __future__ import annotations

from typing import Callable, Iterator

from .ast import Program, PrimitiveNode, HoleNode, ConstNode


def walk_preorder(program: Program) -> Iterator:
    """Yield every node in pre-order traversal."""
    yield from _walk_node_preorder(program.root)


def _walk_node_preorder(node):
    yield node
    if isinstance(node, PrimitiveNode):
        for child in node.children:
            yield from _walk_node_preorder(child)


def walk_postorder(program: Program) -> Iterator:
    """Yield every node in post-order traversal (leaves first)."""
    yield from _walk_node_postorder(program.root)


def _walk_node_postorder(node):
    if isinstance(node, PrimitiveNode):
        for child in node.children:
            yield from _walk_node_postorder(child)
    yield node


def find_holes(program: Program) -> list:
    """Return every HoleNode in the program AST.

    Synthesis uses this to enumerate where to insert new subprograms.
    """
    return [n for n in walk_preorder(program) if isinstance(n, HoleNode)]


def count_primitives(program: Program) -> int:
    """Number of PrimitiveNode instances in the program."""
    return sum(1 for n in walk_preorder(program) if isinstance(n, PrimitiveNode))


def total_mdl_bits(program: Program) -> float:
    """Sum the MDL bits of every primitive in the program.

    A program that uses more primitives, or primitives with more parameters,
    costs more bits. Synthesis prefers shorter programs (MDL prior).
    """
    return sum(
        n.primitive.mdl_bits()
        for n in walk_preorder(program)
        if isinstance(n, PrimitiveNode)
    )


def visit(program: Program, fn: Callable, order: str = "preorder") -> list:
    """Walk the program and apply fn to every node; return collected results.

    Args:
        program: a Program
        fn: a callable taking a single node, returning anything
        order: "preorder" or "postorder"
    """
    walker = walk_preorder if order == "preorder" else walk_postorder
    return [fn(n) for n in walker(program)]
