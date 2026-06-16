"""DSL interpreter — execute a complete (hole-free) Program against an input.

The interpreter walks a PrimitiveNode tree and calls primitive.apply() on
each node, passing children's results upward. ConstNode values are passed
through directly. A HoleNode in the tree raises — the program must be
synthesized to completion before execution.
"""

from __future__ import annotations

from typing import Any

from .ast import Program, PrimitiveNode, HoleNode, ConstNode


class IncompleteProgramError(RuntimeError):
    """Raised when evaluate() encounters a HoleNode."""
    def __init__(self, hole_id: int, expected_type):
        super().__init__(
            f"cannot evaluate program: encountered hole #{hole_id} expecting "
            f"{expected_type.value}; synthesize the program before executing"
        )
        self.hole_id = hole_id


def evaluate(program: Program, *initial_inputs: Any) -> Any:
    """Execute a complete program against initial inputs.

    Args:
        program: a typed Program with all holes filled
        initial_inputs: the leaf-level inputs (usually a single grid)

    Returns:
        the program's output (type determined by program.output_type())

    Raises:
        IncompleteProgramError: if a HoleNode appears in the tree
    """
    return _eval_node(program.root, list(initial_inputs))


def _eval_node(node, leaf_inputs: list[Any]) -> Any:
    if isinstance(node, HoleNode):
        raise IncompleteProgramError(node.hole_id, node.expected_type)

    if isinstance(node, ConstNode):
        return node.value

    if isinstance(node, PrimitiveNode):
        # Evaluate children left-to-right; pass results to primitive.apply().
        child_values = []
        for child in node.children:
            if isinstance(child, HoleNode):
                # A hole at the leaf — bind to the next initial input.
                if leaf_inputs:
                    child_values.append(leaf_inputs.pop(0))
                else:
                    raise IncompleteProgramError(child.hole_id,
                                                 child.expected_type)
            elif isinstance(child, ConstNode):
                child_values.append(child.value)
            elif isinstance(child, PrimitiveNode):
                child_values.append(_eval_node(child, leaf_inputs))
            else:
                raise TypeError(
                    f"unknown node type in AST: {type(child).__name__}"
                )
        return node.primitive.apply(*child_values)

    raise TypeError(f"unknown root node type: {type(node).__name__}")
