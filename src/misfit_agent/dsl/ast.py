"""Typed Program AST — the data structure synthesis searches over.

A Program is a tree of Nodes. A Node is either:
  - PrimitiveNode  — a Primitive instance with bound parameters
  - HoleNode       — an unknown subprogram of a specific DslType (for synthesis)
  - ConstNode      — a literal value of a specific DslType (for scalar params)

Construction-time type checking:
  - When a Node is added as a child of another Node, the parent's expected
    input type at that slot is compared to the child's output type.
  - Mismatches raise TypeMismatchError at construction time, before the
    program ever runs.

This is the mechanism that makes the otherwise vast program space tractable:
the synthesis engine generates type-aware moves only.
"""

from __future__ import annotations

from dataclasses import dataclass, field
import hashlib

from .types import DslType, Signature, TypeMismatchError, type_signature
from .primitives import Primitive


@dataclass
class HoleNode:
    """An unknown subprogram of a specific DslType. Synthesis fills these in."""
    expected_type: DslType
    hole_id: int = 0

    @property
    def output_type(self) -> DslType:
        return self.expected_type

    def is_leaf(self) -> bool:
        return True

    def is_hole(self) -> bool:
        return True

    def is_complete(self) -> bool:
        return False

    def to_string(self) -> str:
        return f"<?:{self.expected_type.value}#{self.hole_id}>"

    def hash_key(self) -> str:
        return f"H[{self.expected_type.value}#{self.hole_id}]"


@dataclass
class ConstNode:
    """A literal value of a specific DslType."""
    value_type: DslType
    value: object

    @property
    def output_type(self) -> DslType:
        return self.value_type

    def is_leaf(self) -> bool:
        return True

    def is_hole(self) -> bool:
        return False

    def is_complete(self) -> bool:
        return True

    def to_string(self) -> str:
        return f"{self.value!r}:{self.value_type.value}"

    def hash_key(self) -> str:
        return f"C[{self.value_type.value}:{self.value!r}]"


@dataclass
class PrimitiveNode:
    """A Primitive instance with typed children for each declared input."""
    primitive: Primitive
    children: list = field(default_factory=list)

    def __post_init__(self):
        # Type-check at construction time.
        sig = type_signature(self.primitive)
        if len(self.children) != len(sig.inputs):
            raise TypeMismatchError(
                where=f"{type(self.primitive).__name__}",
                expected=DslType.GRID,  # placeholder — real error is below
                actual=DslType.GRID,
            )
        for i, (child, (name, expected)) in enumerate(zip(self.children, sig.inputs)):
            actual = child.output_type
            if actual != expected:
                raise TypeMismatchError(
                    where=f"{type(self.primitive).__name__}.{name} [child {i}]",
                    expected=expected,
                    actual=actual,
                )

    @property
    def output_type(self) -> DslType:
        return type_signature(self.primitive).output

    def is_leaf(self) -> bool:
        return all(c.is_leaf() and not c.is_hole() for c in self.children)

    def is_hole(self) -> bool:
        return False

    def is_complete(self) -> bool:
        return all(c.is_complete() for c in self.children)

    def to_string(self) -> str:
        if not self.children:
            return self.primitive.to_string()
        kids = ", ".join(c.to_string() for c in self.children)
        return f"{self.primitive.to_string()}({kids})"

    def hash_key(self) -> str:
        kids = ",".join(c.hash_key() for c in self.children)
        return f"P[{self.primitive.to_string()}]({kids})"


@dataclass
class Program:
    """A typed Program is a wrapper around a root Node with metadata.

    The Program records:
      - the root node (an AST)
      - the desired output type (so synthesis knows when complete)
      - the program's hash (for memoization)
      - the program's depth + node count (for MDL prior)
    """
    root: object  # PrimitiveNode | HoleNode | ConstNode
    desired_output: DslType = DslType.GRID

    def output_type(self) -> DslType:
        return self.root.output_type

    def is_complete(self) -> bool:
        """True if there are no remaining holes — program is executable."""
        return self.root.is_complete()

    def depth(self) -> int:
        return _depth(self.root)

    def node_count(self) -> int:
        return _count(self.root)

    def to_string(self) -> str:
        return self.root.to_string()

    def hash_key(self) -> str:
        return self.root.hash_key()

    def sha256_hash(self) -> str:
        """16-byte hash for memoization tables."""
        h = hashlib.sha256(self.hash_key().encode("utf-8")).hexdigest()
        return h[:32]

    def __repr__(self) -> str:
        return f"Program({self.to_string()}, depth={self.depth()})"


def _depth(node) -> int:
    if isinstance(node, PrimitiveNode):
        if not node.children:
            return 1
        return 1 + max(_depth(c) for c in node.children)
    return 0


def _count(node) -> int:
    if isinstance(node, PrimitiveNode):
        return 1 + sum(_count(c) for c in node.children)
    return 1


def make_hole(output_type: DslType, hole_id: int = 0) -> HoleNode:
    """Construct a hole — synthesis fills these in."""
    return HoleNode(expected_type=output_type, hole_id=hole_id)


def make_program(prim, *children) -> Program:
    """Construct a typed Program from a Primitive instance + child nodes.

    Example:
        from misfit_agent.dsl.primitives import Translate
        from misfit_agent.dsl.ast import make_program, make_hole, Grid
        p = make_program(Translate(dy=1, dx=0), make_hole(Grid))
    """
    node = PrimitiveNode(primitive=prim, children=list(children))
    return Program(root=node, desired_output=node.output_type)
