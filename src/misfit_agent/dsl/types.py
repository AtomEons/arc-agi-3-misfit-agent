"""DSL type system — types are first-class values that compose programs.

Every primitive and combinator has a typed signature. The synthesis engine
type-checks composition at program-construction time so invalid programs are
rejected before they hit the beam search. This is the mechanism that prunes
the otherwise-vast program space.

Type catalog:
  Grid     — 2D numpy array of int 0..9 (the ARC grid)
  Color    — int 0..9 (single ARC color)
  Number   — int ≥ 0 (count, offset, etc.)
  Object   — perceived object record (bbox + centroid + color)
  ObjSet   — list[Object] (output of perceiver)
  Mask     — 2D bool array (per-cell selector)
  Bool     — bool (branch predicate result)

Type checking is *structural* — a primitive declares (input_types, output_type)
and the type checker walks the AST verifying every edge.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Any


class DslType(str, Enum):
    """Atomic DSL types. Compositions are tuples of these."""
    GRID = "Grid"
    COLOR = "Color"
    NUMBER = "Number"
    OBJECT = "Object"
    OBJSET = "ObjSet"
    MASK = "Mask"
    BOOL = "Bool"


# Convenience aliases — make code reading match the type catalog.
Grid = DslType.GRID
Color = DslType.COLOR
Number = DslType.NUMBER
Object = DslType.OBJECT
ObjSet = DslType.OBJSET
Mask = DslType.MASK
Bool = DslType.BOOL


class TypeMismatchError(TypeError):
    """Raised when a program edge connects types that don't match.

    The error message names the primitive, expected type, and actual type so
    debugging the synthesis engine doesn't require staring at AST dumps.
    """
    def __init__(self, where: str, expected: DslType, actual: DslType):
        super().__init__(
            f"DSL type mismatch at {where}: expected {expected.value}, "
            f"got {actual.value}"
        )
        self.where = where
        self.expected = expected
        self.actual = actual


@dataclass(frozen=True)
class Signature:
    """A primitive's typed signature.

    Attributes:
      inputs: tuple of (param_name, DslType) for each input
      output: DslType of the primitive's output
      params: tuple of (param_name, python_type) for non-input parameters
              (these are scalar args like dx, dy, k, axis — not type-checked
              against the grid flow, just validated as the right Python type)
    """
    inputs: tuple[tuple[str, DslType], ...]
    output: DslType
    params: tuple[tuple[str, type], ...] = ()

    def __repr__(self) -> str:
        ins = ", ".join(f"{n}:{t.value}" for n, t in self.inputs)
        ps = ", ".join(f"{n}:{t.__name__}" for n, t in self.params)
        return f"({ins}) [{ps}] → {self.output.value}"


def type_signature(prim: Any) -> Signature:
    """Return a primitive instance's typed signature."""
    if hasattr(prim, "signature_typed") and callable(prim.signature_typed):
        sig = prim.signature_typed()
        if not isinstance(sig, Signature):
            raise TypeError(
                f"{type(prim).__name__}.signature_typed() must return a "
                f"Signature, got {type(sig).__name__}"
            )
        return sig
    raise TypeError(
        f"{type(prim).__name__} does not implement signature_typed() — "
        f"every Primitive subclass must declare its typed signature"
    )
