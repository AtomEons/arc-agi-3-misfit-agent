"""Misfit-Alpha typed lambda-calculus DSL — Day 1 of the 100-day plan.

Architecture:
  types.py       — Type system (Grid, Color, Number, Object, ObjSet, Mask, Bool)
  primitives.py  — 12 atomic Spelke primitives with typed signatures
  combinators.py — 8 lambda combinators (Days 4-10)
  synthesis.py   — Beam search over typed program AST (Days 11-20)
  mdl.py         — MDL prior scoring (Days 21-28)
  refinement.py  — HRM-style outer refinement loop (Days 29-42)

Tier-1 honest by construction:
  - No LLM in synthesis or execution
  - No pretrained weights
  - No learned parameters (pure search over a hand-authored typed grammar)
  - The grammar IS the disclosure — its size, contents, and prior bias are
    fully visible to any reviewer
"""

from .types import (
    DslType, Grid, Color, Number, Object, ObjSet, Mask, Bool,
    type_signature, TypeMismatchError, Signature,
)
from .primitives import (
    Primitive,
    Identity, Translate, Rotate, Reflect, Recolor,
    Crop, Tile, Gravity, Symmetrize, KeepWhere,
    CountObj, ShapeOf,
    ALL_PRIMITIVES,
)

__all__ = [
    "DslType", "Grid", "Color", "Number", "Object", "ObjSet", "Mask", "Bool",
    "type_signature", "TypeMismatchError", "Signature",
    "Primitive",
    "Identity", "Translate", "Rotate", "Reflect", "Recolor",
    "Crop", "Tile", "Gravity", "Symmetrize", "KeepWhere",
    "CountObj", "ShapeOf",
    "ALL_PRIMITIVES",
]
