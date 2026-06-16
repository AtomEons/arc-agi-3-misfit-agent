"""Misfit-Alpha typed lambda-calculus DSL — Day 1 of the 100-day plan.

Architecture:
  types.py             — Type system (Grid, Color, Number, Object, ObjSet, Mask, Bool)
  primitives.py        — 12 atomic Spelke primitives with typed signatures
  combinators/         — Higher-order program-shape primitives
                         (Seq, ForEachObject, IfColor, WhileChanging, MaskBy,
                          Parallel, Reduce, IfShape)
  synthesis.py         — Beam search over typed program AST
  mdl.py               — MDL prior scoring
  refinement.py        — HRM-style outer refinement loop
  resonance_seed.py    — Resonance-library seeded synthesis init

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
from .ast import (
    Program, PrimitiveNode, HoleNode, ConstNode,
    make_program, make_hole,
)
from .interpreter import evaluate, IncompleteProgramError
from .walker import (
    walk_preorder, walk_postorder, find_holes,
    count_primitives, total_mdl_bits, visit,
)
from .combinators import (
    Seq, ForEachObject, IfColor, WhileChanging,
    MaskBy, Parallel, Reduce, IfShape,
)
from .synthesis import synthesize, MDL_LAMBDA
from .mdl import encoding_bits, train_cell_accuracy, score
from .refinement import refine, swap_primitive, wrap_program, mutate_param
from .resonance_seed import (
    seed_from_resonance, task_fingerprint, program_from_signature,
)

__all__ = [
    # types
    "DslType", "Grid", "Color", "Number", "Object", "ObjSet", "Mask", "Bool",
    "type_signature", "TypeMismatchError", "Signature",
    # primitives
    "Primitive",
    "Identity", "Translate", "Rotate", "Reflect", "Recolor",
    "Crop", "Tile", "Gravity", "Symmetrize", "KeepWhere",
    "CountObj", "ShapeOf",
    "ALL_PRIMITIVES",
    # ast
    "Program", "PrimitiveNode", "HoleNode", "ConstNode",
    "make_program", "make_hole",
    # interpreter
    "evaluate", "IncompleteProgramError",
    # walker
    "walk_preorder", "walk_postorder", "find_holes",
    "count_primitives", "total_mdl_bits", "visit",
    # combinators
    "Seq", "ForEachObject", "IfColor", "WhileChanging",
    "MaskBy", "Parallel", "Reduce", "IfShape",
    # synthesis
    "synthesize", "MDL_LAMBDA",
    # mdl
    "encoding_bits", "train_cell_accuracy", "score",
    # refinement
    "refine", "swap_primitive", "wrap_program", "mutate_param",
    # resonance_seed
    "seed_from_resonance", "task_fingerprint", "program_from_signature",
]
