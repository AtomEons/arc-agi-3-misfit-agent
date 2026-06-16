"""DSL combinators — higher-order program-shape primitives.

A combinator is a Primitive subclass whose role is to declare a
composition SHAPE for the synthesis engine (e.g. "two children, both Grid,
output Grid"). At evaluation time the children have already been reduced
to values by the interpreter; the combinator's apply() collapses those
values into the combinator's output.

Integration of teams SEQ, FOREACH_OBJECT, IFCOLOR, WHILECHANGING, MASKBY,
PARALLEL, REDUCE, IFSHAPE. Each module owns its construction-time type
validation and its own test file under tests/.
"""

from __future__ import annotations

from .seq import Seq
from .foreach_object import ForEachObject
from .if_color import IfColor
from .while_changing import WhileChanging
from .mask_by import MaskBy
from .parallel_combinator import Parallel
from .reduce import Reduce
from .if_shape import IfShape

__all__ = [
    "Seq",
    "ForEachObject",
    "IfColor",
    "WhileChanging",
    "MaskBy",
    "Parallel",
    "Reduce",
    "IfShape",
]
