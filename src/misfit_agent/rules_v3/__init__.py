"""Wave 4 rule families — climb from 2.30% toward higher coverage.

Six families, all Spelke priors only, no LLM, no learned parameters.
Wired into arc2_solver._rule_factories() in solver module update.
"""
from .wave4 import (
    KeepLargest, KeepSmallest, KeepByMaxColor, KeepByMinColor,
    DeleteLargest, DeleteSmallest,
    SymmetrizeH, SymmetrizeV, SymmetrizeDiag,
    GravityUp, GravityDown, GravityLeft, GravityRight,
    DrawBorder, FillInterior,
    ApplyUntilStable,
    ALL_WAVE4_RULES,
)

__all__ = [
    "KeepLargest", "KeepSmallest", "KeepByMaxColor", "KeepByMinColor",
    "DeleteLargest", "DeleteSmallest",
    "SymmetrizeH", "SymmetrizeV", "SymmetrizeDiag",
    "GravityUp", "GravityDown", "GravityLeft", "GravityRight",
    "DrawBorder", "FillInterior",
    "ApplyUntilStable",
    "ALL_WAVE4_RULES",
]
