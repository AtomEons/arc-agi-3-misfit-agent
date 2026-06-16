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
from .wave5 import (
    KeepByPredicate, DeleteByPredicate,
    RecolorByPredicate, ForEachObjectGravity,
    ALL_WAVE5_RULES,
)
from .wave6 import (
    ColorMap, ColorReplace,
    CropToContent, CropToColor, KeepOnlyColor,
    ALL_WAVE6_RULES,
)
from .wave7 import (
    RecolorByCountRank, SwapTwoNonBgColors,
    RecolorByAreaRank, CropToObjectByAreaRank,
    PaintObjectByRankWithColorOfRank,
    ALL_WAVE7_RULES,
)
from .wave8 import (
    CropToObjectByColorRank, DeleteAllExceptRankN,
    RecolorAllObjectsToColorOfRank, MirrorAcrossDominantAxis,
    CompleteFrameOf,
    ALL_WAVE8_RULES,
)

__all__ = [
    "KeepLargest", "KeepSmallest", "KeepByMaxColor", "KeepByMinColor",
    "DeleteLargest", "DeleteSmallest",
    "SymmetrizeH", "SymmetrizeV", "SymmetrizeDiag",
    "GravityUp", "GravityDown", "GravityLeft", "GravityRight",
    "DrawBorder", "FillInterior",
    "ApplyUntilStable",
    "ALL_WAVE4_RULES",
    "KeepByPredicate", "DeleteByPredicate",
    "RecolorByPredicate", "ForEachObjectGravity",
    "ALL_WAVE5_RULES",
    "ColorMap", "ColorReplace",
    "CropToContent", "CropToColor", "KeepOnlyColor",
    "ALL_WAVE6_RULES",
    "RecolorByCountRank", "SwapTwoNonBgColors",
    "RecolorByAreaRank", "CropToObjectByAreaRank",
    "PaintObjectByRankWithColorOfRank",
    "ALL_WAVE7_RULES",
    "CropToObjectByColorRank", "DeleteAllExceptRankN",
    "RecolorAllObjectsToColorOfRank", "MirrorAcrossDominantAxis",
    "CompleteFrameOf",
    "ALL_WAVE8_RULES",
]
