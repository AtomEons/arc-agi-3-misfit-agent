"""Typed rule templates over Spelke object priors.

Per docs/TIER_1_DISCLOSURE.md: this rule set is a HAND-AUTHORED grammar
informed by exposure to ARC-AGI-1 and ARC-AGI-2 examples. It is NOT
derived purely from Spelke priors. The grammar is disclosed honestly.

Each rule template has the interface:
    .fit(observations) -> bool      # True if rule consistent with observations
    .predict(state, action) -> state # forward simulation under the rule

A rule has at most 3 free parameters. The composer fits per-object-class
rules from observed (state, action, next_state) tuples by exhaustive
template enumeration with consistency checking against ALL observations.
"""

from __future__ import annotations

from .translate import Translate
from .no_op import NoOp

__all__ = ["Translate", "NoOp"]
