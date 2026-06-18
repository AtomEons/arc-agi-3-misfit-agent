"""misfit_agent — Tier-1 Spelke-priors substrate for ARC-AGI-3.

The Misfit Agent class lives in `misfit_agent.misfit_agent` and requires
the arcengine package (provided in the Kaggle eval environment).
For substrate testing without arcengine, import the priors-only modules
directly: perceptor, episode, fingerprint, resonance, action_search.
"""

__all__ = ["Misfit"]


def __getattr__(name: str):  # noqa: D401
    """Lazy import of Misfit so substrate tests don't require arcengine."""
    if name == "Misfit":
        from .misfit_agent import Misfit  # noqa: F401  (imported for re-export)
        return Misfit
    raise AttributeError(f"module 'misfit_agent' has no attribute {name!r}")
