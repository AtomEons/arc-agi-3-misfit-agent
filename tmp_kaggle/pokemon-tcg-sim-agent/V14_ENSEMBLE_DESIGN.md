# PTCG Agent v14 — Multi-Agent Ensemble Voting Design Document

**Status:** DESIGNED (not implemented). Variance-reduction layer over v7/v8/v9.
Reference agents: `agent_v7.py` (priority schema), `agent_v8.py` (2-ply minimax),
`agent_v9.py` (multi-rollout opponent model).
Tier-1 strict — deterministic vote tally, no learned weights, no LLM.

## 1. Motivation and the Ensemble Honesty Question

Ensembling three weak signals does NOT automatically yield a strong one. The
classical condorcet-jury bound requires both (a) per-voter accuracy `p > 0.5`
and (b) **independent error**. Three correlated agents that fail on the same
states give a vote that is no better than any one of them, and worse than the
best one by the cost of the losers.

v14 is worth building only if v7/v8/v9 have **divergent failure modes** —
they must be wrong on different board states. That claim is defended in §3.

## 2. Architecture

```
def agent(obs_dict):
    obs = to_observation_class(obs_dict)
    if obs.select is None:
        return _own_deck()

    votes  = {}      # option_index -> count
    crashes = 0
    deadline = time.monotonic() + 0.600
    for fn, name in [(_v7, "v7"), (_v8, "v8"), (_v9, "v9")]:
        try:
            with _time_cap(0.200):
                choice = fn(obs)
            votes[choice] = votes.get(choice, 0) + 1
        except Exception:
            crashes += 1
        if time.monotonic() > deadline:
            break

    if crashes >= 1 or not votes:
        return _v8(obs)                       # crash-fallback to best single
    return _tally(votes, tiebreak="v8")       # majority, v8 breaks ties
```

Wall budget: **600 ms / decision**, **200 ms / agent invocation** (hard cap
via `time.monotonic()` polled inside each agent's existing decision loop;
hitting the cap counts as a crash). Tally: 2-of-3 majority wins outright;
otherwise prefer v8's choice.

## 3. Why the Failure Modes Are Divergent

| Agent | Mechanism            | Dominant failure mode                                                |
|-------|----------------------|----------------------------------------------------------------------|
| v7    | Priority schema      | Loses on positions where the schema's hand-coded order misranks      |
|       |                      | a sharp tactical line (no lookahead, no opponent reasoning)          |
| v8    | 2-ply minimax        | Loses on horizon effects past depth 2 and on positions where the     |
|       |                      | leaf score `_score_state` mis-evaluates a delayed payoff             |
| v9    | Multi-rollout opp    | Loses when opponent rollouts mis-estimate opp policy variance        |
|       |                      | (rollout sampling noise dominates at small N)                        |

These error sources are structurally different. v7 fails on **policy
mis-ranking**, v8 on **horizon and leaf-evaluation**, v9 on **rollout
sampling**. A state that fools v7's priority order will not generally fool
v8's search (which re-ranks via leaf score) or v9's opponent rollouts
(which marginalize over opp actions). The errors are not perfectly
independent — all three share v7's priority schema as a prior — but the
**residual** failures past the shared prior are uncorrelated by construction.

## 4. How v14 Could Plausibly Lift Over v8 Alone

v8 is the strongest single agent. Ensembling can only beat v8 if the cases
where **v7 + v9 both override v8 and are right** outnumber the cases where
**v7 + v9 both override v8 and are wrong**.

The mechanism: v8 has high variance on positions whose true value lives past
its 2-ply horizon. On those positions, v8's leaf-score is noisy, v7's
priority schema gives a low-variance (if biased) signal, and v9's opp
rollout gives a low-variance estimate of opp response. If two of three
agree against v8, that agreement is evidence that v8's horizon noise is
the source of disagreement — and the majority is correct.

This is variance reduction, not bias reduction. v14's ceiling is bounded
by the **bias of the majority** — if v7 and v9 share a systematic blind
spot, v14 inherits it. Expected lift over v8: **2–5 pp** in a tight regime,
zero or negative if v7/v9 errors are correlated with v8's.

## 5. Tally Rule and Tiebreak

- **Unanimous (3-0):** ship it.
- **Majority (2-1):** ship the majority choice.
- **All distinct (1-1-1):** tiebreak — prefer v8's choice (highest individual
  win rate in arena). This makes v14 strictly dominate v8 in the limit where
  v7/v9 never agree: every disagreement defaults to v8.
- **All crashed / no votes:** route to `_v8(obs)` directly.

## 6. Time Budget

- 600 ms wall clock per decision.
- 200 ms hard cap per agent (timeout = crash for fallback).
- Sequential execution — Kaggle kernel is single-threaded under the
  competition gateway; thread-based parallelism is unreliable across the
  `KAGGLE_IS_COMPETITION_RERUN` boundary.
- If wall deadline hits mid-vote, tally what has voted (1-2 votes allowed)
  and apply tiebreak.

## 7. Fallback Discipline

Any single agent crash → route directly to `_v8(obs)` for the current
decision. Do not attempt to re-tally with two votes — a crash is signal
that the game state is anomalous (illegal options, parser drift) and the
strongest single agent is the safest commit.

## 8. Non-Goals

- No weighted voting (Tier-1 strict; weights are learned parameters).
- No agent-correlation calibration online (offline arena measurement only).
- No expansion past three agents (diminishing returns vs latency cost).
- No shared cache between agents (independence of decision is the asset).

## 9. Arena Measurement Plan (post-implementation)

Required before claiming v14 > v8: N≥200 self-play games per matchup,
95% CI on win-rate delta, p-value via paired binomial. Honest verdict
only if `v14 vs v8` lifts ≥ 3 pp with `p < 0.05`. Otherwise revert.
