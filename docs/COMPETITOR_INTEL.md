# Competitor Intel — ARC Prize 2026

**Sources:** arcprize.org blog posts 2025-08 through 2026-05, pulled 2026-06-15.

> Read this BEFORE making any architectural decision in the sprint. The bar is much lower than common AI-Twitter discourse suggests.

---

## 1. The bar on ARC-AGI-3 (May 2026, semi-private set)

| Agent | Score | Notes |
|---|---|---|
| **GPT-5.5** | **0.43%** | Frontier closed-source, full reasoning compute |
| **Opus 4.7** | **0.18%** | Frontier closed-source, full reasoning compute |
| Random / brute-force | ~0-1% | Baseline |
| Our Tier-1 target | **3-10%** | Would top the leaderboard at current frontier |

**Source:** `https://arcprize.org/blog/arc-agi-3-gpt-5-5-opus-4-7-analysis` (2026-05-01)

### Frontier LLM common failure modes (all 3 are universal)

1. **Local-vs-global rule misalignment** — model fixates on local pixel changes, misses the goal-level rule
2. **Training-data overfit** — model assumes the puzzle is one it's seen, gets it wrong when it's a novel composition
3. **Cross-level forgetting** — model solves level 1, fails to carry the learned rule to level 2

**Our counter for each:**
1. 50-dim fingerprint captures global episode state, not local pixel state
2. We have no training data to overfit — Tier-1 pure substrate
3. Resonance library + episodic memory carries learned policies across levels by design

---

## 2. ARC Prize 2025 Winners — last year's static-task results

**Source:** `https://arcprize.org/blog/arc-prize-2025-results-analysis` (2025-12-05)

| Rank | Team | Score | Method | Prize | Code/Paper |
|---|---|---|---|---|---|
| 1 | **NVARC** | 24.03% | Architects-TTT + TRM-based hybrid | $25k | [Kaggle code](https://www.kaggle.com/code/gregkamradt/arc2-qwen3-unsloth-flash-lora-batch8-queue-trm2/edit?fromFork=1) · [Paper](https://drive.google.com/file/d/1vkEluaaJTzaZiJL69TkZovJUkPSDH5Xc/view?usp=drive_link) |
| 2 | **ARChitects** | 16.53% | 2D-aware masked-diffusion LLM + recursive self-refinement + perspective-based scoring | $10k | [Code](https://www.kaggle.com/code/gregkamradt/arc-2025-diffusion/edit?fromFork=1) · [Paper](https://lambdalabsml.github.io/ARC2025_Solution_by_the_ARChitects/) |
| 3 | **MindsAI** | 12.64% | TTFT + augmentation ensembles + tokenizer dropout | $5k | [Code](https://www.kaggle.com/code/gregkamradt/mindsai-tufa-2025-v4/edit?fromFork=1) · [Paper](https://arxiv.org/abs/2506.14276) |

### Foundation's stated central pattern

> "Refinement loops emerged as the central theme. A refinement loop iteratively transforms one program into another, where the objective is to incrementally optimize a program towards a goal based on a feedback signal."

### Two dominant paradigms

1. **Program synthesis** using evolutionary search (natural language or Python-based)
2. **Deep learning with zero pretraining** — exemplified by **TRM: 45% ARC-AGI-1 / 8% ARC-AGI-2 with only 7M-parameter network**

> Tier-1 implication: zero-pretraining deep learning IS within our Tier-1 envelope. The TRM result is the floor we should target if we adopt that paradigm. Our priors+substrate approach is the alternative paradigm — equally legitimate per the foundation.

---

## 3. HRM analysis — what actually drives performance

**Source:** `https://arcprize.org/blog/hrm-analysis` (2025-08-15)

The ARC Prize team disassembled HRM's architecture and found:

| HRM component | Real contribution | Hidden driver |
|---|---|---|
| Hierarchical H-L blocks | **~5pp over plain transformer** | not the win |
| **Outer refinement loop** | **+13pp from 0→1 refinement, then doubled 1→8** | **THE WIN** |
| Pre-training task augmentation | Essential (300 augmentations near max, not the claimed 1000) | data > architecture |
| Cross-task transfer | **Negligible** (31% with eval-only training vs 41% claimed) | "zero-pretraining test-time training" |

### Direct implication for our substrate

The hierarchical-reasoning hype is mostly outer-refinement-loop + data-augmentation in a trench coat. We can replicate both:

- **Outer refinement loop:** wrap our `world_model.fit() → predict() → observe() → fit_again()` cycle with k=1-8 iterations per episode. Cheap.
- **Data augmentation:** for ARC-AGI-2 static tasks, augment train pairs with rotations/reflections of grids (color-permutation invariant). Cheap.

Both are already partly in our design (refit every 5 steps, fingerprint cosine similarity for transfer). Make them first-class.

---

## 4. ARC-AGI-3 30-day preview learnings

**Source:** `https://arcprize.org/blog/arc-agi-3-preview-30-day-learnings` (2025-08-19)

### Three preview-game categories

| Category | Example | Skill required |
|---|---|---|
| Agentic / map-based | **ls20** | Navigation with symbol transformations |
| Logic-based | **ft09** | Pattern matching with overlaps |
| Orchestration | **vc33** | Multi-object volume manipulation |

### Preview winners and their methods

- **1st StochasticGoose:** "Convolutional Neural Network Action-learning agent" using RL to predict which actions cause frame changes → more efficient exploration than random
- **2nd Blind Squirrel:** State-graph building agent that prunes loop-creating actions + ResNet18-based value model to rank (state, action) pairs

### Core insight (foundation)

> "Humans are generally good at this: they explore briefly, then execute successfully. Random brute-force agents... require far more actions."

> "Action efficiency provides a clear intelligence signal"

### Our alignment

- ClickQuantizer reduces ACTION6 4096-cell space to ~5-20 priors-derived candidates — action efficiency by design
- AbstainPolicy preserves wall-clock budget for solvable games — quadratic-scoring-aware
- World model's "internal operations don't count" exploitation — thousands of free rollouts per real action

---

## 5. Human performance (April 2026 baseline study)

**Source:** `https://arcprize.org/blog/arc-agi-3-human-dataset` (2026-04-14)

- **458 participants**, 90-min sessions, $130 + $5/solve
- Diverse demographics (no single-cohort confound)
- Public demo set: 25 environments, 342 plays, 145 solves (~42% solve rate per attempt)
- Easiest: **r11l** (10/10 solved)
- Hardest: **tr87** (6/12 solved)
- Onboarding cliff: **cd82** (2/11 stuck at level 2)

### Critical scoring change

> "The human baseline which normalizes scores moves from 2nd-best player to MEDIAN player per level."

**Implication:** the `human_baseline_actions` denominator just got LARGER (median > 2nd-best in action count). Our `(human/agent)^2` ratio improves for the same agent performance.

---

## 6. Strategic posture (synthesized)

### What this changes about our sprint

1. **Score target downward** — we previously aimed for "10% would podium." Reality: **3% would podium** at current frontier. **5-10% likely tops the leaderboard.**
2. **Refinement loops are mandatory** — both 2025 winners + HRM analysis converge. Wrap `world_model.fit` in an outer 1-8 iteration loop. Add to sprint Day 9-10.
3. **Action efficiency is THE metric** — not levels completed. Our ClickQuantizer + AbstainPolicy + MCTS-PUCT-with-free-rollouts are correctly aimed.
4. **Paper Track odds just spiked** — 72 teams, $450k pool, frontier is at noise floor. ANY honest writeup with even modest substrate results is publishable. The Tier-1 disclosure regime alone is novel framing.
5. **Study NVARC, ARChitects, MindsAI code** — open-sourced last year. Mine for: refinement loop patterns, TTT/TTFT tricks, augmentation pipelines.

### Honest bar adjustment

| Old target | New target | Reason |
|---|---|---|
| "100 or past" (operator stretch) | "5% on Milestone #1 likely podiums" | Frontier is at 0.43%, even 3% wins something |
| Tier-1 + Tier-2 ensemble | **Tier-1 strict — winning move** | Frontier LLMs SCORE NEAR ZERO; LLM augmentation hurts here |
| Resonance library as nice-to-have | Resonance library + outer refinement = the core | Both 2025 winners + HRM converge |

---

## 7. Open-source competitor code to study (Day 9 reading list)

| Source | What to extract |
|---|---|
| [NVARC Kaggle code](https://www.kaggle.com/code/gregkamradt/arc2-qwen3-unsloth-flash-lora-batch8-queue-trm2/edit?fromFork=1) | Architects-TTT pipeline; how TRM is wired in |
| [ARChitects 2D-masked-diffusion](https://lambdalabsml.github.io/ARC2025_Solution_by_the_ARChitects/) | Perspective-based scoring; recursive self-refinement loop |
| [MindsAI TUFA paper](https://arxiv.org/abs/2506.14276) | TTFT details; augmentation ensemble construction |
| [TRM paper / repo](https://huggingface.co/papers/2503.10620) | The 7M-param zero-pretraining model that scores 45% on ARC-AGI-1 |
| [HRM analysis](https://arcprize.org/blog/hrm-analysis) | Outer refinement loop implementation details |

### Tier-1 honesty caveat when studying these

Reading competitor code does not contaminate us — we are reading **methods**, not loading their **weights** or copying their **training corpora**. The Tier-1 disclosure tests (`test_tier1_attestation.py`) catch contamination; method study is allowed and disclosed in `docs/CITATIONS.bib`.

---

**Next action:** Add outer-refinement-loop to sprint Day 9-10. Update `mission.yaml` target band. Open competitor-code-study task for Day 11 (post-LakeStrike).
