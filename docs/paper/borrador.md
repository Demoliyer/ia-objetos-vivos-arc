# Offering Is Not Using: Verified Simulators for an LLM Game Agent in ARC-AGI-3, and the Living-Objects Rules Behind Them

**Franco González (Demoliyer)**
Independent. Code: `Demoliyer/ia-objetos-vivos-arc` (MIT-0).

*Draft v3, 2026-10-20 (corrects v2: the two "base" hidden-set runs were not the unmodified harness, and the 29.28 did not replicate; adds the pace analysis, adoption results, the rule-source experiment and the traceability protocol). Not for distribution. Numbers marked `<!-- src: ... -->` point to where they come from; anything marked [PENDING] has not been measured yet.*

*AI assistance: Claude (Anthropic) was used extensively for writing code and for drafting and editing this text. Design decisions, experiments and submissions were directed and run by the author.*

---

## Abstract

ARC-AGI-3 scores an agent by how few actions it needs to clear unseen interactive games, so every wrong belief about the game costs score. We start from a contract inherited from an earlier project, *Living Objects*: evidence is trusted, beliefs are not, and a belief may guide action only after it explains all the evidence recorded for it. We apply it to a strong public LLM agent (Daniel Franzen's adaptation of the Tufa Labs Duck harness) by giving the model, inside its Python sandbox, two zero-cost tools: `check_model(step)`, which replays every observed transition of the level through a simulator `step(grid, action)` written by the model, and `plan_with_model(step, goal, actions)`, which searches inside that simulator.

The main finding is behavioural and, we think, general: **offering verification is not the same as getting it used**. With the tools only described in the prompt, the model called them zero times in about 450 turns on four public games, although it ran more than 20,000 Python snippets of its own analysis. A one-line reminder per turn raised use to one game in four; letting `step` return only the cells it changes and checking it automatically after every snippet raised it to two in four. Where the model did commit to a simulator, it cleared levels with fewer actions (in one game 15, 19 and 22 actions on levels 2–4 against 21, 28 and 42 for the base), but spent more model tokens per level and ran out of time before the last levels; the cost of verification is wall-clock, not actions. Many of its simulators hard-coded level coordinates, which we detect with a one-cell shift test. On the hidden set, the configuration in which the model uses the tools scored 29.28 and then **25.71 with identical code**, against 26.39 and 25.38 for a configuration that only describes the tools; we therefore claim no score gain yet, and report a pre-registered nine-submission comparison against the unmodified harness [PENDING]. An analysis of 64 recorded runs explains why the gain is small: almost all lost score comes from levels the agent does not reach before its time budget ends (27.5 points per game on average, against 2.7 from extra actions), and a verified simulator saves actions, not time. Two further results: making a richer verification API more prominent did not make the model adopt it (0 of 5 games), and the rules the LLM writes and the rules our engine learns by itself cover disjoint games (6% and 7% of levels, no overlap), which points to a hybrid in which the engine supplies rules at no token cost.

We also report, as background and as a negative result, the original Living Objects engine: a population of small, human-readable rules, each verified and repaired against its evidence, with a learned HUD mask, a state graph, a step budget and IW(1) novelty. It reached 0.94% on the public games and 0.07 on the hidden set, a 13× drop that we trace to mechanisms tuned to the public games and to a rule language that cannot express most goals.

---

## 1. Introduction

ARC-AGI-3 asks an agent to play turn-based games it has never seen, with no instructions, and scores it by how few actions it needs relative to humans. The agent must discover the mechanics and the goal by acting, and every exploratory action costs score. Today the strongest agents are LLMs driving a Python tool: they look at the board, write analysis code, act, and repeat.

Such agents fail in a characteristic way: they act on beliefs about the game that the observations already contradict, or that were never tested. Our starting point is a contract from an earlier project, Living Mesh: *specifications are trusted, implementations are not*. Transposed to games, each transition the game returned is trusted evidence, and every belief about the mechanics is an untrusted implementation that may be used only if it explains all of that evidence.

We first built an agent entirely out of that contract: a colony of small verified rules (Section 4). It improved steadily on the public games and collapsed on the hidden set. We then asked a narrower and more useful question: **what does the same contract add to an LLM agent that is already strong?** We let the model write its beliefs as an executable simulator and give it tools to verify the simulator against every observation and to plan inside it, at zero action cost (Section 3).

Our contributions are:

1. A small, dependency-free implementation of verified simulation and planning for the ARC-AGI-3 sandbox, grafted onto a public competitive agent without changing anything else in its official run (`vivos/simulador.py`, `kaggle/injerto_vivos.py`).
2. A measurement of whether the agent actually uses such tools, separating the model's own calls from the text we inject, across three designs of increasing "pull" (Section 3.4). Use went from 0/4 to 2/4 games.
3. Evidence on what verification costs and what goes wrong when it is used: more model tokens per level, and simulators that hard-code the current level's coordinates, which a one-cell shift test detects (Sections 3.6–3.7).
4. A documented negative result for a pure verified-rules agent, with its ablations and the reasons it overfits (Section 4).
5. An analysis of where the score is lost (time, not actions) and of the model's token budget, which bounds what any verification layer can add (Section 3.8).
6. A comparison of rules written by the LLM and rules learned by our engine on the same play, showing disjoint coverage (Section 3.10).
7. The measurement infrastructure that made the above possible on a budget of one hidden-set submission per day: non-submitted runs on public games with full transcripts and the harness's own score, and a traceability protocol that identifies exactly what each submission ran (Section 6).

## 2. Setting

**Task.** ARC-AGI-3 games are rendered as 64×64 grids with 16 colors. <!-- src: docs/IA_OBJETOS_VIVOS.md §1 --> The agent receives frames and a list of available actions: ACTION1–ACTION7 (in practice directional moves, a generic "space"-like action, a click with (x, y) coordinates, and undo) plus RESET. A game has several levels; the agent sees `levels_completed` increase when it clears one and a GAME_OVER state when it loses.

**Scoring.** Per level, the score is `min(1.15, h / a)^2`, where `h` is the human baseline action count and `a` the agent's action count; an unsolved level scores 0. A game's score is the average over levels weighted by level index (level *i* has weight *i*), so later levels dominate. <!-- src: vivos/puntaje.py, citing arc-agi kit docs 0.9.3–0.9.7 --> Two consequences shaped our design: solving level 1 slowly is worth almost nothing, and a level that takes 30 times the human count contributes about 0.001. <!-- (1/30)^2 ≈ 0.0011 -->

**Constraints.** Kaggle evaluation runs without internet access, in a single container, on Kaggle hardware (the starter kit offers CPU, 2×T4, P100 and RTX 6000 accelerators). <!-- src: kaggle/kit/README.md --> The hidden set contains on the order of 110 games <!-- src: empaquetar_kaggle.py docstring, "del orden de 110"; VERIFY against competition page -->, and the public leaderboard is computed on roughly half of them. <!-- src: claude/mediciones-arc-agi-3.md --> Our own agent runs on CPU only; locally we capped each public game at 2000 actions. <!-- src: medir_publicos.sh / mediciones doc -->

**Related work.** Writing the world model as code and checking it against interaction data is not new. WorldCoder (Tang, Key and Ellis, 2024) has an LLM write a Python transition and reward model, refines it until it explains the observed transitions, and plans in it; it introduces an optimism-under-uncertainty objective so that the model also guides exploration. Dainese et al. (2024) generate Code World Models with an LLM guided by Monte Carlo Tree Search over program refinements, verified against offline transitions, and plan with them. Before LLMs, Tsividis et al. (2021) showed human-level learning efficiency on Atari-like video games with EMPA, an agent that represents game mechanics as a theory in a probabilistic program, explores to test it and plans with it. Our contribution is narrower and more practical: (i) a verification contract applied to a population of tiny rules rather than one program, and (ii) the same contract offered as optional tools to an already strong, prompt-tuned LLM agent in a competitive setting with a fixed hidden test set, together with a measurement of whether the agent uses them. The novelty-based exploration priority in Section 4.4 is IW(1) from width-based planning (Lipovetzky and Geffner, 2012).

## 3. Verified simulators for an LLM agent

### 3.1 Motivation

The leaderboard makes clear that strong LLM agents are far ahead of our rule engine (Section 4). Reading Franzen's harness, his gains came from the model, token throughput, perception (animations, 10× image, diff image), UNDO and scheduling, not from the kind of guards we had built. <!-- src: mediciones doc --> So instead of competing with the LLM, we ask what our contract adds to it.

LLM game agents have characteristic failure modes: they assert mechanics that the observations contradict, and they re-take actions that already proved fatal or useless. Both are failures of *unverified beliefs*. The Living Objects contract addresses exactly that: a belief is usable only after it has been checked against all the evidence. The novelty is in who writes the belief. Instead of our fixed rule language, the LLM writes an arbitrary Python simulator, so expressiveness is no longer limited by our labels.

### 3.2 Design (`vivos/simulador.py`)

The model writes its hypothesis as a top-level function `step(grid, action) -> grid`, where `grid` is the board as rows of color characters (the same `.ascii` view the model already sees) and actions use the harness labels (`UP`, `DOWN`, `LEFT`, `RIGHT`, `SPACE`, `UNDO`, `MOUSE(row=R, col=C)`). Two tools are exposed:

- **`check_model(step)`** replays every recorded transition of the current level (excluding RESET, game-over, level-change and automatic transitions) through `step` and returns how many were checked, how many matched exactly, how many matched outside the 4-cell border (where bars live), `all_interior_ok`, and the first mismatches with example cells `(row, col, actual, predicted)`. A crashing simulator counts as a mismatch, not an error. It costs **zero game actions**.
- **`plan_with_model(step, goal, actions)`** runs BFS inside the simulator from the current frame to a state satisfying `goal` (a predicate or a goal board compared on the interior), comparing states on the interior so that a shrinking bar does not prevent loop detection. It is capped at 4,000 states and 8 s by default (all calls in one Python snippet share a ~24 s budget, below the sandbox's 30 s limit) and returns the shortest plan or a reason. <!-- src: simulador.py defaults max_nodes=4000, time_limit=8.0, _PRESUPUESTO_PROCESO=24.0 -->

```
loop (inside the LLM's Python tool):
    write step(grid, action)
    r = check_model(step)
    while not r.all_interior_ok: fix step using r.mismatches; r = check_model(step)
    p = plan_with_model(step, goal, actions)
    execute p.plan with action(...); on divergence, step is wrong -> re-check
```

The system-prompt addition is short and states the rule explicitly: "a simulator that has not passed `check_model` is only a guess." <!-- src: kaggle/injerto_vivos.py AYUDA_PROMPT -->

### 3.3 Integration

The harness's Python sandbox runs with `python -I -S`, pure standard library, a whitelist of imports, and a CPU-time limit per call. `simulador.py` is therefore pure Python (no NumPy), budgeted in time and nodes, and never raises on a faulty `step`. `kaggle/injerto_vivos.py` patches the harness at notebook time: it embeds the simulator source into the sandbox bootstrap, binds `check_model`, `plan_with_model` and `mouse` next to the existing `frame_diff` helper (passing the sandbox's `transitions` and `current_frame`), and appends the usage notes to the existing `FRAME_DIFF_HINT_ADDENDUM` in the prompts. The patch is idempotent and fails loudly if its anchors are not found. `kaggle/preparar_franzen.py --vivos` applies it to Franzen's notebook; a "fast save" mode plays 2 games for about 15 minutes instead of the original 9-hour public-games pass, so each experiment costs roughly 30–40 GPU minutes. <!-- src: preparar_franzen.py docstring; mediciones doc --> The fast save only produces a valid submission file: the model server needs about 12 minutes to start, and in a 15-minute save the games ended with connection errors before the model answered once. Behavioural measurements therefore use longer non-submitted runs (`--juegos=… --minutos=45`) whose outputs are summarised by `kaggle/analizar_salida.py`. <!-- src: mediciones doc, 2026-10-16/17 -->

The tools were tested end to end inside the real harness sandbox with recorded trajectories (`tests/test_simulador.py`, 12 tests; `tests/test_kaggle_injerto.py`, which grafts a copy of the real harness and runs the tools inside its sandbox). <!-- src: tests/ --> A known risk: the Tufa Labs team observed that hand-added tools sometimes reduce score, presumably by distracting the model. <!-- src: mediciones doc; ADD exact source --> Only a submission against the unmodified base can decide.

### 3.4 Results

**Hidden set.**

| Configuration | Hidden-set public LB score |
|---|---|
| Tools described in the prompt (`--vivos`), run 1 | 26.39 <!-- src: Kaggle submission 57265504, 2026-10-16, kernel version 1 --> |
| Tools described in the prompt (`--vivos`), run 2 | 25.38 <!-- src: Kaggle submission 57289126, 2026-10-17, kernel version 2 --> |
| + reminder, partial `step`, automatic check (`--vivos-auto`), run 1 | 29.28 <!-- src: Kaggle submission 57331916, 2026-10-18, kernel version 14 --> |
| same code, run 2 | 25.71 <!-- src: Kaggle submission 57415286, 2026-10-20, kernel version 21; notebook fingerprint identical to run 1 --> |
| `vivos-v3` (cost planner, action brake, level notices) | 26.70 <!-- src: Kaggle submission 57375180, 2026-10-19, kernel version 19 --> |
| Unmodified harness | [PENDING: 3 runs, 22, 25 and 28-oct] |
| `--vivos-auto` | [PENDING: 2 more runs, 23 and 26-oct] |
| `vivos-v4s` (v4 without the brake) | [PENDING: 4 runs, 21, 24, 27 and 29-oct] |

Correction to draft v2: we had treated the two `--vivos` runs as samples of the base agent, because the model never called the tools in that configuration (below). They are not the unmodified harness: `--vivos` adds the tool description to the prompt, which changes the model's input even when the tools are never called. We have no hidden-set sample of the unmodified harness yet; Franzen reports 27.89 for it. The replication of `--vivos-auto` (29.28, then 25.71 with an identical notebook, verified by fingerprint) also shows that the run-to-run spread is larger than the one point we had assumed. We therefore pre-registered, before seeing any of the results, an interleaved schedule of nine submissions (unmodified harness ×3, `--vivos-auto` ×2, `vivos-v4s` ×4), a comparison by means without claiming superiority unless the difference clearly exceeds the spread observed in the final data, and a rule for the four reserve submissions (Section 6). [PENDING: results] <!-- src: Franzen's Kaggle notebook page; VERIFY wording -->

**Does the model use the tools?** We ran the agent on four public games (ls20, ft09, vc33, tu93) for 45 minutes each without submitting, and counted, in its transcripts, the simulators it defined and the tool calls it made, after removing the text we inject ourselves (the prompt description and, in the second condition, the reminder, which quote `def step(grid, action)` and `check_model(step)` verbatim; an earlier count that did not remove them over-reported use). <!-- src: kaggle/analizar_salida.py; mediciones doc 2026-10-17 -->

| Condition | Requests | Actions | Games with a `step` | `check_model` calls | `plan_with_model` calls |
|---|---|---|---|---|---|
| Prompt description only (`--vivos`) | 457 | 977 | 0 / 4 | 0 | 0 |
| + one-line reminder every turn (`--vivos-fuerte`) | 458 | 1121 | 1 / 4 (tu93) | 8 | 2 |
| + reminder, partial `step` allowed, automatic check after every snippet (`--vivos-auto`) | 474 | 1069 | 2 / 4 (ls20, tu93) | 5 by the model, 148 automatic | 4 |

In the second condition the model, in tu93, wrote nine versions of `step`, printed four `check_model` reports covering 20 transitions in total (one of them perfect on its interior) and obtained one plan of 17 actions. In ls20, ft09 and vc33 it kept its usual behaviour: it defined many helper functions of its own (`runs`, `lattice`, `cell_of`, …), over 20,000 Python calls in total, but never a full-board transition function. One run per condition; the difference in actions (977 vs. 1121) is within what we would expect from noise.

### 3.5 Why the tools are not used

We read the transcripts for an explanation and offer three, in decreasing order of confidence.

1. **A full-board simulator is a large commitment.** The model's own code is local: it segments objects, measures displacements and tests one hypothesis at a time. A `step` that must return the entire next 64×64 board asks it to commit to *every* mechanic at once, including the ones it has not yet looked at. The single game where it did so (tu93) has a small, regular board and few moving parts.
2. **The payoff is not visible at the moment of decision.** `check_model` only says how wrong a simulator is after the model has written one; nothing in the turn shows the model that a verified simulator would have saved actions.
3. **Prompt competition.** The harness prompt is long and already prescribes a working loop (maintain a world model, test, act). An extra paragraph near the end, or a single line per turn, competes with instructions that the harness authors tuned against measured scores. This matches the Tufa Labs observation that hand-added tools are often ignored or even harmful. <!-- ADD exact source -->

These point to the obvious next design, which we are testing: make verification *pull* rather than *push*. Whenever a `step` is retained, the harness itself replays the last action through it and reports, in the turn message, whether the prediction was right and which cells were wrong; and `step` may return only the cells it predicts to change, so that a partial theory ("RIGHT moves the green block one cell") is already checkable. This variant (`--vivos-auto`, last row of the table) doubled the number of games where the model committed to a simulator (ls20 joined tu93); the automatic check ran 148 times, and in tu93 seven automatic reports showed a simulator that reproduced every recorded transition of the level. ft09 and vc33, both click games with many objects, remained untouched. On the hidden set this variant scored 29.28 and, with identical code, 25.71; the description-only variant scored 26.39 and 25.38. With two samples per arm this does not establish a gain (Section 3.4).

Decision rule used at the time: a variant was submitted only if it changed behaviour in at least two of the four measurement games. The rule "keep it if it exceeds the spread of two base samples" proved too weak (see the replication above) and was replaced by the pre-registered schedule.

### 3.6 What verification costs: time, not actions

We compared the base agent and `--vivos-auto` on the same eight public games, one 40-minute run each, using the harness's own per-game score (the official formula ×100). The table is a screen, not a result: public games are much easier for this agent than hidden ones (≈69 vs. ≈26 for the base), and a single game moves between 48 and 100 from one run to the next.

| Game | Base | `--vivos-auto` | Model wrote a `step`? |
|---|---|---|---|
| ar25 | 100.0 | 100.0 | no |
| cd82 | 48.9 | 100.0 | no |
| ft09 | 47.6 | 100.0 | no |
| lp85 | 100.0 | 100.0 | yes |
| ls20 | 12.7 | 6.5 | no (yes in other runs) |
| sb26 | 97.6 | 100.0 | no |
| tu93 | 67.1 | 33.3 | yes |
| vc33 | 75.0 | 100.0 | no |
| **Mean** | **68.6** | **80.0** | |

<!-- src: living-mesh-datos salidas/20261017-1358-banco8-base and 20261017-1450-banco8-auto, benchmark.json final_score -->

The mean rises, but the largest gains are in games where the tools were not used, so we read them as noise. A second run of the unmodified base on the same eight games confirms how large that noise is: it scored **50.4** against 68.6 for the first run, with per-game differences of up to 58 points (mean absolute difference 31). The four runs also split by how many tokens the model server generated in the same 40 minutes, presumably reflecting differences between Kaggle machines: the two "fast" runs (base 68.6 and `--vivos-auto` 80.0, ≈1.12–1.13M tokens) scored far higher than the two "slow" ones (base 50.4 and the first `vivos-v3` 57.3, ≈0.93–0.94M tokens). Within each throughput pair the variant with tools scored above the base (+11.4 and +6.9), but with one run per cell this is a hint, not a result. The practical consequences: a single public-games run cannot rank variants; any comparison must at least control for generation throughput; and the hidden-set submission remains the only reliable judge, although its own run-to-run spread is several points (Section 3.4). <!-- src: living-mesh-datos salidas/20261017-1732-banco8-base2; generated_tokens summed over benchmark.json history --> The informative game is tu93, where the model did use its simulator:

| Level | Base: actions, minutes, generated tokens | `--vivos-auto`: actions, minutes, generated tokens |
|---|---|---|
| 1 | 18, 1.2, 5k | 18, 4.2, 18k |
| 2 | 21, 8.3, 36k | 15, 6.9, 31k |
| 3 | 28, 2.4, 11k | 19, 2.1, 9k |
| 4 | 42, 5.6, 23k | 22, 8.5, 38k |
| 5 | 29, 5.0, 22k | 29, 8.7, 39k |
| levels cleared | 8 of 9 | 5 of 9 |

<!-- src: benchmark.json history, wallclock_seconds and generated_tokens per action -->

With a verified simulator the model needed fewer actions per level, which is what the metric rewards, but it spent roughly 1.5–3× more generated tokens per level and ran out of time before levels 6–9. The overhead is in the model's own generation, not in the tools' Python execution. Part of it is re-synthesis: the model rewrote `step` at every level (about ten definitions in five levels).

### 3.7 Simulators that do not generalise across levels

The functions retained by the harness survive across levels, so a simulator that is right on level *L* could be reused on level *L+1* without re-exploring. Reading the simulators the model wrote shows why this rarely happens: in tu93 they located objects by colour in the grid, but in lp85 they enumerated click coordinates and object positions by hand, and in ls20 the goal cells were literal coordinates. Such a simulator reproduces every transition of the level where it was written and is wrong on the next one, so "it explains the earlier levels" is not evidence that it fits the new layout.

We added two cheap checks that cost no prompt tokens unless they fire: a **shift test**, which translates the interior of the board (and any click) by one cell and requires the simulator's prediction to translate with it, and a check that the simulator does not read the live board from globals instead of its `grid` argument. The automatic check suggests reusing a simulator on a new level only when it passes the shift test; otherwise it warns that the simulator probably hard-codes positions. Because some games legitimately depend on absolute positions, the warning never blocks the simulator. These checks, together with a cost-based planner that returns the best partial plan when the goal is out of reach and a brake that asks the model to act after four turns without a game action, formed `vivos-v3`, which scored 26.70 on one hidden-set run. We suspected the brake (it fired seven times in a game where the model's simulator already explained every transition), removed it in `vivos-v4s`, but the effect of the brake is not identified by one run. A competing hypothesis, that the 1-second cap of the automatic check was cutting it short, was refuted by the data: zero cuts in five recorded runs.

### 3.8 Where the score is lost: pace, not actions

We decomposed the score of all 64 recorded public-game runs (11 runs, 8 games, every configuration). Per game, the loss relative to a perfect score splits into levels never cleared and extra actions on cleared levels. On average **27.5 points per game come from levels not reached and 2.7 from extra actions**. All 37 runs that did not win were stopped by the time budget, not by a game over, and the median time spent on the last, unfinished level (5.0 min) is no longer than the median time per cleared level (4.8 min): the agent is not stuck on one level, it is simply not fast enough to reach the later, heavily weighted ones. <!-- src: living-mesh-datos salidas/*/benchmark.json; official formula reproduced with zero error on all 64 games -->

A verified simulator mainly saves actions, which is the small term. Its cost is the large one: model tokens, and therefore time. Per cleared level the base used ≈27.8k generated tokens and our variants ≈26.5k, with large per-game differences in both directions. <!-- src: benchmark.json history, generated_tokens -->

Token budget per decision. Between two executed actions the model generated a median of 2.8k tokens (p90 10.6k, p99 25.2k, max 86k; 2,284 intervals). Intervals above 8k tokens are 16% of all intervals but 51.5% of all generated tokens. They are not single long thoughts: an interval above 12k tokens is a median of six requests with Python execution in between, in which the largest request accounts for 39% of the tokens, and the model ignored a median of three "you yielded, act now" prompts (the harness yields after 2,048 tokens per turn). Only 20 of 4,867 requests (0.4%) hit the 12,288-token output cap. Long intervals complete a level at the same rate per interval as short ones (14–15%), but at a much lower rate per token, and 85% of their tokens are spent on levels that were eventually cleared. We therefore did not test a token cap: the hard cap does not touch long intervals, the soft cap is already ignored, and a hard cap per interval would also cut productive work (in one 45k-token interval the model built a simulator that reproduced 62 of 62 transitions and a verified 42-action plan). <!-- src: revision package 2026-10-20, scripts/analisis_tokens.py -->

### 3.9 Making a tool more prominent does not make it used

We tried a richer verification API: an optional state model (`observe(grid)` returning a small dictionary of named fields, `advance(state, action)`), checked field by field, with blind-spot detection, a planner over frozen states, per-action-family reports and a non-repeating reminder. Placing it first in the instructions did not change adoption: **0 of 5 games** in two runs with different orders of the help text. In a third run the model adopted it spontaneously in one game (vc33) and it predicted 97 of 97 transitions, but that was a game where the model already tends to summarise the board into abstract state. A later variant that explained each simulator error in terms of objects ("in the game the red object moved by (0, +1), your step predicts no change") produced 44 correct messages that the model did not act on. The pattern is consistent: the model uses what it already does (writing `step`) and verification of that; it does not adopt new interfaces because they are offered, explained or ranked first. <!-- src: mediciones doc, medir-v4/v4b/v4c/v5 -->

### 3.10 Who should write the rule?

If simulators help only when they are cheap, they could come from the rule engine of Section 4 instead of from the LLM. We measured both sources on the same recorded play (12,307 transitions). For the LLM, a level counts if the automatic check reported a `step` that explains every recorded transition of the level (at least 3); this is in-sample, which favours the LLM. For the engine (`vivos/mundo.py`, learning online, no LLM), a level counts if, after its third changing transition, it predicts at least 90% of the following ones exactly on the interior; this is out-of-sample.

| Game | LLM wrote any `step` | LLM complete rule | Engine complete rule |
|---|---|---|---|
| ft09 | 0% | 0% | **53%** |
| tu93 | 81% | **15%** | 0% |
| lp85 | 41% | **11%** | 0% |
| ls20 | 17% | 0% | 0% |
| vc33, ar25, cd82, sb26 | 0–2% | 0–2% | 0% |
| **All** | 28% | **6%** | **7%** |

The two sources are complementary: the engine succeeds exactly on the click game where the LLM never writes a simulator, and the LLM on the two games where the engine fails, with no game covered by both. Together they still reach a complete rule on only ≈13% of levels, and in the LLM's own data a complete rule is not associated with faster levels (median 3.9 min with a complete rule against 3.5 min without, confounded by difficulty). The hybrid that follows from this — the engine hands verified rules to the LLM at no token cost, and the LLM completes the rest — is future work; its ceiling is set by the expressiveness of the engine's rule language, which today learns useful rules in 3 of 8 games. <!-- src: experimentos/reglas/ (branch exp-reglas-motor) -->

## 4. Background and negative result: the Living Objects rule engine

This section describes the agent we built first, entirely out of the verification contract, and why it did not compete. The engine (`vivos/`) has five parts: perception, the rule colony, a state graph, a decision policy with novelty and budget, and an experimental simulable world model. It does not import the official kit; it receives frames as arrays and returns action names.

### 4.1 Perception (`vivos/percepcion.py`)

Each response's *last* frame is the state. Objects are 4-connected components of one color; a letterbox or perimeter wall is a "frame" and never a goal. The background is the most frequent color inside the play area, used only as a first guess of "floor".

**Clock/HUD masking.** Many games have step counters, energy bars or timers that change on every action regardless of what the agent does. If they are not masked, every state looks new and the state graph never closes a loop. A color is declared a clock if it changed in at least 80% of transitions (minimum 4), its most common change signature (a shift, or a change in cell count) covers at least 60% of its changes, and that signature appeared in at least two distinct contexts (different action, or same action with and without other changes). <!-- src: DetectorDeRuido docstring and constants UMBRAL=0.8, MINIMO=4, COBERTURA=0.6 --> The mask is the bounding box of the clock's zone *per signature*, not its whole row or column, so the same color can be a clock in the HUD and a game piece elsewhere. Two guards were added after specific failures: a clock is "thin" (at most 6 cells thick in one dimension) and is not extended into a floor of the same color, and clocks inherited from the previous level are restricted to the neighborhood of where they lived (in one public game, inheriting by color alone masked 1,980 floor cells and blinded the agent). <!-- src: percepcion.py heredar() docstring, GROSOR_RELOJ=6 -->

**Effects and rigid translation.** The effect of a transition is computed per object outside the mask. Because the moving sprite is often multicolored and partially occludes the floor, we first look for a rigid multicolor translation that explains most of the local change before falling back to per-color comparison. Translation and "dis-occlusion" of the background are mirror explanations of the same diff; we break the tie by preferring the sprite whose main color is *rarer* in the frame ("the rare thing moves, the abundant thing is background").

### 4.2 Rules with self-verification (`vivos/reglas.py`)

A **slot** is a stable identity such as `accion:ACTION1`, `contacto:9` (touching color 9) or `click:2` (clicking an object of color 2). Each slot holds at most one rule, a short label: `inerte`, `mueve:<color>:<dx>:<dy>`, `bloquea`, `atraviesa`, `fatal`, `activa`, `quita`, `cambia`. Each transition produces an observation per affected slot; a rule's meaning is entirely defined by a judge function returning +1 (confirms), −1 (contradicts) or 0 (not applicable).

```
on new observation obs for slot s:
    s.evidence.append(obs)
    if s.rule is None: repair(s, obs)
    elif judge(s.rule, obs) < 0:
        bad, good = verify(s.rule, s.evidence)       # good = distinct states confirming
        if tolerable(bad, good): s.anomalies += 1    # good >= 8 and 10*bad <= good
        else: s.state = DEGRADED; repair(s, obs)

repair(s, obs):
    candidates = labels supported by obs  ∪ {s.rule, s.last_known_good}
    keep c if tolerable(verify(c, s.evidence))
    pick most confirmations, then fewest anomalies, then most specific
    if none: s.failed += 1; if s.failed >= 3: s.state = QUARANTINED
```

Confirmations count *distinct states*, so repeating an action in the same state adds no evidence. A rule is usable for decisions only when healthy and confirmed a minimum number of times (e.g. 3 for "inert", 2 for "blocks", 1 for "fatal"). <!-- src: reglas.py _MINIMO --> The anomaly tolerance was added because a strict contract broke on legitimate rare events: in one game the avatar respawning at the start after losing a life looks like a teleport and used to destroy a well-established "moves" rule. <!-- src: commit 48af93b --> The states HEALTHY / DEGRADED / QUARANTINED and the threshold of three failed repairs are inherited unchanged from Living Mesh.

### 4.3 State graph and world model (`vivos/grafo.py`, `vivos/mundo.py`)

States are hashed with the clock mask applied. The graph records, per node, which actions were tried and where they led, and a per-action count of game-overs; an action is considered fatal from a state after two deaths (one death may be due to an exhausted energy bar). When the mask changes, nodes that become equal are merged. Shortest known paths are found by BFS.

A separate simulable world model (`vivos/mundo.py`) predicts the next frame from three kinds of rules: a rigid sprite per displacing action (with learned blocking colors and floor memory), a relative patch per click on an object type, and exact memory of previously seen (state, action) pairs. On recorded trajectories, 58% of transitions with any change were merges, splits or shape changes of single-color components, which is why these rules do not depend on segmentation. <!-- src: vivos/mundo.py docstring; VERIFY with evaluar_mundo.py output --> The world model is measured offline (`evaluar_mundo.py`) but **is not yet used by the agent to decide**; we mention it because it is the bridge to Section 3.

### 4.4 Decision policy: search, novelty and budget (`vivos/agente.py`)

```
decide(state):
    if a plan is pending and its next action is not fatal here: follow it
    if an avatar and a goal/contact target are known:
        BFS over avatar displacements on passable cells -> plan to touch target
    candidates = available actions (+ click candidates)
    priority(c) in {-1 fatal, 0 useless per colony, 1..4 promising}
    untried = candidates not yet tried here with priority > 0
    if untried and this state brought no new atoms:
        go to a novel state within 8 steps that has untried actions   # IW(1) as priority
    if untried: try the best one
    path = shortest known path to a state with untried actions
    if RESET + shortest path from level root is better (shorter, or fits the budget): RESET
    else follow path
    never RESET with 0 actions in the level
```

Key mechanisms, each added in its own commit and kept because it improved the 3-seed public score (Section 4.5):

- **Rare-type-first click candidates.** Click candidates are one representative per object type (color and bounding-box size), rare types first: what is repeated is rarely a button. <!-- commit 0b441b9 -->
- **Step budget with deliberate RESET.** If at least two lives ended in GAME_OVER after the same number of steps (±10%), that number is a budget. When the target no longer fits in the remaining budget, or the path from the level root is shorter than walking back, the agent resets and replays the shortest prefix. **The engine restarts the whole game (back to level 1) on a RESET issued with 0 actions in the current level**, so this is forbidden explicitly. <!-- commit c0b830b -->
- **Goal learned from the winning move.** When a level is cleared, the returned frame already belongs to the next level, so the winning move is projected onto the last frame of the cleared level to see which color the avatar would have stepped on; that color becomes a goal prior for later levels. The avatar's step unit is the size with which it was actually seen to move. <!-- commit 008cbe6 -->
- **IW(1) novelty as priority.** Atoms are (color, position, shape) of each non-frame object, with clocks masked. A state that introduced no new atom postpones its untried actions if a novel state with untried actions is within 8 steps. This is a priority, not a pruning rule as in the original IW(1) [Lipovetzky & Geffner 2012]. <!-- commit 41238b4, MAX_DESVIO_NOVEDAD=8 -->

Several of these (budget/reset, rigid translation, learned goal, IW(1)) were proposals from a second-opinion review by another AI model ("Argon" in our notes) that we then implemented and measured. <!-- src: mediciones doc, "Arreglos que quedaron" -->

### 4.5 Experiments with the rule engine

#### Protocol

Offline evaluation uses the 25 public games (183 levels) with the local game engine, no network, 2000 actions per game, seeds 0/1/2 per change. The score is our estimate of the official metric (`vivos/puntaje.py`), averaged over games. Each change was kept only if its 3-seed mean improved over the current base. <!-- src: mediciones doc, "Método" -->

#### Public set

| Commit | Change | Levels solved (s0/s1/s2), of 183 | Mean score (3 seeds) |
|---|---|---|---|
| 177918f | clock boxes, per-game time cap | 7 | ~0.30% |
| 7eb2036 | clocks sharing color with board; no no-op loops | 12 | 0.60% (1 seed) |
| 3e849e0 | clocks inherited by color *and* location | 14/14/14 | — |
| cc91bb5 | thin-bar guard | 15/15/15 | — |
| 0b441b9 | rare-type-first clicks | 16/17/18 | 0.685% |
| c0b830b | step budget + deliberate RESET | 16/16/17 | 0.688% |
| fa068f7 | rigid multicolor translation | 17/16/15 | 0.891% |
| 008cbe6 | goal from winning move | 17/16/15 | 0.893% |
| 41238b4 | IW(1) novelty priority | 17/18/15 | **0.942%** |

<!-- src: claude/mediciones-arc-agi-3.md; per-seed files medicion_local_base_s{0,1,2}.txt give 1.008 / 1.164 / 0.656 % -> mean 0.943 % -->

Scores are percentages on the official 0–100 scale. Note how small the per-level contributions are: e.g. in one game the base solves levels 1–2 with [471, 12] actions (seed 0) and [331, 48] (seed 1), yielding game scores of 0.059 and 0.005 respectively. <!-- src: medicion_local_base_s0/s1.txt, game tu93 --> Level counts and score do not move together; the budget/reset change, for instance, barely changed the score but made the first level of one game three times faster. <!-- mediciones doc, c0b830b -->

With the official runner on the same 25 games the agent scored **0.864**, at about 36 s per game and 0 errors. <!-- src: author's run; ADD log/date -->

#### Hidden set

| Submission | Public LB score | Approx. rank |
|---|---|---|
| Living Objects agent (41238b4), 2026-10-14 | 0.07 | ~3868 of ~4300 |
| Same notebook resubmitted, 2026-10-15 | 0.06 | — |

<!-- src: mediciones doc, Kaggle table -->

For reference, on 2026-10-14 the leaderboard leader (Tufa Labs) had 68.91 and 10th place had 40.09; Franzen's Milestone 2 harness reported 27.89. <!-- src: mediciones doc + Franzen writeup; VERIFY 27.89 against his repo --> The two runs differ by 0.01, which we take as run-to-run noise.

#### Failure analysis: overfitting

The public score (0.86–0.94) is roughly 13× the hidden score (0.06–0.07). <!-- arithmetic: 0.864/0.065 ≈ 13 --> Reading the commit log against the table explains why: almost every improvement was diagnosed on one named public game and fixed with a mechanism tuned to it (a clock sharing color with the board in vc33, a respawn in ls20, a direction indicator inside the sprite in tu93, a bar glued to the floor in tn36). Each fix is general in form, but the *set* of fixes reflects the 25 public games. The hidden games presumably contain mechanics for which the rule language has no label at all.

The rule language is the deeper limit. It can express "this action moves this color by (dx, dy)", "this color blocks", "clicking this type changes this patch". It cannot express a goal that is a configuration ("make these pieces fit"), counting, symmetry, or a rule that depends on an internal avatar state. Our victory-differential analysis on level 1 of public games makes this concrete: there is no single predicate across games (one game is won when color 1 goes from 18 cells to 0; others when two colors redistribute cells as pieces move). <!-- src: mediciones doc, "Diferencial de victoria observado" -->

#### Discarded ideas (ablations)

All were measured with 3 seeds against the base of the time and discarded.

| Idea | Outcome |
|---|---|
| Mirror conjecture for "inert" arrows | no improvement over base [ADD numbers] |
| Masking the bar with "partner" colors | no improvement [ADD numbers] |
| Click memory by position/type (3 variants) | worse: buttons depend on state and switches undo themselves. One variant: 16/17/17 levels, 0.545% mean <!-- src: medicion_local_memoria_s*.txt; base of the time not recorded in file --> |
| Composite avatar body via "companion colors" | worse: 13/14/15 levels, 0.683% mean; rigid translation solved the same problem better <!-- src: medicion_local_cuerpo_s*.txt --> |
| Minimal victory differential as goal | identical to base on seeds 0/1, slower on seed 2 <!-- src: medicion_local_dif_s0/s1.txt identical to base; s2 file missing --> |

A second, smaller set of ablations was run on Franzen's harness: grafting our guards (no RESET at step 0, HUD mask, fatal-action veto) on top of it. These were either already covered by his harness or had an effect below noise. <!-- src: mediciones doc -->

**Clock mask vs. border crop.** Franzen's harness decides "did the game change?" by ignoring a 4-cell border. We compared that test against our learned clock mask over 14,186 recorded transitions (`comparar_borde.py`). Only 178 disagreements (1.3%, in 3 games) were genuine cases where the border crop is wrong and our mask is right; the other ~1,266 disagreements were our detector failing to mask 1-cell counters on the bottom row. <!-- src: mediciones doc + task brief; rerun comparar_borde.py to attach the per-game table --> The simple crop is therefore nearly as good as the learned mask for this purpose, and we did not graft the mask.

## 5. Lessons and limitations

**What transferred.** The verification contract (evidence is trusted, beliefs are not; a belief is used only if it explains all its evidence) is cheap, agent-agnostic and directly addresses known LLM failure modes. The engineering lessons also transfer: never RESET at step 0 (the engine restarts the whole game), mask counters before hashing states, prefer rare object types as click candidates, and measure every change with several seeds because single-seed differences are often noise.

**What did not transfer.** Mechanism-by-mechanism fixes derived from public games. Every fix was justified by a real failure, measured, and kept only if it helped, yet the result was a 13× drop from public to hidden. Three seeds on 25 games guard against noise but not against selection on the same 25 games; we had no held-out set.

**The prior-knowledge gap.** Humans bring priors about objects, goals, agency and common game genres; strong LLMs bring a compressed version of the same priors. A colony of rules learned from scratch inside one game has none, and its rule language fixes in advance what it can ever learn. This is, in our view, the main reason the pure engine plateaued near 1% while LLM agents score tens of points.

**Offering is not using.** The most transferable lesson of the hybrid is negative and, we think, general: a verification tool that an LLM agent must choose to call will often not be called, however cheap it is, and making it more prominent or more informative does not change that (Section 3.9). What worked was verifying, automatically, what the model already writes. Measuring use, not only score, is what revealed this.

**Pace bounds the benefit of verification.** In this regime the agent loses score by not reaching later levels in time, not by wasting actions (Section 3.8). Any addition that costs model tokens competes with the dominant term; additions that cost none (engine-supplied rules, Section 3.10) are the promising direction.

**Know exactly what you submitted.** Two of our early hidden-set runs were labelled "base" in our notes but carried an option that changes the prompt; we found out only by reading the submission descriptions. Section 6 describes the protocol we adopted afterwards.

**Engineering lessons from the hybrid.** Two server-configuration changes that looked like free throughput (more parallel streams, more speculative-decoding steps) were measured before submission: the second makes the model server refuse to start, and the first produced fewer actions per minute (550 vs. 872 in the same 20-minute, 14-game scenario). Without the non-submitted measurement runs, either would have cost a daily submission, and the second would have scored close to zero. <!-- src: mediciones doc 2026-10-17 -->

**Limitations of this report.** The offline score is our estimate of the metric, not the server's (but our reproduction of the harness's score matches it exactly on 64 games). The world model of Section 4.3 is measured offline (Section 3.10) but not used in play. Hidden-set conclusions rest on few samples per configuration; the pre-registered comparison is pending. Several ablation rows lack their exact base numbers. Behavioural measurements of the hybrid are single runs on four public games. Leaderboard ranks are approximate.

## 6. Measuring on a budget

The competition allows one hidden-set submission per day, and a submission only returns a score hours later. Non-submitted runs, by contrast, are limited only by a weekly GPU quota. We automated both through GitHub Actions (the development container cannot reach Kaggle): a workflow prepares Franzen's notebook with our options, saves it on Kaggle, optionally submits it, and, for non-submitted runs, stores the full output (transcripts, events, the harness's `benchmark.json` with per-game scores) in a private repository, because transcripts contain game boards. `kaggle/analizar_salida.py` then reports per-game scores, tool use (after removing injected text), automatic-check outcomes and server health. Three lessons came from using it: the default 15-minute "fast save" never reaches the model (the server needs ~12 minutes to start), so it cannot measure behaviour; two server settings that looked like free throughput were harmful and one of them would have produced an empty submission; and a counting script that did not strip our own injected text over-reported tool use by a factor of four. An independent review of the graft by a separate agent, before the first submission of `--vivos-auto`, found two critical defects (a `print` inside the model's simulator broke the sandbox protocol; an infinite loop in it timed out every later snippet), which we fixed and covered with tests.

**Traceability protocol (adopted 2026-10-20).** The launcher downloads Franzen's notebook on every run, so freezing our own commit is not enough to repeat an experiment. Each submission is now identified by a fingerprint of the prepared notebook (SHA-256 of the ordered cell sources, plus a second hash that includes cell types, since turning a code cell into a raw cell keeps the sources but stops execution); the launcher refuses to upload if the origin notebook or the prepared notebook does not match the expected fingerprints. Past submissions were reconstructed from commit and options and then confirmed directly: the Kaggle API returns any past version of a notebook when the version is passed as `version_label="v<N>"` (the CLI of the time sent it inside the slug and got a 403), and the recovered notebooks of all five hidden-set submissions match their reconstructions. The link from a submission to its kernel version is not exposed by the API and is inferred from launch order and fingerprint. Dependencies are recorded per run as observed in the saved run (harness commits, server and model versions); those of the hidden re-run are not observable. A rehearsal of each new procedure without submitting caught one defect in the post-submission recovery step before it mattered. A second-opinion review by another AI model (GPT, "Astra" in our notes) found the mislabelled base runs, the cell-type gap in the fingerprint and a log-parsing bug, and provided the typed-cell fingerprint we use.

## 7. Reproducibility

Code: `Demoliyer/ia-objetos-vivos-arc` (MIT-0), extracted from the private development repository. The rule-engine results correspond to commit `41238b4` of the development repository; the hidden-set configurations are identified by notebook fingerprint: `--vivos` be36c273659a, `--vivos-auto` ff42d7f18a5a, `vivos-v3` 5e0b90762a34, `vivos-v4s` 413a0b289a64, unmodified harness 5c66785823b6 (origin notebook 4056fbfc69bb).

```bash
# environment
python -m venv .venv && .venv/bin/pip install -r requirements.txt -r requirements-arc.txt
# download public games once (needs network), then play offline
.venv/bin/python jugar_arc.py --todos
ETIQUETA=base SEMILLA=0 bash medir_publicos.sh      # repeat with SEMILLA=1, 2
# unit tests
.venv/bin/python -m pytest tests/test_vivos.py tests/test_presupuesto.py tests/test_mundo.py tests/test_simulador.py
# world-model accuracy and clock-mask vs border-crop comparison (recorded trajectories)
.venv/bin/python grabar_trayectorias.py   # VERIFY arguments
.venv/bin/python evaluar_mundo.py
.venv/bin/python comparar_borde.py
# Kaggle: single-file agent for the official starter kit
.venv/bin/python empaquetar_kaggle.py
# Kaggle: Franzen notebook with the verified-simulator graft
python3 kaggle/preparar_franzen.py kaggle/franzen_nb/original.ipynb out.ipynb --vivos
```

<!-- VERIFY each command on a clean checkout before submission -->

## Acknowledgements

Daniel Franzen's Milestone 2 solution (notebook and harness) is the base of the hybrid in Section 3; his harness builds on the Duck harness by Tufa Labs (Jeroen Cottaar). Several mechanisms in Section 4.4, and the shift test, cost-based planner and action brake of Section 3.7, originated as suggestions from a second-opinion AI review (Gemini, "Argon" in our notes) that we then implemented and measured. A second AI reviewer (GPT, "Astra" in our notes) audited the measurements and the traceability protocol (Section 6). Claude (Anthropic) was used for code and writing throughout.

## References

1. ARC Prize Foundation. *ARC Prize 2026 — ARC-AGI-3* (Kaggle competition). https://www.kaggle.com/competitions/arc-prize-2026-arc-agi-3 ; documentation: https://docs.arcprize.org/
2. ARC Prize Foundation. *ARC-AGI-3 Kaggle Starter*. https://github.com/arcprize/ARC-AGI-3-Kaggle-Starter
3. D. Franzen. *ARC-AGI-3 solution (Milestone 2)*. https://github.com/da-fr/arc-agi-3-solution
4. Tufa Labs. *Duck harness*. https://github.com/Tufalabs/duck-harness ; J. Cottaar, *Tufa Labs Duck Harness — June 30 milestone winner* (Kaggle notebook). https://www.kaggle.com/code/jeroencottaar/tufa-labs-duck-harness-june-30-milestone-winner
5. N. Lipovetzky and H. Geffner. *Width and Serialization of Classical Planning Problems*. Proc. ECAI 2012. <!-- VERIFY pages -->
6. H. Tang, D. Key and K. Ellis. *WorldCoder, a Model-Based LLM Agent: Building World Models by Writing Code and Interacting with the Environment*. NeurIPS 2024. arXiv:2402.12275.
7. N. Dainese, M. Merler, M. Alakuijala and P. Marttinen. *Generating Code World Models with Large Language Models Guided by Monte Carlo Tree Search*. NeurIPS 2024. arXiv:2405.15383.
8. P. A. Tsividis, J. Loula, J. Burga, N. Foss, A. Campero, T. Pouncy, S. J. Gershman and J. B. Tenenbaum. *Human-Level Reinforcement Learning through Theory-Based Modeling, Exploration, and Planning*. arXiv:2107.12544, 2021.
