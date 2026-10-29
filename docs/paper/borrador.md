# A Mesh of Living Objects: Rules Learned at Zero Token Cost Beside an LLM Agent, and Why They Must Be Enforced in ARC-AGI-3

Franco González (Demoliyer) · 29 October 2026

## Abstract

We introduce a *mesh of living objects*: many small, human-readable rules about a game, each held to one contract. Observed transitions are trusted evidence, and a rule may guide action only after it explains all of that evidence. The mesh learns on the CPU from transitions the agent has already paid for, so its knowledge costs the language model no tokens. We grafted it onto a competitive public ARC-AGI-3 agent (Daniel Franzen's adaptation of the Tufa Labs Duck harness), whose score turns out to be set by its token budget rather than by unverified beliefs.

Inside the model's Python sandbox the layer adds `check_model(step)`, which replays every recorded transition through a simulator the model writes, and `plan_with_model`, which searches inside it. The mesh offers its own rule through the same interface, as `engine_step`. Other variants added a per-level hypothesis-testing prompt, a ledger of actions blocked by guards, a timed hint and a hard gate on long deliberation.

What we measured, on 12 hidden-set submissions and 37 recorded practice runs (23.7M generated tokens):

1. **The mesh knows what the model does not. LLM-written and mesh-learned rules cover disjoint games** (6% and 7% of levels, no overlap). In ft09 the mesh held a complete rule for 53% of levels, where the model wrote none, with a median of 11 of 20 actions still to play; it learned 514 times over 1,050 snippets in 12.8 s of CPU. We built the hybrid: at practice budgets the mesh offered its rule in 2 of 25 games and the model never used it, and with a stricter confidence criterion it offered none.
2. **Offering is not using.** With the tools only described, the model called them 0 times in about 450 turns. A per-turn reminder and an automatic check after every snippet raised adoption to 2 of 4 games; a richer API placed first in the prompt was adopted in 0 of 5. A hint timed to the moment the model postponed a test it had already written was ignored (cited in 2 of 48 cases); a gate that refuses inspection-only code after 12k tokens without a game action cut the median wait for the next action from 5,730 to 599 tokens, without inducing action loops.
3. **Identical code varies by up to 10 points on the hidden set** (31.52 and 21.38). Twelve submissions of eight configurations fall in 21.4–31.5 (mean 25.65, sd 2.67), so none separates from the others or from the base.
4. **The agent is budget-limited.** Within the same trajectories, the agent cleared 2.12 levels after 64k generated tokens and 4.44 after 177k. Of the runs that ended on an unfinished level and could be classified, 63% were cut before spending what that level usually costs; 13% looked stuck.
5. **A stall detector would mostly interrupt productive work.** 87% of level attempts that exceeded twice the typical cost went on to clear the level.
6. **Little compute is recoverable from the player.** An optimistic bound on avoidable generation in cleared levels is 8.5%, about +0.25 levels per game; 72% of the server's input processing is recomputation of evicted context, which belongs to serving, not to the player.

## 1. Introduction

ARC-AGI-3 asks an agent to play turn-based games it has never seen, with no instructions, and scores it by how few actions it needs compared with humans. The agent must discover both the mechanics and the goal by acting, so every wrong belief about a game costs score. The strongest agents today are LLMs that drive a Python tool: they read the board, write analysis code, act, and repeat.

Our starting point is a contract from an earlier project, Living Mesh: *specifications are trusted, implementations are not*. Moved to games, each transition the game returned is trusted evidence, and each belief about the mechanics is an untrusted implementation. A belief may guide action only after it explains all the evidence recorded for it. We call a small belief that is held to this contract a *living object*: it is written down, checked against what was observed, and repaired or discarded when it fails. Many of them held in memory at once, each scored by its own evidence, form what we call a mesh of living objects (Section 3.5).

We first built an agent made only of such objects: a colony of small, human-readable rules with a learned HUD mask, a state graph and novelty-driven exploration. It improved steadily on the public games (0.94%) and collapsed on the hidden set (0.07). We then asked a narrower question: **what does the same contract add to an LLM agent that is already strong?**

The answer turned out to depend less on the contract than on the agent's budget. This paper reports that path, including the parts that did not work, because the negative results bound what any verification layer can add to this class of agent.

Contributions:

1. A mesh of living objects: an engine that learns verified, human-readable game rules on the CPU at no token cost, compared with the rules the LLM writes on the same play, and a hybrid in which the mesh offers its rule to the LLM (Sections 3.5 and 5.7).
2. A dependency-free layer of verified simulation and planning for the ARC-AGI-3 sandbox, grafted onto a public competitive agent without touching anything else in its official run (Section 3).
3. A measurement of whether the agent uses such tools, separating its own calls from the text we inject, across five designs of increasing pull (Section 5.2).
4. A measurement of the hidden-set noise of identical submissions, and a traceability protocol that identifies exactly which notebook each submission ran (Section 4).
5. Evidence that the agent is budget-limited: a progress curve within its own trajectories, a classification of unfinished levels, a refuted stall detector, and bounds on recoverable compute (Sections 5.4–5.6).
6. A controlled test of suggesting against enforcing on the same player: a well-timed hint and a hard gate aimed at one audited inefficiency, the postponed test (Section 5.8).
7. Audits of the model's own simulators and perception errors on replayed boards, which showed the capacity already present and stopped three components before they were built (Section 5.9).

## 2. Setting and related work

**Task and score.** Games are 64×64 grids with 16 colors. The agent sees frames and a list of valid actions: four directions, a generic action, a click with coordinates, undo, and reset. Each game has several levels. A cleared level scores min(115, (h / a)² × 100), where h is the human baseline action count and a the agent's; an uncleared level scores 0. Levels are weighted by their index, and the game score is capped by the weighted share of levels cleared. Two consequences matter here: late levels dominate the score, and a level that takes many times the human count is worth almost nothing.

**Competition constraints.** The hidden evaluation runs offline in one Kaggle container with one RTX PRO 6000 GPU, on about 110 games. Each team gets one submission per day; a submission's hidden score appears 8–14 hours later. Code we wrote is released under MIT-0.

**The agent we build on.** Daniel Franzen's milestone-2 solution adapts the Tufa Labs Duck harness. It serves Qwen3.8-Flash-Next (W4A16 quantization, speculative decoding with an MTP draft model) on an SGLang fork. All 110 games start at once and a scheduler admits ten at a time, preferring games whose current level looks likely to clear: its priority falls as a level absorbs tokens and actions without completing. The model works through a Python sandbox (standard library only, `python -I -S`) that exposes the current frame, its history, object segmentation and frame diffs, and executes actions with guards that stop stale or repeated actions. The run lasts 532 minutes; Franzen reported 27.89 on the hidden set for his configuration.

**Related work.** Writing a world model as code and checking it against interaction data is not new. WorldCoder has an LLM write a Python transition and reward model, refines it until it explains the observed transitions, and plans in it. Code World Models generate such programs with an LLM guided by tree search over refinements and verify them on offline transitions. Before LLMs, EMPA learned Atari-like games at human efficiency by representing mechanics as a probabilistic program that it tested by exploration. Our rule engine's exploration priority is IW(1) from width-based planning. Our contribution is narrower and empirical: the same contract offered as optional tools to an already strong, prompt-tuned agent under a fixed hidden test set, with a measurement of whether the agent uses them and of the budget regime that decides whether they can help.

## 3. The Living Objects layer

The layer adds five parts to the harness, each a separate candidate we could submit on its own. All of them are patched in at notebook time by `kaggle/injerto_vivos.py`, fail safe (any error leaves the original harness running), and change nothing else in the official run.

*Figure: The Living Objects layer around one LLM turn · letters are the variants of Section 5.1 (diagram in the online version of this paper).*

The model's own turn is untouched: every part either adds a line to what it reads, a function to what it can call, or a process on the host that it never sees.

### 3.1 Verified simulators

The model writes its hypothesis as a top-level `step(grid, action)`, where `grid` is the board as rows of color characters and actions use the harness labels. `step` may return the whole next board or only the cells it predicts to change. Two tools cost no game actions:

- `check_model(step)` replays every recorded transition of the current level (excluding reset, game-over, level-change and undo transitions) and reports how many matched outside the 4-cell border where budget bars live, plus the first mismatches with example cells.
- `plan_with_model(step, goal, actions)` runs breadth-first or best-first search inside the simulator from the current frame, capped at 4,000 states and 8 seconds.

Because the model rarely called these tools on its own (Section 5.2), later variants checked the retained `step` automatically after every snippet and appended one line to the tool output. A one-cell **shift test** warns when a simulator hard-codes positions, which was common: such a simulator explains the level it was written on and fails on the next.

### 3.2 Hypothesis check (variant F)

Once per level, after the first failed attempt or 64 actions without clearing it, the turn message asks the model to separate movement rules from completion conditions, state two hypotheses with the outcomes each predicts, and run a short test that distinguishes them. The text names no game.

### 3.3 Execution ledger (variant G)

The harness's guards stop an action before it runs when the state looks stale or the action is known to do nothing. The model often read such a stop as "this action has no effect". The ledger, built on the host from the harness's own action results and never from the model's text, records each blocked, cut or unknown action with the state it was attempted in. An entry closes only when the same action later executes in the same state (same level, reset count, step and board hash); it is never turned into "no change". Open entries are shown in the turn message, at most five lines.

### 3.4 Startup guard

On days when Kaggle loaded the model weights slowly (up to 47 minutes for 38 shards), all games found no server and ended with zero actions. A guardian game now resets every 300 seconds while the server's health check fails, and the other games wait. When the server starts normally, the guard does nothing.

### 3.5 The mesh of living rules (variant M)

The idea at the centre of this project is a *mesh of living objects*: many small beliefs held in memory at once, each carrying its own record of evidence, competing to predict the next state. No belief is trusted for what it says, only for how often it has been right. The LLM's `step` is one such belief, written by the model; the engine grows the others by itself.

Our engine (`vivos/mundo.py`) keeps three kinds of living rule for the current level:

- a **rigid sprite** that an action moves, with the colors it was seen to collide with and the floor it uncovers;
- a **relative patch** that a click applies to an object of a given type, which captures switches, appearances and shape changes, and carries its own precondition (the cells it expects before acting);
- an **exact memory** of a state already seen with an action.

Each rule is born from a transition it explains. On every later transition, every rule that would have acted predicts first and is then scored: a hit or a miss on its own counter. A rule with more misses than hits loses its say and the next one speaks; a rule that keeps predicting stays. The mesh is never told the mechanics; its content is whatever has survived the evidence so far.

In variant M the mesh lives in the RAM of the harness process, beside the LLM's game session, and learns from the transitions the model has already played in the current level. When, after its third changing transition, the mesh as a whole predicts at least 90% of at least three further changes exactly, before seeing them, it is serialized and handed to the model's sandbox. There a pure-Python copy exposes it as `engine_step(grid, action)`, which the model can pass to `check_model` and `plan_with_model` like its own simulator, and a one-line `[Engine]` note says it exists and how much it predicted. Learning costs no model tokens and no game actions, about 3 ms per transition on the CPU.

The sandbox copy reproduces the NumPy mesh exactly outside the clock mask on samples from 14 recorded trajectories, and the whole path (learning on the host, serialization, prediction in the sandbox, planning) is tested inside a copy of the real harness. Today the mesh is rebuilt from the level's transitions at every model turn and dies at the end of each level; letting its rules live across levels and games is the step we consider next (Section 7).

*Figure: Life of one rule in the mesh · every rule is scored on every transition it would have acted on (diagram in the online version of this paper).*

Nothing in the mesh is deleted for being wrong once: a demoted rule is still checked, and it speaks again if its record recovers.

## 4. Measuring under one hidden submission a day

With one hidden-set submission per day and a 10-point spread between identical runs, the hidden score cannot rank small changes. We therefore measured behaviour on public games and used the hidden set only to check that nothing broke and to estimate its noise.

**Practice runs.** A non-submitted Kaggle run plays chosen public games for a fixed number of minutes and stores the full transcripts, the harness's per-action log (generated tokens and wall-clock time per action) and the model server's log. We reproduce the official score formula with zero error on all 64 games we checked. Early practice runs lost 13–18 of their 45 minutes waiting for the model server; from 24 October the budget starts only after the server answers its health check, so runs on different days get the same playing time.

**Compute per game.** Practice compute is not the official compute. Franzen reports a harness-side generation rate of 609 tokens/s, so the 532-minute official run generates about 19.4M tokens, roughly 177k per game on average (the scheduler spreads it unevenly). A one-game practice gives that game about 300k tokens; a 25-game practice with all games at once gives about 64k. Public-game scores move with this budget: ft09, lp85 and vc33 scored 100 in 5-game practices and 21–58 in 25-game practices.

**Traceability.** Every notebook version gets two fingerprints: a short hash of the code cells and a SHA-256 over typed cells. The launcher refuses to submit unless the prepared notebook matches the expected fingerprint for the named candidate, and after each submission we download the version Kaggle actually ran and compare again. This caught an early submission whose description named the wrong notebook. A health guard also refuses to submit when the model server logs a real failure, while allowing slow weight loads, which the startup guard (Section 3.4) handles.

**Noise of identical code.** Two candidates were submitted twice with the same player code:

| Candidate | Run 1 | Run 2 | Difference |
| --- | --- | --- | --- |
| Verified simulators with automatic check (B) | 29.28 | 25.71 | 3.57 |
| B + hypothesis check (F) | 31.52 | 21.38 | 10.14 |

The F runs differed only in a startup-guard cell that does nothing when the server starts normally, as it did on the 31.52 day. The model samples at temperature 0.7, and in practice runs a single public game moved between 0 and 100 from one run to the next with the same code. Any hidden-set difference below about 5 points between single submissions is noise.

**Pre-registration.** Before seeing results we fixed an interleaved schedule, the comparison (means, no claim of superiority unless the difference clearly exceeds the observed spread) and the criteria each practice comparison had to meet. Where we departed from it (to submit a promising variant earlier), the doc records why.

## 5. Results

### 5.1 Hidden set

No variant separates from the others: twelve valid submissions of eight configurations score 21.38–31.52, mean 25.65, standard deviation 2.67. Franzen reported 27.89 for his unmodified configuration. We did not sample it ourselves: the last daily slots went to repeating the best player instead.

| Date (UTC) | Configuration | Hidden score |
| --- | --- | --- |
| 2026-10-28 | M: F + mesh of living rules offering engine\_step | 24.08 |
| 2026-10-27 | G: F + execution ledger (our implementation) | 25.42 |
| 2026-10-26 | G (implementation by a second assistant) | 24.04 |
| 2026-10-25 | FB: B + hypothesis check | 24.12 |
| 2026-10-24 | F: A + hypothesis check, run 2 | 21.38 |
| 2026-10-23 | F, run 1 | 31.52 |
| 2026-10-21 | A: B + shift test, cost-based planner, reports by action family | 23.75 |
| 2026-10-20 | B, run 2 | 25.71 |
| 2026-10-19 | v3: B + action brake and level notices | 26.70 |
| 2026-10-18 | B: simulators + reminder + automatic check, run 1 | 29.28 |
| 2026-10-17 | Tools only described in the prompt, run 2 | 25.38 |
| 2026-10-16 | Tools only described in the prompt, run 1 | 26.39 |

A submission on 22 October scored 0.00 because the model server never started (a slow-loading day on Kaggle); it is excluded and led to the startup guard. Our pure rule engine, before any of this, scored 0.07.

**Context.** The highest-scoring public notebook on 28 October is identical to Franzen's in its official run; only its practice settings differ. Public copies of his configuration therefore spread over the same range as our variants, and a single score ranks luck as much as method. With 31.52 the team stood 411th of 4,944 on the public leaderboard on 28 October.

### 5.2 Does the model use what it is offered?

We ran four public games for 45 minutes per condition and counted the model's own simulators and tool calls in its transcripts, after removing the text we inject. An earlier count that skipped this step reported 4 of 4 games when the true figure was 1 of 4.

| Condition | Requests | Games where the model wrote a `step` | Model's `check_model` calls | Plans |
| --- | --- | --- | --- | --- |
| Tools described in the prompt | 457 | 0 / 4 | 0 | 0 |
| + one-line reminder per turn | 458 | 1 / 4 | 8 | 2 |
| + partial `step`, automatic check after every snippet | 474 | 2 / 4 | 5 (+148 automatic) | 4 |

The model ran more than 20,000 Python snippets of its own analysis across these runs: it builds local helpers, not whole-board simulators. Two richer interfaces confirmed the pattern. A state model (`observe`, `advance`, checked field by field) placed first in the instructions was adopted in 0 of 5 games. Error messages that translated a simulator's wrong cells into objects ("the red object moved by (0, +1); your step predicts no change") were delivered 44 times without the model acting on one.

The hypothesis check of variant F was followed sometimes. In ls20 the model once stated two hypotheses (shape and color must both match, or only color) and ran the test that separates them; in other runs it stated a hypothesis without a new test, or kept its plan. The execution ledger of G fired in 9 of 25 games in a 25-game practice run (lf52 15 times, m0r0 11, ar25 and r11l 10), for example reporting that a SPACE stopped by the stale-state guard had not been tested.

When the model did commit to a verified simulator, it cleared levels with fewer actions but more tokens: in tu93, levels 2–4 took 15, 19 and 22 actions against 21, 28 and 42 for the base, at 1.5–3× the generated tokens per level, and the run ended five levels in against eight for the base.

### 5.3 Practice comparisons

Single-game comparisons in ls20 did not separate F from A: A won two of three comparisons on levels cleared (5/7 against 4/7 twice), F one on actions (19% fewer for the same levels). FB scored 47.01 in one ls20 run and 24.12 on the hidden set.

In a 25-game practice with the same budget for both, F and the second G scored 14.79 and 13.73: 49 against 46 levels, a paired difference of +1.06 points with a standard error of 5.1. The same game moved from 0 to 100 (cd82) and from 95 to 28 (sb26) between the two runs. In sb26 the ledger never fired, so both runs played identical code there: that 67-point gap is sampling noise.

### 5.4 The agent is budget-limited

We measured, inside each recorded trajectory, how many levels had been cleared by the time the run had generated a given number of tokens, using only runs that reached the largest budget in each row, so no run is cut short before a point it is counted at.

| Runs | 64k tokens | 100k | 177k | 250k |
| --- | --- | --- | --- | --- |
| 50 runs reaching ≥ 177k (ls20 21, tu93 13, vc33 11, ft09 3, cd82 1, lp85 1) | 2.12 | 2.84 | 4.44 | — |
| 16 runs reaching ≥ 250k (ls20 8, tu93 5, vc33 3) | 1.81 | 2.56 | 4.06 | 5.38 |

Progress is close to linear in tokens over this range, about two levels per 100k tokens between 100k and 177k. The curve comes from six public games and one harness; it is not a prediction for the hidden games or for another scheduler.

We then classified the last, unfinished level of the 113 runs that did not win, comparing the tokens spent on it with the median cost of the same level when cleared in other runs:

| Last level | Runs | Share of classifiable |
| --- | --- | --- |
| Cut before reaching the level's typical cost | 50 | 63% |
| Between 1× and 2× the typical cost | 20 | 25% |
| Over 2× the typical cost (possibly stuck) | 10 | 13% |
| Never cleared in any run (no reference) | 33 | — |

In an earlier sample of 64 runs, all 37 that did not win were stopped by the clock, none by a game over, and the median time on the last level (5.0 min) was no longer than the median time per cleared level (4.8 min).

### 5.5 A stall detector would interrupt productive work

A natural next variant intervenes when a level absorbs far more tokens than usual. We back-tested that detector on all recorded level attempts, with the typical cost taken from the other runs. The criterion, fixed before the test, required at least 70% of real stalls detected and fewer than 15% false alarms.

| Fires when the attempt exceeds | Fires | Attempts that cleared the level anyway | Real stalls with budget left to recover |
| --- | --- | --- | --- |
| 1.5× typical cost | 156 | 88% | 7 |
| 2× | 78 | 87% | 4 |
| 3× | 22 | 82% | 3 |

Long attempts usually end in a cleared level. Franzen's scheduler already lowers the priority of levels that absorb tokens without completing, calibrated on 725 level attempts, so the official run reallocates this compute at the level of games rather than interrupting the model's reasoning.

### 5.6 How much compute could the player recover?

We estimated generated tokens per model request from its generated characters, calibrated per run against the harness's token count, and classified each request in 148 transcribed runs (20.7M tokens).

| Spend in levels that were cleared (16.4M tokens) | Share |
| --- | --- |
| Snippet raised an error and executed no action | 8.5% |
| Reply without a tool call | 2.5% |
| Executed actions with no gameplay change | 0.4% |
| Only action blocked by a guard | 0.0% |
| Code identical to an earlier snippet at the same step | 0.0% |

Errors are ordinary programming errors spread across types (NameError 18%, the harness's own stale-state stop 16%, TypeError 15%, KeyError 12%, a blocked import 11%). Even if every erroring request were wasted, which overstates it, removing them would free about 12k tokens per game at the official budget, about +0.25 levels per game on the curve above. The part a harness change could plausibly remove (blocked imports) is 1–2% of tokens.

On the input side, the model server processed 4.50M new input tokens against 1.59M generated in a 25-game run. Only about 1.27M of that input was new text added by turns (Franzen's turn prompt 36%, the board image 35%, tool output 16%, our own notes 8%, retained-function list 5%); the other 72% was recomputation of context evicted from a key-value cache running at 99% occupancy. Trimming our own notes would save about 2% of input processing.

### 5.7 Who should write the rule?

On the same recorded play (12,307 transitions), we measured when each source has a complete rule for a level. For the LLM, its `step` explains every recorded transition of the level (in-sample). For the engine, after its third changing transition it predicts at least 90% of the following ones exactly before seeing them (out-of-sample, which disfavours it).

| Game | LLM wrote any `step` | LLM complete rule | Engine complete rule |
| --- | --- | --- | --- |
| ft09 | 0% | 0% | 53% |
| tu93 | 81% | 15% | 0% |
| lp85 | 41% | 11% | 0% |
| ls20 | 17% | 0% | 0% |
| vc33, ar25, cd82, sb26 | 0–2% | 0–2% | 0% |
| All levels | 28% | 6% | 7% |

The engine succeeds exactly where the LLM never writes a simulator, and no game is covered by both. In ft09 the engine had its rule with a median of 11 of the level's 20 actions still to play.

Variant M (Section 3.5) put this to work, with a negative result. In a first 25-game practice the mesh offered its rule in 2 games (ft09, tn36) and the model never called `engine_step`. A review by a second assistant found that hits from exact memory inflated the confidence criterion: counting only state-action pairs not seen before left ft09 unchanged (36 levels with a complete rule) and removed all of tu93's (4 to 0). With that criterion, an automatic check of `engine_step` after every snippet, and failure counters, a second practice learned 514 times over 1,050 snippets at 12.8 s of host CPU in total, with zero failures, and offered a rule zero times: at about 64k tokens per game, levels end before the mesh sees enough new transitions. M was therefore not submitted.

Both M runs scored 18.59 and 18.50 against 14.79 for F. In the second run nothing from the mesh reached the model, so it played as F: two runs of the same player over 25 games differed by 3.7 points of mean score (paired standard error 5.0), the practice-level counterpart of the hidden-set noise in Section 4.

### 5.8 Suggesting versus enforcing

The model acts on what the harness enforces, not on what it suggests. An audit of 60 reasoning intervals (40 over 8k tokens, 20 controls, labelled by us and re-labelled blind by a second reviewer on a subsample of 15) found one robust inefficiency: the model writes down a cheap test that would separate its hypotheses, then postpones it while it keeps theorising. All 6 cases both reviewers agreed on were in long intervals, none in the controls, and in 3 of them the model itself wrote that probing was now warranted.

We attacked it twice, each time as a single change to the F player, in the same five public games and 45 minutes per game as the F reference run:

- **Hint.** After 4,096 tokens without a game action, the next turn message suggests running that test now. It is shown once per stall and re-arms at the next action.
- **Gate.** After 12,288 tokens without a game action, Python snippets that execute no game action are refused with an explanation. Simulator checks, planning calls and retained helpers that act are exempt; it re-arms at every executed action. The threshold came from replaying the gate over 3,649 recorded intervals: at 8,192 most refusals fell on productive planning.

| Measure | F (reference) | Hint | Gate |
| --- | --- | --- | --- |
| Tokens from crossing 12k to the next action, median | 5,730 | 2,761 | 599 |
| Share of tokens in intervals over 8k | 54.6% | 56.8% | 43.6% |
| Turns with a single action | 45% | 43% | 47% |
| Tokens per cleared level | 25.6k | 29.8k | 30.1k |
| Practice score, 5 games | 80.6 | 82.2 | 71.5 |

The hint fired 48 times and the reasoning that followed cited it twice. The gate fired 10 times and every time the next snippet acted: 7 executed a plan already built or a decisive test, 1 was a no-op, and 2 acted before a search had finished. Enforcing changed behaviour at once; in this single run it did not make levels cheaper, and the score difference is inside the spread of single practice runs (Section 5.3).

### 5.9 Audit before building

Three audits on recorded play replaced three proposed components, each by showing that the capacity was already there.

- **The model's own simulators.** Replaying every version of `step` the model wrote in 113 practice games, the current version predicted 72% of the following transitions in the same level (95% CI 60–82; 86% on 100 further games), against 5% for a do-nothing baseline. In 195 transitions where several versions existed, an older one never got right what the current one got wrong, so an archive of past simulators adds nothing.
- **Perception errors.** The 7 intervals labelled as perception errors, replayed on the real boards, were indexing bugs in the model's own crop code (6) and one misattributed move: three key presses had moved a wall, and the model compared against an outdated segmentation. The harness's segmentation and `frame_diff` resolved each case when called. A dedicated coordinate verifier would have endorsed the model's false "9-column offset" in the last one.
- **Automatic diff summaries.** The harness's own text diff, off in the reference, reaches 1 of those 7 cases. It costs about 97 input tokens per turn at its default budget and degrades to bare counts in 37% of turns at a 100-token budget.

## 6. Theory: verification in a budget-limited regime

A layer added to this agent raises the score only if it raises levels cleared per generated token by more than the tokens it consumes. Let T be the tokens a game receives and ρ the levels cleared per token, so L ≈ ρT over the range we measured (Section 5.4). A layer that makes the model e times more productive per token while taking a fraction c of its tokens yields:

```latex
L' = \rho (1 + e)(1 - c)\, T, \qquad L' > L \iff e > \frac{c}{1 - c}
```

**Why verified simulators did not pay.** Their benefit lands on the small term of the score. Extra actions on cleared levels cost 2.7 points per game on average; levels never reached cost 27.5. A simulator the model writes and debugs moves the cost to the large term: where it was used, it multiplied tokens per level by 1.5–3 (c large) to save actions (e near zero in levels per token). Verification by the model is self-defeating under this budget.

**Why offering is not using.** The model adopts what continues its existing habits. It already writes local Python helpers, so a partial `step` that may return only the changed cells, checked without being asked, was adopted in 2 of 4 games. Interfaces that require a new commitment (a whole-board simulator, a state model, a new API placed first) were adopted in 0. Text that explains an error is not acted on unless it fits a step the model was about to take. A layer should therefore act on the model's own artefacts and ask nothing new of it.

**Why enforcing works where suggesting fails.** A suggestion asks the model to change a habit; a gate changes what the habit can return. Section 5.8 tests this directly: a hint timed to the postponed test was ignored, and a refusal of inspection-only code moved the next action ten times closer. For a model that already holds the knowledge, a layer's lever is what it enforces, not what it explains.

**Where a zero-cost rule fits.** A rule learned by the engine has c ≈ 0 in model tokens: it is learned on the CPU from transitions the model already paid for. Any e > 0 then helps, and the engine succeeds exactly where the model never writes a simulator (ft09: 53% of levels against 0%). The open question is adoption, which the theory above predicts to be the bottleneck. Variant M tests it.

**What the hidden set can detect.** With a per-submission standard deviation σ ≈ 2.9, detecting a true difference Δ between two configurations at 5% significance and 80% power takes about

```latex
n \approx \frac{2 (z_{0.975} + z_{0.8})^2 \sigma^2}{\Delta^2}
```

submissions per configuration: about 33 for Δ = 2 points and 5 for Δ = 5. With one submission a day, only changes of about 5 points or more can be confirmed within a competition; smaller ones must be judged by mechanism, measured in behaviour.

## 7. Limitations, negative results and future work

**Limitations.** Most behavioural results come from one run per condition on a few public games, where the agent scores far higher than on the hidden set. The budget curve and the stall back-test use six public games with up to 21 runs each, and reference costs from other runs of the same games. Token counts per request are estimated from characters, calibrated per run. The hidden-set comparison has one or two samples per configuration and a spread of 10 points between identical runs, so it supports no ranking.

Section 5.8 rests on one practice run per condition. The gate shortened stalls but did not lower tokens per level in that run. In tu93, where a test action can end the game, the gate run had more game overs (8, against 3 for F and 6 for the hint), which one run cannot attribute.

**Negative results.**

- The pure Living Objects agent, a colony of verified rules with a learned HUD mask, a state graph and IW(1) novelty, reached 0.94% on the public games and 0.07 on the hidden set. Its mechanisms were tuned to the public games and its rule language cannot express most goals.
- An action brake ("stop modelling, act") fired on a model whose simulator already explained every transition.
- Cutting a batch of actions at the first divergence from a verified simulator could have saved at most 74 actions over 13 practice runs (< 0.5%): a verified simulator was present in 24 of 924 multi-action turns.
- A token cap per decision was not tried after measurement: the hard output cap touched 0.4% of requests, the soft cap was already ignored, and one 45k-token interval built a simulator that reproduced 62 of 62 transitions and a verified 42-action plan.
- A stall detector, a lighter prompt and dropping the board image were rejected on data before spending a run (Sections 5.5 and 5.6).

**An action arbiter, measured in shadow mode.** A natural next design puts the mesh on the execution path and lets it block actions it can show to be useless. We replayed 23,060 recorded actions, with the mesh learning online and predicting each action's effect before it ran. 9.4% of actions changed nothing and 0.53% ended in a game over. Blocking only exact repeats of a state and action already seen (a guard Franzen's harness offers but leaves off by default, having found no clear benefit) would have stopped 1.9% of actions and been wrong 10% of the time, because the state hides counters and phases. Blocking where the mesh predicts no change in a new state would have been wrong 87% of the time: without a rule, the mesh predicts nothing happens, which is not evidence. Even a perfect arbiter would save actions, the small term of the score, not tokens. A verifier of the model's reasoning has a similar ceiling: of the 660 intervals above 8k generated tokens (52% of all tokens), only 35, holding 3.0% of all tokens, began while the mesh had a complete rule for the level, and 102 (8.8%) while it held at least one partial rule with three or more hits and no more misses than hits.

**Future work.** The engine's rule language is the lever: it has no conditional object behaviour, counters or pushing, and learns a complete rule in 3 of 8 public games. A richer language learned on the CPU would keep the token cost at zero while covering more games. The other lever is serving: 72% of input processing is recomputation of evicted context, so cache capacity and eviction policy are worth more compute than anything we could remove from the player. Finally, a living object today lives for one level. We measured what carrying the mesh further would give on 651 recorded levels: keeping the rules alive across levels of the same game raises per-transition accuracy from 28% to 31% but adds no level with a complete rule (83 in both cases); starting from another game's mesh adds nothing more; and carrying the clock detector across levels cuts complete rules from 83 to 30, because its mask accumulates. A second direction reverses the flow: the LLM already discovers mechanics the engine cannot express, so its own simulators could join the mesh as living objects, each scored by action family and kept while the evidence holds. Memory across games needs rules stated independently of color and shape ("the action moves the controlled object one cell"), which is the next step for the project.

## 8. Conclusion

The mesh of living objects learns, at no token cost, rules the model never writes: in ft09 it held a complete rule for 53% of levels where the model held none. Around a strong agent whose score is set by its token budget, that is exactly the kind of knowledge a layer can add without taking from the budget. It did not raise the hidden score (twelve submissions of eight configurations, 25.65 ± 2.67, best 31.52), because the model never took the rule it was offered.

The controlled test of Section 5.8 shows why, and what to build next. The model ignores a well-timed suggestion and changes its behaviour at once when the harness enforces a rule. The next mesh should therefore not offer its rules but apply them: when a rule explains every transition of the level, the harness plans and executes with it, and the model keeps the levels the mesh cannot model. That keeps the token cost at zero and removes the adoption bottleneck this paper measured.

## 9. Reproducibility, licence and acknowledgements

**Code.** Our code is released under MIT-0 at [github.com/Demoliyer/ia-objetos-vivos-arc](https://github.com/Demoliyer/ia-objetos-vivos-arc), one branch per submitted configuration with its notebook fingerprint: `motor` (variant M, fingerprint 97652a37e9be), `f2b` (F, de260c82176f), `v4f` (F with startup guard, 4b78f4916fb9), `v4s` (A) and `main` (B and the tools-only runs). The layer lives in `vivos/` (`simulador.py` for the tools, `mundo.py` and `percepcion.py` for the engine, `motor_llm.py` and `motor_puro.py` for variant M) and `kaggle/` (`injerto_vivos.py` patches the harness, `preparar_franzen.py` prepares the notebook, `experimento.sh` launches a run and refuses to submit an unexpected fingerprint). The Kaggle code submission linked to this paper is variant M: notebook `demoliyer22/arc-agi-3-vivos`, version 61, submission 57718028 (28 October 2026, hidden score 24.08). The analyses in Sections 5.4–5.6 are reproducible from the stored practice outputs with three short scripts.

**Each submission is reproducible by fingerprint.** The table in Section 5.1 maps to notebook versions whose code-cell hashes were checked before and after submission.

**Acknowledgements.** This work builds directly on Daniel Franzen's milestone-2 solution and on the Duck harness by Tufa Labs; all score above 0.07 comes from their agent. We thank the ARC Prize team for the competition.

**AI assistance.** Claude (Anthropic) wrote most of the code and drafted this text under the author's direction; a second assistant reviewed designs and wrote one of the two ledger implementations. The author chose what to build, what to submit and what to report.

## Sources

- [ARC Prize 2026 — competition overview and rules](https://arcprize.org/competitions/2026)
- [ARC Prize 2026 — Paper Prize](https://arcprize.org/competitions/2026/paper)
- [ARC Prize 2026 — ARC-AGI-3 track](https://arcprize.org/competitions/2026/arc-agi-3)
- Daniel Franzen, [ARC-AGI-3 milestone-2 solution](https://github.com/da-fr/arc-agi-3-solution): write-up and configuration guide (repository `ARC3-Inference`, `WRITEUP.md` and `CONFIGURATION.md`)
- Tang, Key and Ellis, [*WorldCoder, a Model-Based LLM Agent*](https://arxiv.org/abs/2402.12275) (NeurIPS 2024); Dainese et al., [*Generating Code World Models with Large Language Models Guided by Monte Carlo Tree Search*](https://arxiv.org/abs/2405.15383) (NeurIPS 2024); Tsividis et al., [*Human-Level Reinforcement Learning through Theory-Based Modeling, Exploration, and Planning*](https://arxiv.org/abs/2107.12544) (2021); Lipovetzky and Geffner, [*Width and Serialization of Classical Planning Problems*](https://repositori.upf.edu/items/890ae8cc-3f6e-4209-bcd1-2ad5fa8f6c82) (ECAI 2012)
