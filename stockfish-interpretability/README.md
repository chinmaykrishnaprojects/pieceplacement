# Stockfish on the strength–cost–interpretability Pareto frontier

An empirical study of where chess engines and chess LLMs sit on the trade-off
surface between **playing strength (Elo)**, **compute cost per move**, and
**interpretability to humans** — starting from Stockfish and measuring, rather
than assuming, what interpretability costs.

Everything here runs from scratch on a 4-core box: Stockfish 16 (NNUE),
python-chess, and 12,000 lichess games that carry lichess's server analysis
(Stockfish at depth ~18–22), extracted by streaming the January 2024 open
database dump. **No files outside this folder are read or touched.**

> Stockfish 16 is used rather than the latest release because this
> environment's network policy blocks GitHub release downloads; SF16 is fully
> NNUE-based, so all conclusions about the NNUE era apply.

## Contents

| File | What it does |
|---|---|
| `scripts/extract_lichess.py` | Streams the lichess dump, keeps games with `[%eval]`, writes a per-position dataset (material counts, depth-18 eval, result) |
| `scripts/analyze_positions.py` | Eval→score curve, reverse-engineered piece values, difference-vs-ratio material models, trade-when-ahead tests |
| `scripts/node_ladder.py` | Self-play ladder: SF16 at fixed node budgets (32→65,536) plus a material-only αβ engine; measures Elo per node |
| `scripts/multipv_cost.py` | Measures the Elo price of interpretability: SF forced to resolve k candidate lines (MultiPV=k) vs normal SF at the same node budget |
| `scripts/explain_move.py` | **Prototype interpretable Stockfish**: emits Gemini-prompt-style JSON (candidate lines in SAN, win-probability via the fitted curve, rule-based English rationale) at ~10 ms/move |
| `scripts/extract_fens.py` | Samples quiet positions with FEN + the human move actually played (for human-likeness + eval decomposition) |
| `scripts/ruleset_ladder.py` | Places two interpretable no-/low-search engines (a page-of-principles rule-set αβ, and SF's NNUE as pure depth-1 policy) on the node ladder |
| `scripts/humanlike.py` | How often SF (per node budget) plays the human move; by rating band; human centipawn-loss |
| `scripts/eval_decompose.py` | Regresses SF16's opaque NNUE eval onto its own classical human-readable terms (R²≈0.74) |
| `scripts/coach.py` | Human coach layer: diffs SF16's classical eval terms before/after the engine's move to explain *why* in plain language |
| `scripts/sprt.py` | Fishtest-style GSPRT match harness (pentanomial, log-likelihood bounds) |
| `src/` (built) | Stockfish 16 from Ubuntu source; `sf-explain` = `Explain` UCI patch (`results/explain.patch`), bench-identical; `sf-nonull` = a known-regression control |
| `scripts/make_charts.py` | Renders all charts in `results/` |
| `data/positions.csv.gz`, `data/fen_moves.csv.gz` | 783k material/eval positions; 38k FEN + human-move records (2024-01 dump) |
| `results/` | Fitted parameters (JSON), match results (CSV), charts (PNG), `explain.patch` |
| `PONYTAIL_NOTE.md` | The ponytail YAGNI ladder (fetched from the web) and how it was applied here |

## Findings

### 1. What an eval is actually worth (`chart_eval_score.png`)

Fitting *actual game results* against lichess's depth-18 evals over 115k quiet
positions (checks, captures and opening plies excluded; positions subsampled
within games to reduce correlation):

- Expected score ≈ σ(cp / **468**): +1.00 gives only a **55%** expected score
  for these (mostly 1200–2200) players.
- **The "+1.00 ⇒ 50% chance of winning" rule is strength-dependent.** For this
  population, 50% win probability arrives at **+0.63**, not +1.00 — and the
  curve sharpens with rating: k ≈ 577cp below 1600, 413cp at 1600–2000,
  **332cp above 2000**. Stronger players convert the same eval more reliably;
  Stockfish's own WDL normalization (+1.00 ↔ 50% win) is calibrated to
  engine-level play, which continues this trend.

### 2. Reverse-engineered piece values (`chart_piece_values.png`)

Two independent regressions on the same positions:

| Piece | Classical | SF depth-18 eval–implied | Outcome-implied |
|---|---|---|---|
| Pawn | 1 | 1.00 (= 135cp) | 1.00 |
| Knight | 3 | 2.77 | 2.18 |
| Bishop | 3 | 3.07 | 2.59 |
| Rook | 5 | 4.21 | 3.69 |
| Queen | 9 | 7.26 | 6.89 |

- SF's NNUE at depth 18 prices a pawn at ~135cp of normalized eval and
  compresses the heavy pieces relative to 1/3/3/5/9 (Q ≈ 7.3, R ≈ 4.2), while
  confirming the bishop's edge over the knight (3.07 vs 2.77).
- The *outcome*-implied values (what material actually converts to for
  amateurs) are compressed further: an amateur's extra queen is worth ~6.9
  pawns of win-probability, not 9. Big material is systematically harder to
  convert than its eval suggests.

### 3. Difference vs ratio, and "when ahead, trade" (`chart_conversion.png`)

Log-loss of logistic models predicting decisive-game outcomes from material:

| Model | Log-loss |
|---|---|
| material difference | 0.6327 |
| log material ratio | 0.6294 |
| **difference / √(total material)** | **0.6276** |

The user-hypothesized ratio-flavored model beats the raw difference, and the
best simple form is the difference *scaled up as material comes off*. The
effect is big and monotone: **the same +2 to +4 edge converts at 62.6% with
65–80 points of material on the board and 80.5% with ≤20 remaining** — folk
wisdom ("when up, trade pieces") quantified at ~18 percentage points of
expected score. Complementarily, a leader whose remaining material is
pawn-heavy converts best (68% → 79% across pawn-share quartiles), the mirror
image of "when down, trade pawns".

Nuance: for predicting *SF's own eval*, the plain per-piece difference is the
best material model (R² = 0.61 vs 0.47 for the ratio) — SF's eval is close to
linear in material. The ratio/scaling structure lives in *conversion*, i.e.
between eval and result. An interpretable "human" eval should therefore be
two-stage: a linear material+positional score, then a conversion curve that
sharpens as material comes off.

### 4. Strength per node (`chart_ladder.png`, `results/ladder_elo.json`)

Self-play ladder at fixed nodes/move (60 games per adjacent pair, balanced
random openings, 1 thread). SF16 refuses to search fewer than ~32 nodes (it
always completes a depth-2 iteration), so 32 is the floor — even "1-node"
Stockfish is already a search engine. 65,536 nodes ≈ depth 17, conveniently
close to the depth-18 evals used above. See `results/ladder_elo.json` for
measured relative Elo; the chart anchors 65k nodes ≈ 3100 Elo (stated
assumption, not a measurement).

Measured curve (anchored): 32 nodes ≈ 1468, 64 ≈ 1520, 256 ≈ 1561,
1024 ≈ 1927, 4096 ≈ 2439, 16 384 ≈ 2820, 65 536 ≡ 3100. The curve is far
from log-linear: the first doublings above 32 nodes are nearly free
(~+90 Elo total from 32 → 256), then the middle regime pays ~300–500 Elo
per 4× (256 → 1024: +366; 1024 → 4096: +512). Intelligence-per-node is not
constant — the eval alone (depth-2 SF) already plays ~1450-level chess, and
search buys the rest.

A material-only alpha-beta (depth 2 + capture quiescence — the maximally
interpretable classical engine) lands at ≈ **1009** on the same scale via
direct matches against the 32- and 64-node rungs: NNUE's opaque eval is
worth ~460 Elo over transparent material counting at equal (tiny) search.

### 5. The measured price of interpretability (`results/multipv_results.csv`)

The concrete "interpretable Stockfish" of this study: force the engine to
resolve **k candidate lines** (MultiPV=k) from the same node budget and play
the top one — exactly the information a human-readable move card needs. This
is measured head-to-head vs normal SF at 16,384 nodes (40 games per k):

| Candidate lines (MultiPV) | Score vs normal SF | Elo cost |
|---|---|---|
| 2 | 36.2% | **−98** |
| 3 | 22.5% | **−215** |
| 5 | 18.8% | **−254** |

The measured ladder slope at this budget is ~+280 Elo per 4× nodes, so a
full 3-line explanation is bought back with roughly **3× nodes** — i.e.
**interpretability has a measurable, purchasable price in compute**, about
1.5 doublings for k=3. (See `chart_multipv.png`.)

`scripts/explain_move.py` packages this: give it a FEN or PGN and it returns
the same JSON contract as the Gemini grandmaster prompt (candidate variations
in SAN, expected score from the fitted curve of §1, a rule-based English
rationale from legible features) — at ~10 ms of CPU (~$3×10⁻⁷) per move
instead of cents, with *faithful* explanations (the numbers shown are the
numbers that chose the move, unlike an LLM's post-hoc rationale).

### 6. The frontier (`chart_pareto.png`, `results/pareto_points.json`)

Measured points (this study) are joined by literature-estimate reference
points — general LLMs (dubesor.de-style prompting, ~1400–1800), a 50M
chess-GPT (Karvonen-style, ~1500, the user's app), Maia (human-move
prediction, policy-only), Leela with and without search. Interpretability is
scored 0–10 on an explicit rubric (native English rationale, legible
intermediate quantities, mechanistic transparency, faithfulness) in the
report. The shape that emerges:

- **Cost spans ~8 orders of magnitude for ~2000 Elo of range.**
- The interpretable end is occupied twice, in opposite corners: material-only
  αβ (cheap, weak, fully transparent) and frontier LLMs (expensive, weak-ish,
  fluent but unfaithful explanations).
- Search engines dominate on strength-per-dollar; their interpretability
  problem is *volume* (millions of legible nodes), not opacity per node —
  which is why MultiPV-style summarization is cheap, while LLM opacity is
  structural.

## Reproducing

```bash
python3 -m venv venv && venv/bin/pip install chess zstandard numpy scipy pandas matplotlib scikit-learn
venv/bin/python scripts/extract_lichess.py 12000 data/positions.csv.gz
venv/bin/python scripts/analyze_positions.py
venv/bin/python scripts/node_ladder.py 60 results/ladder_results.csv
venv/bin/python scripts/multipv_cost.py 40 results/multipv_results.csv
venv/bin/python scripts/make_charts.py
venv/bin/python scripts/explain_move.py "<FEN>" 16384
```

## Relation to fishtest

Stockfish improvements are accepted via [fishtest](https://tests.stockfishchess.org)
SPRT: a patch must prove a non-negative Elo effect over tens of thousands of
games at two time controls. The framing here is deliberately different:
fishtest optimizes a *single point* (max Elo at fixed cost); this study maps
the *frontier*, so that an interpretability patch can be judged as "−x Elo
for +y explanation quality at −z nodes" rather than rejected outright. The
MultiPV experiment is exactly a fishtest-shaped question ("what is the Elo
cost of always resolving 3 lines?") answered at small scale.

## Honest limitations

- 60 games per ladder rung ⇒ ±~45 Elo (1σ) per edge; chained edges accumulate error.
- Absolute Elo anchoring is an assumption (relative curve is measured).
- Lichess evals are SF depth ~18–22 *of the era of the game*; treated as one oracle.
- Outcome-implied piece values inherit lichess players' conversion skill; they
  are "value of material *to a ~1800 human*", which is the point, but not a
  universal constant.
- Reference points for LLM/Maia/Leela strength and cost are literature
  estimates, marked as such.

## Part 2: building Stockfish from source + the fishtest loop

GitHub is blocked in this environment, but Ubuntu's archive isn't:
`apt-get source stockfish` yields the full Stockfish 16 C++ tree (NNUE net
included). `make profile-build ARCH=x86-64-avx2` reproduces the official
**bench signature 2593605** — the same signature fishtest uses to verify a
submitted patch compiles to the intended search.

### The C++ patch: `results/explain.patch` (UCI option `Explain`)

MultiPV buys candidate lines by *splitting the search* (−215 Elo for 3 lines,
measured above). But alpha-beta already spends real effort on refuted root
moves — it just throws that information away. The patch adds per-root-move
**effort accounting** (subtree nodes attributed to each root candidate, ~10
lines in `search.cpp`/`search.h`) and prints an MCTS-style distribution at
bestmove time:

```
info string candidate e1b1 effort 76% score cp 43
info string candidate f3h5 effort 15% score cp -32001→(hidden when unknown)
bestmove e1b1
```

In forced positions the distribution collapses (99% on one move — the human
"only move" signal); in rich positions it spreads over 3–5 candidates. This
is the user-hypothesized "search with a probability distribution over
candidate moves, like a human", extracted from stock alpha-beta.

**Cost: provably zero.** The patched binary's bench signature is identical
(2593605) — in fishtest terms a verified no-functional-change patch. The
interpretability was free all along; MultiPV was the wrong price to pay.

### SPRT harness: `scripts/sprt.py`

Implements the fishtest acceptance test at small scale: game pairs with
color-swapped balanced openings, pentanomial pair statistics, GSPRT
log-likelihood ratio for H0: elo=elo0 vs H1: elo=elo1, stopping bounds
±log(19) (α=β=0.05). Two demonstration runs (8192 nodes/move):

- **Known regression** (null-move pruning disabled; bench 3284598):
  SPRT [0, 5] — see `results/sprt_nonull.csv`.
- **Explain patch** vs base: SPRT [−10, 0] non-regression bounds — see
  `results/sprt_explain.csv`.

A real Stockfish 19 submission would run the same loop at fishtest scale
(tens of thousands of games, STC then LTC, elo bounds like [0, 2]); the
machinery here is the same shape, scaled to one 4-core box.

## Part 3: making the eval genuinely useful to a human

### The scope of "modify Stockfish": version archaeology

`apt-get source` gives SF16, and that turns out to be the **ideal** version for
interpretability work — not a limitation. Stockfish removed its classical,
hand-crafted evaluation in PR #4674 (right after SF16), for a whole-engine
strength cost of only ~2 Elo. For the *engine* the hand-crafted terms were dead
weight; for a *human* they are the entire vocabulary. SF16 is the last release
that still ships both: it plays with NNUE but `eval` still prints the classical
term table (Material, Imbalance, Pawns, per-piece, Mobility, King safety,
Threats, Passed, Space, Winnable). So the interpretability layer here rides on
official Stockfish code that exists in exactly one release.

### How much of the black box is legible? (`results/eval_decompose.json`)

Regressing SF16's opaque **NNUE eval** onto its own classical human-readable
terms over ~2500 positions:

- **R² ≈ 0.74** (n = 2500) — about three-quarters of the NNUE number is
  reconstructible from concepts a coach already uses.
- Material + imbalance alone give R² ≈ 0.66; the positional terms add ~7
  points, led (incremental R² over material) by **King safety, Passed pawns,
  and Mobility** — i.e. NNUE's "secret sauce" over material is mostly those
  three legible ideas, not something ineffable. (Positional terms alone,
  ignoring material, explain only R² ≈ 0.18 — material is still the backbone.)

The ~26% that classical terms miss is the genuine NNUE edge (subtle
piece-coordination / long-range king-safety patterns) — a quantified ceiling on
how far hand-crafted interpretability can go.

### Two shipping interpretability tools

1. **`Explain` UCI patch** (`results/explain.patch`, Part 2): per-root-move
   search effort as an MCTS-style candidate distribution. **Bench-identical, 0
   Elo.**
2. **`scripts/coach.py`**: a human coach layer. It plays the engine's move, then
   diffs SF16's classical term table before/after to say *why* in plain terms.
   Built as a thin wrapper over the stock `eval` command (ponytail rung 1: the
   term table already exists — don't reimplement it in C++). Examples:

   ```
   c5:  advances/creates a passed pawn (+0.39).            [K+P endgame]
   a3:  improves king safety (+0.28); creates threats      [Italian]
        (+0.15); gains mobility (+0.14).
   ```

   This is faithful (the terms shown are Stockfish's own), unlike an LLM's
   post-hoc story, and costs one extra `eval` call (~1 ms).

### Placing the interpretable engines on the ladder (`chart_ladder.png`)

Both no-/low-search interpretable players were measured against the 32- and
64-node SF rungs (40 games each):

| Engine | Elo | Interpretability |
|---|---|---|
| material-only αβ | ~1009 | total (one number, addable by hand) |
| **rule-set αβ** (Huber piece values + Michniewski PST principles) | **~1324** | total (fits on a page) |
| **SF NNUE as pure policy** (depth 1, no search) | **~1336** | low (opaque net, one ply) |
| SF16 @ 32 nodes | ~1468 | medium |

The punchline for the frontier: a **fully human-readable rule set (1324) is
within ~12 Elo of Stockfish's own neural net used without search (1336).**
Stockfish's NNUE is trained to be a *search heuristic*, not a standalone
player — so stripping its search throws away almost everything, landing it next
to a page of principles. This is the exact opposite of Leela, whose net is a
policy+value net that reaches ~2500 with no search: the "2500 with no search"
regime the user asked about is a Leela/AlphaZero property, not reachable by
removing search from Stockfish. Filling that part of the frontier requires a
Leela-style *value* net, not SF's search-oriented one.

## Part 4: quantifying "less human-like" (`chart_humanlike.png`)

The chess-LLM / Maia selling point is that they *predict human moves*, not best
moves. We measured the Stockfish baseline they must beat: over 3,500 quiet
lichess positions with the human move known, how often does SF (at several node
budgets) play the same move?

- **SF move-match with humans is non-monotonic and peaks at ~256 nodes / ~1560
  Elo (35% top-1, 68% top-3), then FALLS as SF gets stronger** (65k nodes:
  29% / 57%). A weak-but-searching engine is not maximally human; there's a
  sweet spot around club level, and super-GM Stockfish is *less* human-like
  than 1560-Elo Stockfish.
- **None of the SF settings reach the ~46–52% top-1 that Maia / a dedicated
  chess-LLM report** — that gap is exactly the value those human-imitation
  models add, and it is real: matching human moves is a different objective
  than playing well, and Stockfish optimizes the wrong one.
- **Stronger humans play more Stockfish-like moves** monotonically (24% → 45%
  top-1 at 256 nodes across <1400 → 2200+ bands). "Playing like the engine" is
  almost a definition of chess strength — which is why an engine is a poor
  model of a *weak* human specifically.
- Mean centipawn-loss of the human move rises with the yardstick's depth (83cp
  at 32 nodes → 109cp at 65k): deeper search finds refutations that make human
  moves look worse, i.e. "how far from optimal are humans" is itself
  depth-dependent.

Takeaway for the frontier: to be *both* strong and human-like you need a model
trained on human games (Maia, chess-LLM), not a de-tuned engine. Stockfish buys
strength by becoming less human; the interpretability tools in Parts 2–3 are
therefore about explaining engine choices in human terms, not about making the
engine choose human moves — two different goals this data cleanly separates.
