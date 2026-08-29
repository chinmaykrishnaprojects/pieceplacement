# Shared research wiki — "push the interpretable chess frontier"

Every agent working on this goal **reads this file first** and **appends its
findings before finishing**. This is the only channel between agents.

## The goal (fixed, not negotiable by agents)

Take a frozen 50M char-level chess language model (Karvonen `lichess_16layers`,
trained on human lichess PGN) that plays ~1130–1300 Elo with **zero search**, and
raise its playing strength using **as few extra LM calls per move as possible**,
while keeping the method **human-legible** (a person can read off the handful of
candidate lines the player actually considered).

Success = a strength-per-LM-call curve that beats the no-search baseline.

## Fitness (ungameable — do not try to optimise around it)

`arena.py` plays your candidate against **Stockfish at fixed node budgets** and
computes Elo from the actual game results. You cannot self-report a score.

- You submit **only** `candidates/<name>.py` defining `create_player(oracle, budget)`.
- `oracle.policy(board, pgn) -> {move: prob}` is the **only** access to the model.
  **Every call costs 1 unit of `budget`**, metered by the harness. Exceeding the
  per-move quota raises `BudgetExhausted` → that game is **forfeit**.
- Illegal move → forfeit. Crash → forfeit.
- Banned in candidate source: `subprocess`, `socket`, `urllib`, `requests`,
  `ctypes`, `importlib`, `eval`, `exec`, `open`, `chess.engine`, `stockfish`.
  A candidate containing any of these is rejected unscored.
- Free and encouraged: python-chess, arithmetic, your own heuristics (material,
  mobility, king safety, SEE, move ordering, quiescence over captures).

## The interface

```python
class Player:
    def __init__(self, oracle, budget):
        self.oracle = oracle      # .policy(board, pgn) -> {chess.Move: float}
        self.budget = budget      # per-move LM call quota

    def play(self, board, pgn):   # -> chess.Move (must be legal)
        ...

def create_player(oracle, budget):
    return Player(oracle, budget)
```

Notes that matter:
- `pgn` is the Karvonen-format prefix (`;1.e4 e5 2.Nf3 `). If you search deeper,
  **extend it correctly** for child nodes or the LM loses its context and its
  policy degrades badly. Helper pattern:
  `pgn + (f"{board.fullmove_number}.{san} " if board.turn == chess.WHITE else f"{san} ")`
- The policy is a *human-move* distribution, not an evaluation. It says what a
  club player would play, which is not always what wins.
- **Spend the budget adaptively.** Most positions are obvious; a few are sharp.
  A player that spends 1 call on quiet positions and its whole budget on tactical
  ones dominates one that spends uniformly.

## Baseline to beat

| candidate | budget | Elo vs SF@32 | note |
|---|---|---|---|
| `baseline_policy` | 1 | ~1130 (4-game pilot) | pure argmax policy, no search |

## Results log

Machine-written to `wiki/results.jsonl` by the orchestrator. Read it for what has
already been tried and how it scored. **Do not repeat a strategy that scored
badly — build on what worked or try a genuinely different axis.**

## Agent notes

Append below. Be concrete: what you tried, why, what you predict, what you'd try
next given the result. Future agents rely on this.

---

## policy_negamax (gen 1)

**Shape.** Best-first negamax over an explicit tiny tree. One LM call = one node
expansion; each call goes to the current principal-variation leaf, so the budget
self-allocates (quiet positions stay flat, sharp ones get depth). Root breadth
`min(5, max(2,B))` (5 at B=1), reply breadth 2 for B<=3 else 3, depth cap 6.
B=1 is root-only: policy candidates ranked by my own quiescence score — strictly
more than argmax at the same cost. Root pick = `-childval + 55*policy_prob`
(70 at B=1), minus 25 for a root move that was never expanded.

**Leaf eval** (free): tapered MG/EG PSTs, material, mobility (N4/B4/R3/Q1 per
safe square), pawn shield + king-ring attack pressure, passed/doubled/isolated
pawns, bishop pair, rook on pawn-free / semi-free file and 7th, tempo.

**Quiescence:** yes — captures filtered by full occupancy-based SEE (x-ray
correct), delta pruning, check evasions, depth 5 / 500 nodes. Free extras:
mate-in-1 shortcut, mate-in-1-reply refutation on root children, and SEE>0
captures added to candidate sets the LM's human distribution omits.

**Prediction:** +80–150 Elo over baseline at B=4; small gain (+20–40) at B=1.
Main risk: the LM's top-3 replies miss the real refutation, so the tree is
optimistic; the SEE extras only patch the material case.

**Next:** tune PRIOR_W / eval weights empirically, and widen reply breadth only
when the root's top two moves are within ~50cp (cheap uncertainty trigger).

## blunder_filter (gen 1)

ONE policy call, then zero-budget tactics on its top 6 moves (plus any capture
with SEE >= 95 the policy ignored). Checks, all free: full **SEE** with x-ray
re-attackers (verified -220 on the classic Nxe5 swap test); a **capture-only
quiescence** (depth 4, check-evasion extension, SEE<0 pruned, 3000-node cap);
**mate-in-1 for us** (played instantly, costs 0 budget — so budget=0 positions
are still handled); **opponent mate-in-1 after our move** (scored -MATE);
stalemate/insufficient-material avoidance when ahead.

Tradeoff: score = dv + 150*p, where dv is the candidate's quiescence value minus
the best candidate's. Drops under 60cp are forgiven outright (quiescence is not
an evaluation, and human moves encode strategy we cannot see); beyond 60cp every
centipawn counts **double**, so no policy confidence buys a hung piece. Budget>1
spends extra calls asking the policy for the opponent's top-3 replies to our
leading candidates and takes the worst case (catches quiet forks/threats that
capture-only quiescence misses).

Cost: ~2.6ms/move at budget=1. Prediction vs baseline at budget=1: +150 to +250
Elo — the baseline's losses are overwhelmingly one-move material gifts.

Next: replace material-only quiescence leaves with a small PST/mobility term,
and widen the candidate set adaptively when top-2 policy mass is flat.

## adaptive_budget (gen 1)

**Spend rule.** 0 calls if the move is forced or a free mate-in-1 exists. 1 call
otherwise. Extra calls only if the position is *sharp*: top1 prob < 0.60 **and**
margin p1−p2 < 0.33, **or** the free search disagrees with the policy favourite
by ≥45cp, **or** cheap tactics exist (in check / a capture with raw gain ≥140cp /
one of our ≥minor pieces attacked by something cheaper), **or** the top two
candidates score within 25cp.

**Always-free blunder filter.** Even at budget=1 the top-≥3 policy moves are
re-ranked by `qval + 110·log(p/p_top)` where qval is a depth-1 + quiescence
search over material + PSTs + bishop pair. It can only overrule the policy when
a cheap search says the favourite drops material. This is the main change vs
baseline at equal cost.

**Sharp branch.** One child policy call per top-k move (k=min(budget−1,3)); the
opponent's LM top-3 replies ∪ 3 material-best replies are each searched a
further ply. ~4-ply policy-pruned lines, all named in `last_lines`.

**Prediction.** Budget=1 should beat baseline outright (free filter, same cost).
Budget=4 spends ~2.7 calls/move on peaked policies, so it should land above
baseline on the strength-per-call curve rather than just on Elo. Risk: K_POL=110
may be too trusting of the LM to catch its worst blunders.

**Next variation.** Tune K_POL (try 70) and, more promisingly, make the sharpness
test *asymmetric*: only spend when the free search suspects the policy favourite
loses material, since uncertainty alone often just means "several fine moves".

---

---

# GEN-1 RESULTS (orchestrator, measured — read before proposing gen 2)

Opponent: Stockfish @ 32 nodes (= 1468 Elo on this project's ladder).
No candidate ever recorded an `illegal`, `crash`, or `budget_overrun` flag.

## Same-cost comparison (budget = 1 LM call/move), pooled over 50 games
seed 7 (20 games, selection) + seed 99 (30 games, HELD OUT):

| candidate | pooled score | Elo | vs baseline | significance |
|---|---|---|---|---|
| **adaptive_budget** | 23.0/50 = .460 | **1440** | **+247** | 3.3 sigma |
| blunder_filter | 18.0/50 = .360 | 1368 | +175 | 2.2 sigma |
| baseline_policy | 8.5/50 = .170 | 1193 | — | — |

Winner's curse was real: adaptive_budget scored 1503 on the selection seed but
1398 on the held-out seed. Always validate on an unseen seed. The baseline
reproduced across seeds (1199 vs 1188), so the harness itself is stable.

## Strength per LM call (budget 1 -> 4, 20 games, seed 7)

| candidate | b=1 | b=4 | gain from 3x more model calls |
|---|---|---|---|
| adaptive_budget | 1503 (37 calls/game) | 1521 (121) | **+18** |
| blunder_filter | 1398 (32) | 1398 (100) | **0** |
| policy_negamax | 1360 (36) | 1451 (168) | +91 |
| baseline_policy | 1199 (29) | — | — |

## THE FINDING

**The LM's missing Elo is tactical, not strategic, and recovering it is free.**
~+250 Elo comes from zero-budget arithmetic (SEE, capture quiescence, mate-in-1)
vetoing the policy's blunders. Tripling actual model calls adds ~0-90 more.
The human-trained policy already picks good plans; it just hangs pieces.

Corollary for gen 2: **do not spend effort on more LM calls.** The marginal
value of model compute here is near zero. Spend it on better free arithmetic,
or on deciding *when* the policy is about to blunder.

## Open questions for gen 2
1. adaptive_budget beat blunder_filter by ~70 Elo — is that its 0-cost
   `qval + 110*log(p/p_top)` re-ranking, or the sharpness gating? Ablate.
2. Both winners are very drawish (14-15 draws/20-30). They avoid losing but
   rarely win. Is there a free way to convert drawn-ish positions?
3. policy_negamax is the only one still climbing at b=4. Does it overtake
   above b=8, or is its budget self-allocation just inefficient at low b?
4. Nobody tried: using the policy's probability as a *danger signal* (low
   top1 prob = the model is confused = likely blunder ahead) to trigger deeper
   free search rather than more LM calls.
