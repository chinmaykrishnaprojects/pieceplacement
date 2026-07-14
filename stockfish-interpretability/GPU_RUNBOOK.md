# GPU runbook — training-heavy experiments for your local machine

This box is 4 CPU cores, no GPU, so the items below are specified here as
ready-to-run recipes for an overnight run on your GPU machine. Each says what to
run, the expected cost, and the eval that decides success (all evals reuse the
CPU-side harness in `scripts/` so results drop straight onto the frontier).

Base weights everywhere: `adamkarvonen/chess_llms` (16-layer ~50M nanoGPT,
`lichess_16layers_ckpt_no_optimizer.pt`). Clone nanoGPT
(`github.com/karpathy/nanoGPT`) and Karvonen's `chess_gpt_eval` /
`chess_llm_interpretability` for the data prep + training loop.

## 0. Environment
```bash
git clone https://github.com/karpathy/nanoGPT && cd nanoGPT
pip install torch numpy transformers datasets tiktoken wandb tqdm python-chess
# GPU: a single 24GB card (3090/4090) trains the 50M model comfortably at
# block_size 1023, batch 24, bf16.
```

## 1. BPE tokenizer + retrain (efficiency → more search per token)
**Why:** the model is char-level (vocab 32). BPE over a PGN corpus merges
frequent move-strings (`e4`, `Nf3`, ` O-O `) into single tokens, cutting
sequence length ~3-4x → longer game context and ~3-4x cheaper inference at the
same board coverage. This directly buys frontier movement (cost axis).

**Measure first (CPU, already runnable here):** `scripts/bpe_compress.py`
(this repo) reports the char→BPE compression ratio on a lichess PGN sample so
you know the win before spending GPU.

**Run:** train a BPE tokenizer (HF `tokenizers`, vocab ~4k) on ~1e8 chars of
lichess PGN; re-run nanoGPT `prepare.py` with it; retrain the 16-layer config
(`n_layer=16 n_embd=512 block_size=1023`, but now block_size covers ~3-4x more
moves). ~600k iters ≈ overnight on one 4090.

**Eval:** point `scripts/chessgpt_local.py` at the new checkpoint (swap the
vocab), then run `scripts/chessgpt_humanmatch.py` and `scripts/chessgpt_ladder.py`.
Success = same/upper human-match and Elo at lower tokens/move.

## 2. LoRA fine-tune on Stockfish evals (strength without full retrain)
**Why:** cheapest path to more Elo. The lichess model is human (good for
human-match) but ~1200 playing Elo; a LoRA toward Stockfish-preferred moves
should raise Elo while keeping most of the human prior + interpretability.

**Data:** you already have `data/positions.csv.gz` (lichess depth-18 evals) and
can pull more from the lichess eval API. Build `(pgn_prefix, best_move)` pairs
where `best_move` = argmax Stockfish move (or the top-eval legal move). Format in
Karvonen PGN.

**Run (QLoRA, fits a 12-16GB card):**
```bash
# peft + bitsandbytes; rank 16, alpha 32, lr 1e-4, ~1-3 epochs over ~1-5M pairs
python train_lora.py --base lichess_16layers.pt --data sf_moves.jsonl \
  --rank 16 --alpha 32 --lr 1e-4 --epochs 2 --bf16
```
**Eval:** `scripts/chessgpt_ladder.py` vs SF rungs for Elo; `chessgpt_humanmatch.py`
for the humanness it trades away. Plot both variants → a measured
strength↔humanness curve (extends Part-5 finding from 2 points to a curve).

## 3. Pause / thinking tokens (search-free skill bump)
**Why:** inserting learnable `<pause>` tokens (Goyal et al.) or a short scratchpad
before the move gives the model extra forward-compute per move with no tree
search — the "add thinking without search" idea. Cheap to try as a fine-tune.

**Run:** add K pause tokens to the vocab; fine-tune so the target is
`<pause>*K <move>`; at inference emit the pauses then read the move. Sweep K∈{1,4,16}.
**Eval:** `chessgpt_ladder.py` — does Elo rise monotonically with K at fixed
params? Report Elo(K) and tokens/move (cost axis).

## 4. Diffusion / multi-token move prediction (inference throughput)
**Why:** a discrete-diffusion or multi-token head predicts a whole move (or
several plies) per step instead of char-by-char, cutting latency and enabling
cheap parallel rollouts → makes the rudimentary search in
`scripts/chessgpt_search.py` much cheaper (more nodes per second, frontier win).

**Run:** simplest first — a multi-token (block) prediction head fine-tuned to
emit the next full SAN token-group; compare to char AR. Diffusion (SEDD/MDLM
style over the 32-char vocab) is the ambitious version.
**Eval:** moves/sec at equal move-match; then feed into `chessgpt_search.py` and
re-measure Elo-vs-nodes (should shift the strength-per-node curve left).

## 5. Probe-guided / J-space steering (interpretability → control)
**Why:** the board-state probes (`scripts/probe_boardstate.py`, run on CPU here)
give linear directions for "piece on square". Adding a probe-derived board
embedding as an auxiliary input, or steering along probe directions, tests
whether making the world-model explicit improves play — and gives a J-space-style
control knob.
**Run:** train the probes at scale (GPU makes the per-square logistic → a small
MLP feasible), then either (a) concatenate the decoded board to the residual as
an aux loss, or (b) activation-steer at layer 11 (the best probe layer, which is
also the `l11` in your app) and watch move changes.
**Eval:** move-match + ladder with/without steering; qualitative: does steering a
"my king is unsafe" direction make it play safer moves?

## Priority for one overnight run
1. **LoRA on Stockfish (#2)** — highest Elo/hour, directly extends the measured
   strength↔humanness curve.
2. **Pause tokens (#3)** — cheap, clean "thinking without search" result.
3. **BPE retrain (#1)** — bigger lift, needs a full retrain; do after #2 confirms
   the harness end-to-end.

Ping me the machine access and I'll turn the chosen item into an exact,
copy-paste launch script against your data paths.
