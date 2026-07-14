# Copy-paste prompt for a GPU-enabled Claude session

Paste everything below the line into a fresh Claude Code session running on your
GPU machine. It is self-contained: it clones the base model, builds the data,
runs the highest-value training item (LoRA fine-tune of the 50M chess-GPT toward
Stockfish-preferred moves), and evaluates it with the same harness used in the
CPU study.

---

You are working on a chess-AI research project. Goal: LoRA fine-tune Adam
Karvonen's 16-layer ~50M nanoGPT chess model (`lichess_16layers`) toward
Stockfish-preferred moves, to trace a strength-vs-humanness curve. This machine
has a GPU; use it.

## Background facts (already measured on a CPU box, don't re-derive)
- Base model: `adamkarvonen/chess_llms`, file
  `lichess_16layers_ckpt_no_optimizer.pt`. Config: n_layer=16, n_head=8,
  n_embd=512, block_size=1023, char-level vocab of 32 (Karvonen PGN format
  `;1.e4 e5 2.Nf3 ...`).
- The lichess-trained model plays ~1212 Elo (argmax, no search) and matches
  human moves 42.5% of the time. The stockfish-trained sibling plays ~1407 Elo
  but only 34.3% human-match. Hypothesis: a LoRA toward Stockfish moves moves the
  lichess model UP in Elo and DOWN in human-match, tracing the tradeoff curve.

## Steps
1. **Setup**
   ```bash
   git clone https://github.com/karpathy/nanoGPT
   pip install torch peft transformers datasets python-chess zstandard tqdm
   huggingface-cli download adamkarvonen/chess_llms lichess_16layers_ckpt_no_optimizer.pt --local-dir ckpt
   ```
   Reconstruct the GPT class from the checkpoint's `model_args` (or use nanoGPT's
   `model.py` GPTConfig with those args). The char vocab is the 32 sorted chars
   of `" #+-.0123456789;=BKNOQRabcdefghx"` (stoi = sorted index).

2. **Build training data** — `(pgn_prefix, target_move)` where target = the move
   Stockfish prefers. Two options:
   - Cheap: stream a lichess monthly dump
     (`https://database.lichess.org/standard/lichess_db_standard_rated_2024-02.pgn.zst`),
     keep games with `[%eval]`, and for each quiet position use the move that
     leads to the best eval among the moves actually seen — OR
   - Better: run a local Stockfish at depth 12 over ~1-2M sampled positions and
     record its bestmove. Format each example as the Karvonen prefix followed by
     the target move's SAN, so the LM is trained to continue the PGN with SF's move.

3. **LoRA fine-tune** (QLoRA fits a 12-16GB card; full-precision LoRA on 24GB):
   - Target the attention projection matrices (`c_attn`, `c_proj`) and MLP
     (`c_fc`, `c_proj`) of each block. rank=16, alpha=32, dropout=0.05.
   - lr=1e-4 cosine, batch as large as fits, bf16, 1-3 epochs over the data.
   - Loss = next-char cross-entropy on the target-move characters only (mask the
     prefix), so you teach "given this game, emit SF's move".

4. **Evaluate** (this is the deliverable). Copy these two scripts from the CPU
   repo (they only need python-chess + the model): `chessgpt_local.py` (nanoGPT
   forward + char vocab + `policy`), `chessgpt_humanmatch.py`, `chessgpt_ladder.py`.
   Point them at the merged LoRA checkpoint and run:
   - human-match top-1 on lichess positions (expect it to DROP from 42.5%),
   - ladder vs Stockfish node rungs for Elo (expect it to RISE from ~1212).
   Report both numbers and, if you sweep LoRA strength (alpha ∈ {8,16,32,64}),
   plot Elo vs human-match — that curve is the result.

5. **Sanity**: greedy decode from `;1.` must still produce legal PGN. If it
   emits garbage, the vocab/order is wrong — fix stoi before training conclusions.

Deliver: the merged checkpoint, the two eval numbers (and the sweep curve if
run), and a one-paragraph summary of where the LoRA'd model lands on the
strength-vs-humanness tradeoff versus the base lichess and base stockfish models.
