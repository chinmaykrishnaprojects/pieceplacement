"""Retrain chess2vec on the SAME games used for the LM comparison.

Skip-gram over positions: a position is known by the positions that follow it in
the same game. Pairs are drawn from data/fen_moves_pgn.csv.gz grouped by game_id,
so both sides of the comparison see the same underlying games.

Encoder: 768-d color-agnostic board planes -> 256-d unit vector.
Also trains a decoder for the embedding->board reconstruction probe.
"""
import sys
import time

import chess
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F

sys.path.insert(0, "scripts")
from chess2vec import Encoder, Decoder, board_vec

D = 256
EPOCHS = 8
BATCH = 512


def main():
    torch.manual_seed(0)
    df = pd.read_csv("data/fen_moves_pgn.csv.gz")
    # positive pairs: two positions from the same game (later position = context)
    pairs = []
    for _, g in df.groupby("game_id"):
        g = g.sort_values("ply")
        fens = list(g.fen)
        for i in range(len(fens) - 1):
            pairs.append((fens[i], fens[i + 1]))
    print(f"{len(pairs)} same-game position pairs")

    t0 = time.time()
    A = torch.tensor(np.stack([board_vec(chess.Board(a)) for a, _ in pairs]))
    C = torch.tensor(np.stack([board_vec(chess.Board(b)) for _, b in pairs]))
    print(f"encoded boards in {time.time()-t0:.0f}s")

    enc, dec = Encoder(D), Decoder(D)
    opt = torch.optim.Adam(list(enc.parameters()) + list(dec.parameters()), lr=1e-3)
    n = len(A)
    idx = np.arange(n)
    for ep in range(EPOCHS):
        np.random.shuffle(idx)
        sg = rec = 0.0
        for s in range(0, n, BATCH):
            bi = idx[s:s + BATCH]
            if len(bi) < 8:
                continue
            a, c = A[bi], C[bi]
            za, zc = enc(a), enc(c)
            loss_sg = F.cross_entropy(za @ zc.t() / 0.1, torch.arange(len(bi)))
            loss_rec = F.binary_cross_entropy_with_logits(dec(za), a)
            loss = loss_sg + 0.5 * loss_rec
            opt.zero_grad(); loss.backward(); opt.step()
            sg += loss_sg.item() * len(bi); rec += loss_rec.item() * len(bi)
        print(f"epoch {ep+1}/{EPOCHS} skipgram {sg/n:.4f} recon {rec/n:.4f}",
              flush=True)

    torch.save({"enc": enc.state_dict(), "dec": dec.state_dict(), "d": D},
               "models/chess2vec.pt")
    print("saved models/chess2vec.pt")


if __name__ == "__main__":
    main()
