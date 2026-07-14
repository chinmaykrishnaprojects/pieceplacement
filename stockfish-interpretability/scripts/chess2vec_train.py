"""Train chess2vec: skip-gram encoder + embedding->FEN decoder, on CPU.

Loss = skip-gram negative sampling (cosine): pull f(P) toward f(P_future),
push away from in-batch negatives; PLUS a reconstruction loss training a decoder
g(f(P)) -> board (so the embedding is invertible: embedding->FEN).

Saves: models/chess2vec.pt (encoder+decoder), and results/chess2vec_emb.npz
(embeddings + labels for the whole corpus, for nearest-neighbor + probing).
"""
import sys
import time

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F

sys.path.insert(0, "scripts")
from chess2vec import Encoder, Decoder, board_vec
import chess

IN = "data/pairs.csv.gz"
D = int(sys.argv[1]) if len(sys.argv) > 1 else 256
EPOCHS = int(sys.argv[2]) if len(sys.argv) > 2 else 6
BATCH = 512


def encode_frame(fens):
    return np.stack([board_vec(chess.Board(f)) for f in fens])


def main():
    torch.manual_seed(0)
    df = pd.read_csv(IN)
    print(f"{len(df)} pairs", flush=True)
    # precompute board vectors (768-dim) for anchors and contexts
    t0 = time.time()
    A = torch.tensor(encode_frame(df.fen.values))
    Cx = torch.tensor(encode_frame(df.fen_ctx.values))
    print(f"encoded boards in {time.time()-t0:.0f}s", flush=True)

    enc, dec = Encoder(D), Decoder(D)
    opt = torch.optim.Adam(list(enc.parameters()) + list(dec.parameters()), lr=1e-3)
    n = len(A)
    idx = np.arange(n)
    for ep in range(EPOCHS):
        np.random.shuffle(idx)
        tot_sg = tot_rec = 0.0
        for s in range(0, n, BATCH):
            bi = idx[s:s + BATCH]
            a = A[bi]
            c = Cx[bi]
            za = enc(a)                 # [B,D] unit
            zc = enc(c)
            # skip-gram: cosine sim matrix, positives on diagonal (InfoNCE)
            logits = za @ zc.t() / 0.1  # temperature
            labels = torch.arange(len(bi))
            loss_sg = F.cross_entropy(logits, labels)
            # reconstruction: decode anchor embedding back to its board
            recon = dec(za)
            loss_rec = F.binary_cross_entropy_with_logits(recon, a)
            loss = loss_sg + 0.5 * loss_rec
            opt.zero_grad()
            loss.backward()
            opt.step()
            tot_sg += loss_sg.item() * len(bi)
            tot_rec += loss_rec.item() * len(bi)
        print(f"epoch {ep+1}/{EPOCHS}  skipgram {tot_sg/n:.4f}  recon {tot_rec/n:.4f}",
              flush=True)

    torch.save({"enc": enc.state_dict(), "dec": dec.state_dict(), "d": D},
               "models/chess2vec.pt")

    # embed the whole corpus (anchors) for NN + probing
    enc.eval()
    with torch.no_grad():
        emb = enc(A).numpy()
    np.savez_compressed("results/chess2vec_emb.npz",
                        emb=emb, fen=df.fen.values,
                        n_pieces=df.n_pieces.values, stm_material=df.stm_material.values,
                        phase=df.phase.values, closed=df.closed.values)
    # reconstruction accuracy (per-square piece-presence)
    with torch.no_grad():
        recon = torch.sigmoid(dec(enc(A[:2000]))).numpy()
    true = A[:2000].numpy()
    acc = float(((recon > 0.5) == (true > 0.5)).mean())
    print(f"\nembedding->board per-square accuracy: {acc:.3f}")
    print("saved models/chess2vec.pt + results/chess2vec_emb.npz")


if __name__ == "__main__":
    main()
