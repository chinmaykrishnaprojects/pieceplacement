"""Run Adam Karvonen's chess-GPT (nanoGPT) checkpoints LOCALLY on this server.

These are the actual weights behind the user's app (adamkarvonen/chess_llms,
16-layer ~50M, lichess_l11 / stockfish_l11 variants). Char-level autoregressive
GPT over PGN text, no search. We reimplement the minimal nanoGPT forward pass
(so no repo dependency) and the char vocab, then expose a policy over legal
moves by scoring each legal move's SAN continuation under the LM.

The model consumes PGN in Karvonen's exact format: ";1.e4 e5 2.Nf3 ..." — a
leading ';', move number + '.', no space after the dot, space between the two
half-moves. We build that string from the board's move stack.
"""
import math
import sys

import chess
import torch
import torch.nn as nn
import torch.nn.functional as F

# Karvonen's 32-char vocabulary (nanoGPT prepare.py: sorted(set(data))).
# Verified empirically: with this ordering the model generates legal PGN.
VOCAB = sorted(list(" #+-.0123456789;=BKNOQRabcdefghx"))
STOI = {c: i for i, c in enumerate(VOCAB)}
ITOS = {i: c for i, c in enumerate(VOCAB)}


class Block(nn.Module):
    def __init__(self, n_embd, n_head):
        super().__init__()
        self.ln_1 = nn.LayerNorm(n_embd, bias=False)
        self.attn = CausalSelfAttention(n_embd, n_head)
        self.ln_2 = nn.LayerNorm(n_embd, bias=False)
        self.mlp = nn.ModuleDict(dict(
            c_fc=nn.Linear(n_embd, 4 * n_embd, bias=False),
            c_proj=nn.Linear(4 * n_embd, n_embd, bias=False),
        ))

    def forward(self, x):
        x = x + self.attn(self.ln_1(x))
        m = self.mlp
        x = x + m.c_proj(F.gelu(m.c_fc(self.ln_2(x))))
        return x


class CausalSelfAttention(nn.Module):
    def __init__(self, n_embd, n_head):
        super().__init__()
        self.c_attn = nn.Linear(n_embd, 3 * n_embd, bias=False)
        self.c_proj = nn.Linear(n_embd, n_embd, bias=False)
        self.n_head = n_head
        self.n_embd = n_embd

    def forward(self, x):
        B, T, C = x.shape
        q, k, v = self.c_attn(x).split(self.n_embd, dim=2)
        k = k.view(B, T, self.n_head, C // self.n_head).transpose(1, 2)
        q = q.view(B, T, self.n_head, C // self.n_head).transpose(1, 2)
        v = v.view(B, T, self.n_head, C // self.n_head).transpose(1, 2)
        y = F.scaled_dot_product_attention(q, k, v, is_causal=True)
        y = y.transpose(1, 2).contiguous().view(B, T, C)
        return self.c_proj(y)


class GPT(nn.Module):
    def __init__(self, args):
        super().__init__()
        self.block_size = args["block_size"]
        self.transformer = nn.ModuleDict(dict(
            wte=nn.Embedding(args["vocab_size"], args["n_embd"]),
            wpe=nn.Embedding(args["block_size"], args["n_embd"]),
            h=nn.ModuleList([Block(args["n_embd"], args["n_head"])
                             for _ in range(args["n_layer"])]),
            ln_f=nn.LayerNorm(args["n_embd"], bias=False),
        ))
        self.lm_head = nn.Linear(args["n_embd"], args["vocab_size"], bias=False)

    def forward(self, idx):
        B, T = idx.shape
        pos = torch.arange(T, device=idx.device)
        x = self.transformer.wte(idx) + self.transformer.wpe(pos)
        for blk in self.transformer.h:
            x = blk(x)
        x = self.transformer.ln_f(x)
        return self.lm_head(x)


def load_model(path, device="cpu"):
    ck = torch.load(path, map_location=device, weights_only=False)
    args = ck["model_args"]
    model = GPT(args)
    sd = {k.replace("_orig_mod.", ""): v for k, v in ck["model"].items()}
    # nanoGPT ties/【transposes】 nothing here; drop attn bias buffers if present
    sd = {k: v for k, v in sd.items() if not k.endswith(".attn.bias")}
    model.load_state_dict(sd, strict=True)
    model.eval().to(device)
    return model


def board_to_pgn_prefix(board: chess.Board) -> str:
    """Karvonen format ';1.e4 e5 2.Nf3 Nc6 ...' up to (not incl.) next move."""
    moves = list(board.move_stack)
    b = chess.Board()
    s = ";"
    for i, m in enumerate(moves):
        san = b.san(m)
        if i % 2 == 0:
            s += f"{i//2+1}.{san} "
        else:
            s += f"{san} "
        b.push(m)
    return s


class ChessGPT:
    def __init__(self, path, device="cpu"):
        self.model = load_model(path, device)
        self.device = device
        self.path = path

    @torch.no_grad()
    def _logprob_of(self, prefix_ids, continuation):
        """sum log p(continuation chars | prefix), teacher-forced."""
        ids = prefix_ids + [STOI[c] for c in continuation]
        x = torch.tensor([ids], device=self.device)
        logits = self.model(x)
        logp = F.log_softmax(logits[0], dim=-1)
        total = 0.0
        start = len(prefix_ids)
        for j, c in enumerate(continuation):
            total += logp[start + j - 1, STOI[c]].item()
        return total

    @torch.no_grad()
    def _logprob_batch(self, prefix_ids, continuations):
        """Total logprob of each continuation, in ONE padded forward pass.

        All sequences share the prefix; we right-pad to the longest, run the
        model once, and gather per-continuation-char logprobs (masking pad)."""
        P = len(prefix_ids)
        seqs = [prefix_ids + [STOI[c] for c in cont] for cont in continuations]
        maxlen = max(len(s) for s in seqs)
        maxlen = min(maxlen, self.model.block_size)
        pad = STOI[" "]
        batch = torch.full((len(seqs), maxlen), pad, dtype=torch.long,
                           device=self.device)
        for i, s in enumerate(seqs):
            batch[i, :min(len(s), maxlen)] = torch.tensor(s[:maxlen])
        logits = self.model(batch)                      # [B, T, V]
        logp = F.log_softmax(logits, dim=-1)
        totals = []
        for i, cont in enumerate(continuations):
            t = 0.0
            for j in range(len(cont)):
                pos = P + j
                if pos >= maxlen:
                    break
                t += logp[i, pos - 1, STOI[cont[j]]].item()
            totals.append(t)
        return totals

    def policy(self, board, pgn_prefix=None, temperature=1.0):
        """Prob over legal moves via their SAN char-continuation likelihood.

        pgn_prefix: the Karvonen-format string up to the move to predict, e.g.
        ';1.e4 e5 2.Nf3 Nc6 3.Bc4 '. If None, it is rebuilt from the board's
        move stack (only valid when the board carries its full history)."""
        prefix = pgn_prefix if pgn_prefix is not None else board_to_pgn_prefix(board)
        prefix_ids = [STOI[c] for c in prefix if c in STOI]
        # White's move is written "<n>.SAN"; Black's is just "SAN".
        num = f"{board.fullmove_number}." if board.turn == chess.WHITE else ""
        moves, conts = [], []
        for mv in board.legal_moves:
            cont = num + board.san(mv) + " "   # trailing space = move delimiter
            if all(c in STOI for c in cont):
                moves.append(mv)
                conts.append(cont)
        if not moves:
            return {}
        totals = self._logprob_batch(prefix_ids, conts)
        scores = dict(zip(moves, totals))
        mx = max(scores.values())
        exp = {m: math.exp((v - mx) / max(temperature, 1e-6)) for m, v in scores.items()}
        s = sum(exp.values())
        return {m: v / s for m, v in exp.items()}

    @torch.no_grad()
    def policy_with_acts(self, board, pgn_prefix=None, layer=11):
        """Policy AND per-move residual-stream activations, from ONE forward pass.

        The batched forward already appends each legal move's SAN to the prefix,
        so the hidden state at the LAST CHARACTER OF THE MOVE is a move-conditioned
        representation the model has already computed. Reading it out is free:
        no extra LM compute, just a slice of activations we were discarding.

        Returns ({move: prob}, {move: np.ndarray[n_embd]}).
        """
        prefix = pgn_prefix if pgn_prefix is not None else board_to_pgn_prefix(board)
        prefix_ids = [STOI[c] for c in prefix if c in STOI][-600:]
        num = f"{board.fullmove_number}." if board.turn == chess.WHITE else ""
        moves, conts = [], []
        for mv in board.legal_moves:
            cont = num + board.san(mv) + " "
            if all(c in STOI for c in cont):
                moves.append(mv)
                conts.append(cont)
        if not moves:
            return {}, {}

        P = len(prefix_ids)
        seqs = [prefix_ids + [STOI[c] for c in c2] for c2 in conts]
        maxlen = min(max(len(s) for s in seqs), self.model.block_size)
        pad = STOI[" "]
        batch = torch.full((len(seqs), maxlen), pad, dtype=torch.long,
                           device=self.device)
        for i, s in enumerate(seqs):
            batch[i, :min(len(s), maxlen)] = torch.tensor(s[:maxlen])

        grabbed = {}
        blk = self.model.transformer.h[layer - 1]
        h = blk.register_forward_hook(
            lambda m, i, o: grabbed.__setitem__("a", o.detach()))
        logits = self.model(batch)
        h.remove()
        acts = grabbed["a"]                       # [B, T, n_embd]
        logp = F.log_softmax(logits, dim=-1)

        totals, vecs = [], []
        for i, cont in enumerate(conts):
            t = 0.0
            for j in range(len(cont)):
                pos = P + j
                if pos >= maxlen:
                    break
                t += logp[i, pos - 1, STOI[cont[j]]].item()
            totals.append(t)
            # last character OF THE MOVE (cont ends with the delimiter space)
            last = min(P + len(cont) - 2, maxlen - 1)
            vecs.append(acts[i, last].cpu().numpy())

        mx = max(totals)
        exp = [math.exp(v - mx) for v in totals]
        ssum = sum(exp)
        pol = {m: e / ssum for m, e in zip(moves, exp)}
        return pol, {m: v for m, v in zip(moves, vecs)}

    def play(self, board, pgn_prefix=None, temperature=0.0):
        pol = self.policy(board, pgn_prefix=pgn_prefix,
                          temperature=temperature if temperature > 0 else 1.0)
        return max(pol, key=pol.get) if pol else None


if __name__ == "__main__":
    path = sys.argv[1] if len(sys.argv) > 1 else "models/lichess_16layers_ckpt_no_optimizer.pt"
    eng = ChessGPT(path)
    # 1) raw generation sanity: does it produce legal PGN from ';1.'?
    import torch as T
    ids = [STOI[c] for c in ";1."]
    x = T.tensor([ids])
    for _ in range(40):
        with T.no_grad():
            lg = eng.model(x)
        nxt = int(lg[0, -1].argmax())
        x = T.cat([x, T.tensor([[nxt]])], dim=1)
    print("greedy gen:", "".join(ITOS[int(i)] for i in x[0]))
    # 2) policy on start + Italian
    b = chess.Board()
    pol = eng.policy(b)
    print("startpos top5:", [(b.san(m), round(p, 3))
                             for m, p in sorted(pol.items(), key=lambda kv: -kv[1])[:5]])
