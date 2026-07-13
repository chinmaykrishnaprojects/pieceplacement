"""Human coach layer over Stockfish 16's classical eval terms.

SF16 still exposes `eval`'s classical term table (the last version that does;
removed right after in PR #4674). This wrapper turns the engine's chosen move
into a coach-style sentence by diffing the classical term breakdown of the
position before and after the move, in the vocabulary a human uses
(king safety, passed pawns, mobility, space, threats, ...).

Reuses the stock binary + `eval` command (ponytail: no new C++). Combine with
the `Explain` patch (per-move effort) for the full picture.

Usage: coach.py "<FEN>" [nodes]
"""
import json
import re
import subprocess
import sys

import chess

SF_PLAY = "/usr/games/stockfish"
SF_EVAL = "/home/user/stockfish-interp/src/stockfish-16/src/stockfish"
K_CP = 468.0  # human expected-score scale from analysis.json if present

TERMS = ["Material", "Imbalance", "Pawns", "Knights", "Bishops", "Rooks",
         "Queens", "Mobility", "King safety", "Threats", "Passed", "Space"]
PHRASE = {
    "Material": "wins material", "Imbalance": "improves the piece mix",
    "Pawns": "improves the pawn structure", "Knights": "activates a knight",
    "Bishops": "activates a bishop", "Rooks": "activates a rook",
    "Queens": "improves the queen", "Mobility": "gains mobility",
    "King safety": "improves king safety", "Threats": "creates threats",
    "Passed": "advances/creates a passed pawn", "Space": "gains space",
}
ROW_RE = {t: re.compile(r"\|\s*" + re.escape(t) +
                        r"\s*\|.*\|.*\|\s*(-?\d+\.\d+)\s+(-?\d+\.\d+)\s*\|")
          for t in TERMS}
FINAL_RE = re.compile(r"Final evaluation\s+([+-]?\d+\.\d+)")


def eval_terms(fen):
    p = subprocess.run([SF_EVAL], input=f"position fen {fen}\neval\nquit\n",
                       capture_output=True, text=True, timeout=20)
    if "in check" in p.stdout:
        return None, None
    terms = {}
    for t, rx in ROW_RE.items():
        m = rx.search(p.stdout)
        terms[t] = float(m.group(1)) if m else 0.0  # White-POV MG total
    f = FINAL_RE.search(p.stdout)
    return terms, (float(f.group(1)) if f else None)


def best_move(fen, nodes):
    out = subprocess.run(
        [SF_PLAY], input=f"position fen {fen}\ngo nodes {nodes}\n",
        capture_output=True, text=True, timeout=30).stdout
    m = re.search(r"bestmove (\S+)", out)
    sc = re.findall(r"score cp (-?\d+)", out)
    return m.group(1), (int(sc[-1]) if sc else None)


def coach(fen, nodes=200000):
    board = chess.Board(fen)
    stm = board.turn
    bm_uci, cp = best_move(fen, nodes)
    move = chess.Move.from_uci(bm_uci)
    san = board.san(move)
    t0, _ = eval_terms(fen)
    board.push(move)
    t1, _ = eval_terms(board.fen())
    # term deltas from the mover's POV (White-POV terms; flip if Black moved)
    sign = 1 if stm == chess.WHITE else -1
    deltas = {t: sign * (t1[t] - t0[t]) for t in TERMS} if t0 and t1 else {}
    ranked = sorted(deltas.items(), key=lambda kv: -kv[1])
    gains = [f"{PHRASE[t]} ({d:+.2f})" for t, d in ranked[:3] if d > 0.03]
    costs = [f"{PHRASE[t]} ({d:+.2f})" for t, d in ranked[::-1][:2] if d < -0.03]
    why = "; ".join(gains) if gains else "keeps the position balanced"
    if costs:
        why += " — at the cost of " + ", ".join(costs)
    return {
        "move": san,
        "eval_pawns": round((cp or 0) / 100, 2),
        "coach": f"{san}: {why}.",
        "term_deltas": {t: round(d, 2) for t, d in ranked if abs(d) > 0.03},
    }


if __name__ == "__main__":
    fen = sys.argv[1] if len(sys.argv) > 1 else chess.Board().fen()
    nodes = int(sys.argv[2]) if len(sys.argv) > 2 else 200000
    print(json.dumps(coach(fen, nodes), indent=2))
