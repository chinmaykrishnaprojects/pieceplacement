"""Interpretable-Stockfish prototype: human-style move cards.

Given a FEN (or a PGN movetext), runs SF with MultiPV=3 at a modest node
budget and emits the kind of output the Gemini grandmaster-prompt produces —
candidate lines in SAN, win-probability (via the empirically fitted
eval->score curve from lichess depth-18 data), and a rule-based English
rationale built from legible features (material, checks, captures, passed
pawns, king safety proxies) — at ~1e-7 the cost of an LLM call.

Usage:
    python explain_move.py "<FEN>" [nodes]
    python explain_move.py --pgn "1. e4 c6 2. Bc4 d5 ..." [nodes]
"""
import io
import json
import math
import sys

import chess
import chess.engine
import chess.pgn

SF = "/usr/games/stockfish"
K_CP = 350.0  # overwritten from results/analysis.json when present


def load_k():
    global K_CP
    try:
        with open("results/analysis.json") as fh:
            K_CP = json.load(fh)["eval_to_score"]["k_cp"]
    except Exception:
        pass


def win_prob(cp):
    return 1.0 / (1.0 + math.exp(-cp / K_CP))


def describe_line(board, pv, score_cp):
    """Rule-based English rationale from legible features of the line."""
    b = board.copy()
    notes = []
    captured = 0
    gives_check = False
    for i, mv in enumerate(pv[:8]):
        if b.is_capture(mv):
            piece = b.piece_at(mv.to_square)
            val = {chess.PAWN: 1, chess.KNIGHT: 3, chess.BISHOP: 3,
                   chess.ROOK: 5, chess.QUEEN: 9}.get(piece.piece_type if piece else chess.PAWN, 1)
            captured += val if i % 2 == 0 else -val
        b.push(mv)
        if i == 0 and b.is_check():
            gives_check = True
    first = board.san(pv[0])
    if gives_check:
        notes.append("gives check, forcing the reply")
    if captured > 0:
        notes.append(f"wins material (≈{captured} pawn-units over the line)")
    elif captured < 0:
        notes.append(f"sacrifices material (≈{-captured} pawn-units) for initiative")
    mv = pv[0]
    if board.is_castling(mv):
        notes.append("castles, securing the king")
    pc = board.piece_at(mv.from_square)
    if pc and pc.piece_type == chess.PAWN and chess.square_rank(mv.to_square) in (0, 7):
        notes.append("promotes")
    center = {chess.D4, chess.E4, chess.D5, chess.E5}
    if mv.to_square in center:
        notes.append("fights for the center")
    if not notes:
        notes.append("improves piece activity / keeps the tension")
    return first, "; ".join(notes)


def explain(board, nodes=16384, k=3):
    eng = chess.engine.SimpleEngine.popen_uci(SF)
    eng.configure({"Threads": 1, "Hash": 16})
    infos = eng.analyse(board, chess.engine.Limit(nodes=nodes), multipv=k)
    eng.quit()
    lines = []
    for info in infos:
        pv = info.get("pv", [])
        if not pv:
            continue
        cp = info["score"].pov(board.turn).score(mate_score=2500)
        san_line = board.variation_san(pv[:8])
        first, why = describe_line(board, pv, cp)
        lines.append({
            "move": first,
            "line": san_line,
            "eval_pawns": round(cp / 100, 2),
            "win_prob_stm": round(win_prob(cp), 3),
            "reasoning": why,
        })
    best = lines[0]
    return {
        "candidates": lines,
        "reasoning": (f"Best is {best['move']} ({best['eval_pawns']:+.2f}, "
                      f"{best['win_prob_stm']:.0%} expected score): {best['reasoning']}. "
                      + (f"Alternatives: "
                         + "; ".join(f"{l['move']} ({l['eval_pawns']:+.2f}) — {l['reasoning']}"
                                     for l in lines[1:]) if len(lines) > 1 else "")),
        "move": best["move"],
    }


def main():
    load_k()
    args = sys.argv[1:]
    if args and args[0] == "--pgn":
        game = chess.pgn.read_game(io.StringIO(args[1]))
        board = game.board()
        for mv in game.mainline_moves():
            board.push(mv)
        nodes = int(args[2]) if len(args) > 2 else 16384
    else:
        board = chess.Board(args[0]) if args else chess.Board()
        nodes = int(args[1]) if len(args) > 1 else 16384
    print(json.dumps(explain(board, nodes), indent=2))


if __name__ == "__main__":
    main()
