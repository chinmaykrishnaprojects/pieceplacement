"""Quantify how human-like Stockfish is, as a function of search depth.

For a sample of quiet positions where we know the move a human actually played
(from lichess), measure at several node budgets:
  - top1_match: fraction where SF's chosen move == the human move
  - in_top3:    fraction where the human move is in SF's top-3 (MultiPV)
  - human_cploss: mean centipawn loss of the human move vs SF's best (a proxy
    for "how close to optimal humans play")

The Maia / chess-LLM claim is that they predict human moves better than a strong
engine. This measures the SF baseline curve those models must beat, and tests
whether *weak* (low-node) Stockfish is actually more human-like than strong SF.

Output: results/humanlike.csv + results/humanlike.json
"""
import csv
import json
import sys

import chess
import chess.engine
import pandas as pd

SF = "/usr/games/stockfish"
IN = sys.argv[1] if len(sys.argv) > 1 else "data/fen_moves.csv.gz"
N_POS = int(sys.argv[2]) if len(sys.argv) > 2 else 4000
OUT = sys.argv[3] if len(sys.argv) > 3 else "results/humanlike"
NODE_LEVELS = [32, 256, 4096, 65536]


def main():
    df = pd.read_csv(IN)
    df = df.sample(min(N_POS, len(df)), random_state=1).reset_index(drop=True)
    df["mover_elo"] = pd.to_numeric(df.mover_elo, errors="coerce").fillna(0).astype(int)

    eng = chess.engine.SimpleEngine.popen_uci(SF)
    eng.configure({"Threads": 1, "Hash": 32})

    rows = []
    for i, r in df.iterrows():
        board = chess.Board(r.fen)
        human = chess.Move.from_uci(r.human_move)
        if human not in board.legal_moves:
            continue
        rec = {"elo": int(r.mover_elo)}
        for nodes in NODE_LEVELS:
            infos = eng.analyse(board, chess.engine.Limit(nodes=nodes), multipv=3)
            best_cp = infos[0]["score"].pov(board.turn).score(mate_score=2500)
            top_moves = [inf["pv"][0] for inf in infos if inf.get("pv")]
            rec[f"match{nodes}"] = int(top_moves and top_moves[0] == human)
            rec[f"top3_{nodes}"] = int(human in top_moves)
            # eval of the human move at this budget (root eval of resulting pos)
            board.push(human)
            hinfo = eng.analyse(board, chess.engine.Limit(nodes=nodes))
            human_cp = -hinfo["score"].pov(board.turn).score(mate_score=2500)
            board.pop()
            rec[f"cploss{nodes}"] = max(0, best_cp - human_cp)
        rows.append(rec)
        if (i + 1) % 500 == 0:
            print(f"{i+1}/{len(df)} positions", flush=True)
    eng.quit()

    out = pd.DataFrame(rows)
    out.to_csv(OUT + ".csv", index=False)
    summ = {"n": len(out)}
    for nodes in NODE_LEVELS:
        summ[f"top1_match_{nodes}"] = float(out[f"match{nodes}"].mean())
        summ[f"in_top3_{nodes}"] = float(out[f"top3_{nodes}"].mean())
        summ[f"human_cploss_{nodes}"] = float(out[f"cploss{nodes}"].mean())
    # by rating band, top-1 match at each level
    bands = [(0, 1400, "u1400"), (1400, 1800, "1400-1800"),
             (1800, 2200, "1800-2200"), (2200, 4000, "2200+")]
    summ["by_band"] = {}
    for lo, hi, name in bands:
        m = out[(out.elo >= lo) & (out.elo < hi)]
        if len(m) > 50:
            summ["by_band"][name] = {"n": int(len(m)),
                **{f"top1_{nd}": float(m[f"match{nd}"].mean()) for nd in NODE_LEVELS}}
    json.dump(summ, open(OUT + ".json", "w"), indent=2)
    print(json.dumps(summ, indent=2))


if __name__ == "__main__":
    main()
