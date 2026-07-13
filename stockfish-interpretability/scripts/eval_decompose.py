"""Explain Stockfish's opaque NNUE eval with its own human-readable terms.

SF16 is the LAST version that still ships the classical, hand-crafted
evaluation (Material, Imbalance, Pawns, Knights, Bishops, Rooks, Queens,
Mobility, King safety, Threats, Passed, Space, Winnable). The engine plays by
NNUE, but `eval` still prints the classical term table. We:

  1. parse the classical term table (White-POV total, MG value) for many FENs,
  2. record the NNUE eval and the final (played) eval,
  3. fit NNUE ~ classical terms to see which HUMAN concepts explain the
     black-box number, and how much variance is legible.

This is the "genuinely useful for humans" bridge: it labels *why* NNUE likes a
position in the vocabulary a coach uses.

Output: results/eval_terms.csv + results/eval_decompose.json
"""
import json
import re
import subprocess
import sys

import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression

SF = sys.argv[1] if len(sys.argv) > 1 else \
    "/home/user/stockfish-interp/src/stockfish-16/src/stockfish"
IN = sys.argv[2] if len(sys.argv) > 2 else "data/fen_moves.csv.gz"
N = int(sys.argv[3]) if len(sys.argv) > 3 else 2500
OUT = sys.argv[4] if len(sys.argv) > 4 else "results/eval_decompose"

TERMS = ["Material", "Imbalance", "Pawns", "Knights", "Bishops", "Rooks",
         "Queens", "Mobility", "King safety", "Threats", "Passed", "Space",
         "Winnable"]
ROW_RE = {t: re.compile(r"\|\s*" + re.escape(t) +
                        r"\s*\|.*\|.*\|\s*(-?\d+\.\d+)\s+(-?\d+\.\d+)\s*\|")
          for t in TERMS}
NNUE_RE = re.compile(r"NNUE evaluation\s+([+-]?\d+\.\d+)")
FINAL_RE = re.compile(r"Final evaluation\s+([+-]?\d+\.\d+)")


def eval_position(fen):
    p = subprocess.run([SF], input=f"position fen {fen}\neval\nquit\n",
                       capture_output=True, text=True, timeout=20)
    out = p.stdout
    if "in check" in out:
        return None
    row = {}
    for t, rx in ROW_RE.items():
        m = rx.search(out)
        if not m:
            return None
        row[t] = float(m.group(1))  # MG total, White POV
    n = NNUE_RE.search(out)
    f = FINAL_RE.search(out)
    if not n or not f:
        return None
    row["NNUE"] = float(n.group(1))
    row["Final"] = float(f.group(1))
    return row


def main():
    df = pd.read_csv(IN).sample(N, random_state=3).reset_index(drop=True)
    rows = []
    for i, fen in enumerate(df.fen):
        r = eval_position(fen)
        if r:
            rows.append(r)
        if (i + 1) % 500 == 0:
            print(f"{i+1}/{len(df)} ({len(rows)} parsed)", flush=True)
    data = pd.DataFrame(rows)
    data.to_csv(OUT + ".csv", index=False)

    X = data[TERMS].values
    y = data["NNUE"].values
    reg = LinearRegression().fit(X, y)
    r2_full = reg.score(X, y)
    # material-only baseline
    r2_mat = LinearRegression().fit(data[["Material", "Imbalance"]].values, y).score(
        data[["Material", "Imbalance"]].values, y)
    # positional-only (drop material)
    posc = [t for t in TERMS if t not in ("Material", "Imbalance")]
    r2_pos = LinearRegression().fit(data[posc].values, y).score(data[posc].values, y)
    # incremental R^2 of each term over material
    base = LinearRegression().fit(data[["Material", "Imbalance"]].values, y)
    resid = y - base.predict(data[["Material", "Imbalance"]].values)
    incr = {}
    for t in posc:
        incr[t] = float(LinearRegression().fit(data[[t]].values, resid).score(
            data[[t]].values, resid))
    out = {
        "n": int(len(data)),
        "r2_all_terms": float(r2_full),
        "r2_material_only": float(r2_mat),
        "r2_positional_only": float(r2_pos),
        "coefs": {t: float(c) for t, c in zip(TERMS, reg.coef_)},
        "incremental_r2_over_material": dict(
            sorted(incr.items(), key=lambda kv: -kv[1])),
        "corr_nnue_final": float(np.corrcoef(data.NNUE, data.Final)[0, 1]),
    }
    json.dump(out, open(OUT + ".json", "w"), indent=2)
    print(json.dumps(out, indent=2))


if __name__ == "__main__":
    main()
