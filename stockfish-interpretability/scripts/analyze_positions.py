"""Analysis of lichess depth-18 evaluated positions.

1. Eval -> expected score curve (logistic fit): what does +1.00 really mean?
2. Reverse-engineered piece values, two ways:
   a. linear regression of SF eval on material imbalance (SF's implied values)
   b. logistic regression of the *actual game result* on material imbalance
      (outcome-implied values)
3. Difference vs (log-)ratio material models, and the trade-when-ahead test:
   does a fixed material edge convert better with less material on the board,
   and does pawn share matter?

Outputs JSON of fitted params + PNG charts into results/.
"""
import gzip
import json
import sys

import numpy as np
import pandas as pd
from scipy.optimize import curve_fit
from scipy.special import expit
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.metrics import log_loss

IN = sys.argv[1] if len(sys.argv) > 1 else "data/positions.csv.gz"
PREFIX = sys.argv[2] if len(sys.argv) > 2 else "results/analysis"

PIECES = ["p", "n", "b", "r", "q"]
BASE = {"p": 1, "n": 3, "b": 3, "r": 5, "q": 9}


def load():
    df = pd.read_csv(IN)
    # quiet, middlegame-ish sample: skip opening theory plies, checks,
    # and the position right after a capture (unresolved tactics)
    df = df[(df.ply >= 12) & (df.is_check == 0) & (df.is_capture == 0)]
    for pc in PIECES:
        df[f"d{pc}"] = df[f"w{pc}"] - df[f"b{pc}"]
    df["wmat"] = sum(BASE[p] * df[f"w{p}"] for p in PIECES)
    df["bmat"] = sum(BASE[p] * df[f"b{p}"] for p in PIECES)
    df["matdiff"] = df.wmat - df.bmat
    df["mattot"] = df.wmat + df.bmat
    df["logratio"] = np.log((df.wmat + 1) / (df.bmat + 1))
    # pawn share of own material
    df["wpawnshare"] = df.wp / (df.wmat + 1e-9)
    df["bpawnshare"] = df.bp / (df.bmat + 1e-9)
    # de-correlate within games: keep every 4th recorded position per game
    df = df[df.groupby("game_id").cumcount() % 4 == 0]
    return df


def fit_eval_score(df, out):
    """Expected score = sigmoid(cp / k). SF-style: score in [0,1]."""
    d = df[np.abs(df.eval_cp) <= 1500]
    x = d.eval_cp.values.astype(float)
    y = d.result.values

    def model(x, k, b):
        return expit((x + b) / k)

    (k, b), _ = curve_fit(model, x, y, p0=[300.0, 0.0])
    # empirical curve by bins
    bins = np.arange(-1000, 1001, 100)
    d2 = d.assign(bin=pd.cut(d.eval_cp, bins))
    emp = d2.groupby("bin", observed=True).agg(cp=("eval_cp", "mean"),
                                               score=("result", "mean"),
                                               n=("result", "size"))
    out["eval_to_score"] = {
        "k_cp": float(k), "bias_cp": float(b),
        "score_at_plus100": float(model(100, k, b)),
        "cp_for_75pct": float(k * np.log(3) - b),
        "empirical_bins": emp.reset_index(drop=True).to_dict("records"),
        "n": int(len(d)),
    }
    return k, b


def piece_values(df, out):
    X = df[[f"d{p}" for p in PIECES]].values
    # (a) SF-eval implied (linear, cp)
    keep = np.abs(df.eval_cp) <= 1500
    lin = LinearRegression().fit(X[keep], df.eval_cp[keep])
    cp_vals = dict(zip(PIECES, lin.coef_.round(1)))
    # (b) outcome-implied (logistic on result; treat draws as 0.5 by
    # duplicating rows w/ win+loss? simpler: fit on win-vs-loss only)
    dec = df[df.result != 0.5]
    Xd = dec[[f"d{p}" for p in PIECES]].values
    logit = LogisticRegression(max_iter=1000).fit(Xd, (dec.result == 1.0).astype(int))
    coef = logit.coef_[0]
    pawn = coef[0]
    outcome_vals = {p: float(c / pawn) for p, c in zip(PIECES, coef)}
    out["piece_values"] = {
        "sf_eval_implied_cp": {k: float(v) for k, v in cp_vals.items()},
        "sf_eval_implied_pawn_units": {k: float(v / cp_vals["p"]) for k, v in cp_vals.items()},
        "outcome_implied_pawn_units": outcome_vals,
        "outcome_logit_per_pawn": float(pawn),
        "n_eval": int(keep.sum()), "n_decisive": int(len(dec)),
    }


def ratio_vs_diff(df, out):
    dec = df[df.result != 0.5]
    y = (dec.result == 1.0).astype(int).values
    feats = {
        "diff": dec[["matdiff"]].values,
        "logratio": dec[["logratio"]].values,
        "diff_scaled_by_total": (dec.matdiff / np.sqrt(dec.mattot)).values.reshape(-1, 1),
        "diff+interaction": np.column_stack([dec.matdiff,
                                             dec.matdiff * dec.mattot / 78.0]),
        "diff+pawnshare_scale": np.column_stack([
            dec.matdiff,
            dec.matdiff * (dec.wpawnshare + dec.bpawnshare) / 2.0]),
    }
    res = {}
    for name, X in feats.items():
        m = LogisticRegression(max_iter=1000).fit(X, y)
        ll = log_loss(y, m.predict_proba(X))
        res[name] = {"logloss": float(ll),
                     "coefs": [float(c) for c in m.coef_[0]]}
    out["ratio_vs_diff"] = res

    # trade-when-ahead: for positions where white is up exactly 2-4 points,
    # bucket by total material remaining and report conversion rate
    up = df[(df.matdiff >= 2) & (df.matdiff <= 4)]
    buckets = pd.cut(up.mattot, [0, 20, 35, 50, 65, 80])
    conv = up.groupby(buckets, observed=True).agg(
        n=("result", "size"), score=("result", "mean"),
        matdiff=("matdiff", "mean"), pawnshare=("wpawnshare", "mean"))
    out["trade_when_ahead"] = conv.reset_index(drop=True).to_dict("records")

    # same conditioned on SF eval ~ equal-ish? No: condition only on material.
    # also: among up-2..4 positions, split by *pawn share of the leader*
    ps = pd.qcut(up.wpawnshare, 4, duplicates="drop")
    conv2 = up.groupby(ps, observed=True).agg(n=("result", "size"),
                                              score=("result", "mean"))
    out["leader_pawnshare_effect"] = conv2.reset_index(drop=True).to_dict("records")


def main():
    df = load()
    out = {"n_positions": int(len(df)), "n_games": int(df.game_id.nunique())}
    fit_eval_score(df, out)
    piece_values(df, out)
    ratio_vs_diff(df, out)
    with open(PREFIX + ".json", "w") as fh:
        json.dump(out, fh, indent=2, default=str)
    print(json.dumps({k: v for k, v in out.items()
                      if k in ("n_positions", "n_games")}, indent=2))
    print("piece values (SF cp):", out["piece_values"]["sf_eval_implied_cp"])
    print("piece values (outcome, pawn=1):",
          {k: round(v, 2) for k, v in out["piece_values"]["outcome_implied_pawn_units"].items()})
    print("eval->score k =", round(out["eval_to_score"]["k_cp"], 1),
          "cp; P(win-equiv) at +100cp =", round(out["eval_to_score"]["score_at_plus100"], 3))
    print("model loglosses:", {k: round(v["logloss"], 4)
                               for k, v in out["ratio_vs_diff"].items()})


if __name__ == "__main__":
    main()
