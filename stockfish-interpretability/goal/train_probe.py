"""Train the blunder probe and answer: did the model already know?

A LINEAR probe on the layer-11 residual stream, predicting "this move loses
>=150cp". Deliberately linear: if a linear readout works, the information is
represented directly, not merely recoverable by an arbitrarily powerful decoder.

Reported against two baselines that matter:
  - majority class (predict "not a blunder")
  - the POLICY ITSELF (does a low policy probability already flag blunders?)
The probe is only interesting if it beats what the policy already tells us.

Saves results/probe.npz (weights) for the inference-time candidate.
"""
import json
import sys

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score

IN = sys.argv[1] if len(sys.argv) > 1 else "results/probe_data.npz"


def main():
    d = np.load(IN)
    X, y, prob, cp = d["X"], d["y"], d["prob"], d["cp"]
    n = len(y)
    ntr = int(0.8 * n)
    Xtr, Xte = X[:ntr], X[ntr:]
    ytr, yte = y[:ntr], y[ntr:]

    mu, sd = Xtr.mean(0), Xtr.std(0) + 1e-6
    Ztr, Zte = (Xtr - mu) / sd, (Xte - mu) / sd

    clf = LogisticRegression(max_iter=2000, C=0.05)
    clf.fit(Ztr, ytr)
    p_probe = clf.predict_proba(Zte)[:, 1]

    out = {
        "n": int(n), "layer": int(d["layer"]),
        "blunder_rate": float(y.mean()),
        "probe_auc": float(roc_auc_score(yte, p_probe)),
        "probe_acc": float((clf.predict(Zte) == yte).mean()),
        "majority_acc": float(max(yte.mean(), 1 - yte.mean())),
        # does the policy probability alone already predict blunders?
        "policy_auc": float(roc_auc_score(yte, -prob[ntr:])),
    }
    # probe + policy combined (cheap logistic on the two scalars)
    comb = np.column_stack([p_probe, prob[ntr:]])
    c2 = LogisticRegression(max_iter=1000).fit(comb, yte)
    out["probe_plus_policy_auc"] = float(
        roc_auc_score(yte, c2.predict_proba(comb)[:, 1]))

    np.savez("results/probe.npz", w=clf.coef_[0].astype(np.float32),
             b=np.float32(clf.intercept_[0]), mu=mu.astype(np.float32),
             sd=sd.astype(np.float32), layer=int(d["layer"]))
    json.dump(out, open("results/probe_metrics.json", "w"), indent=2)

    print(f"n={n}  blunder rate {out['blunder_rate']:.3f}  (layer {out['layer']})")
    print(f"  probe   AUC {out['probe_auc']:.3f}   acc {out['probe_acc']:.3f} "
          f"(majority {out['majority_acc']:.3f})")
    print(f"  policy  AUC {out['policy_auc']:.3f}   <- what the LM's own ranking says")
    print(f"  combined AUC {out['probe_plus_policy_auc']:.3f}")
    print()
    if out["probe_auc"] > out["policy_auc"] + 0.03:
        print("=> the activations carry blunder information the POLICY RANKING DOES NOT.")
        print("   The model computed it and argmax was discarding it.")
    elif out["probe_auc"] > 0.6:
        print("=> probe works, but adds little beyond the policy ranking itself.")
    else:
        print("=> no linear blunder signal at this layer. The model does not")
        print("   linearly represent 'this move hangs material' here.")


if __name__ == "__main__":
    main()
