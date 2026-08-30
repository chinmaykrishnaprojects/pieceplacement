"""Orchestrator: evaluate every candidate on the arena, publish to the wiki.

Separation of concerns that keeps the loop honest and fast:
  - subagents THINK and write candidates/<name>.py (cheap, parallel, no CPU)
  - this orchestrator SCORES them on the ungameable arena (expensive, serial)
  - results land in wiki/results.jsonl, which the next generation reads

Usage:
  python orchestrate.py --budget 4 --games 10 --rungs 32          # score all new
  python orchestrate.py --only candidates/foo.py --games 20 --rungs 32,256
"""
import argparse
import glob
import json
import os
import time

import torch

import arena

RESULTS = "wiki/results.jsonl"


def already_scored(path, budget, games, tag=None):
    if not os.path.exists(RESULTS):
        return False
    key = (os.path.basename(path), budget, games, tag)
    for line in open(RESULTS):
        try:
            r = json.loads(line)
        except Exception:  # noqa: BLE001
            continue
        if (os.path.basename(r.get("candidate", "")), r.get("budget"),
                r.get("games_per_rung"), r.get("tag")) == key:
            return True
    return False


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--budget", type=int, default=4)
    ap.add_argument("--games", type=int, default=10)
    ap.add_argument("--rungs", default="32")
    ap.add_argument("--seed", type=int, default=7)
    ap.add_argument("--only", default=None)
    ap.add_argument("--threads", type=int, default=4)
    ap.add_argument("--force", action="store_true")
    ap.add_argument("--model", default="models/lichess_16layers.pt")
    ap.add_argument("--book", default=None)
    ap.add_argument("--tag", default=None)
    a = ap.parse_args()

    torch.set_num_threads(a.threads)
    rungs = [int(x) for x in a.rungs.split(",")]
    paths = [a.only] if a.only else sorted(glob.glob("candidates/*.py"))
    gpt = arena.load_gpt(a.model)
    book = arena.load_book(a.book) if a.book else None
    os.makedirs("wiki", exist_ok=True)

    for p in paths:
        if not a.force and already_scored(p, a.budget, a.games, a.tag):
            print(f"skip (scored): {p}")
            continue
        print(f"\n=== {os.path.basename(p)}  budget={a.budget} games={a.games} ===",
              flush=True)
        t0 = time.time()
        try:
            res = arena.evaluate(p, a.budget, a.games, rungs, a.seed, gpt=gpt,
                                 book=book)
        except Exception as e:  # noqa: BLE001
            res = {"candidate": p, "budget": a.budget, "error": repr(e)[:300]}
        res["games_per_rung"] = a.games
        res["tag"] = a.tag
        res["wall_seconds"] = round(time.time() - t0, 1)
        with open(RESULTS, "a") as fh:
            fh.write(json.dumps(res) + "\n")
        if "error" in res:
            print("  ERROR:", res["error"])
        else:
            print(f"  => {res['elo']:.0f} Elo   "
                  f"({res['lm_calls_total']} LM calls, "
                  f"{res.get('cache_hits',0)} cached, {res['wall_seconds']}s)")


if __name__ == "__main__":
    main()
