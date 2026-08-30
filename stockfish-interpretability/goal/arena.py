"""Arena: the UNGAMEABLE evaluator for candidate chess players.

Contract for a candidate file (candidates/<name>.py):

    def create_player(oracle, budget):
        '''Return an object with .play(board, pgn_prefix) -> chess.Move'''

The candidate gets ONLY:
  - `oracle`: a PolicyOracle wrapping the frozen chess-GPT. Every call to
    oracle.policy(board, pgn) costs 1 unit and is COUNTED BY THE HARNESS.
    When the per-move quota is spent, further calls raise BudgetExhausted.
  - `budget`: the per-move quota (LM forward passes).
  - Free: python-chess, arithmetic, its own heuristics (material, mobility...).
    Those are not the bottleneck and stay human-legible.

Why this is hard to game:
  * The candidate never loads the model or picks the opponent.
  * Compute is metered inside the oracle, not self-reported.
  * Elo comes from real games vs Stockfish at fixed nodes, scored here.
  * Source is scanned for escape hatches (subprocess/net/engine imports).
  * Illegal move or budget overrun = forfeit that game.
  * Final ranking uses a HELD-OUT opening set the agents never see.

Fitness = Elo at a given budget. You cannot fake winning games.
"""
import argparse
import ast
import importlib.util
import json
import math
import random
import re
import sys
import time

import chess
import chess.engine

sys.path.insert(0, "scripts")

SF_PATHS = ["/usr/games/stockfish", "/usr/bin/stockfish", "stockfish"]
MAX_PLIES = 130
# Elo anchors for the Stockfish node rungs, measured earlier in this project.
ANCHOR = {32: 1468, 64: 1520, 256: 1561, 1024: 1927}

# Escape-hatch detection. Done on the AST, not the raw text: a regex both misses
# real evasion (string-built imports) and false-positives on docstrings that
# merely mention "eval". We flag banned IMPORTS and banned CALLS only.
BANNED_MODULES = {
    "subprocess", "socket", "urllib", "urllib2", "requests", "httpx", "http",
    "multiprocessing", "ctypes", "importlib", "os", "sys", "shutil", "pathlib",
    "pickle", "marshal", "builtins", "runpy", "tempfile", "glob", "io",
}
BANNED_CALLS = {"eval", "exec", "open", "__import__", "compile", "globals",
                "locals", "vars", "input", "memoryview"}


def scan_source(src):
    """Return a reason string if the candidate uses an escape hatch, else None."""
    try:
        tree = ast.parse(src)
    except SyntaxError as e:
        return f"syntax error: {e}"
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for al in node.names:
                root = al.name.split(".")[0]
                if root in BANNED_MODULES or al.name.startswith("chess.engine"):
                    return f"import {al.name}"
        elif isinstance(node, ast.ImportFrom):
            mod = node.module or ""
            root = mod.split(".")[0]
            if root in BANNED_MODULES or mod.startswith("chess.engine"):
                return f"from {mod} import ..."
            if mod == "chess" and any(a.name == "engine" for a in node.names):
                return "from chess import engine"
        elif isinstance(node, ast.Call):
            f = node.func
            if isinstance(f, ast.Name) and f.id in BANNED_CALLS:
                return f"call to {f.id}()"
            if isinstance(f, ast.Attribute) and f.attr in ("popen", "system",
                                                           "Popen", "popen_uci"):
                return f"call to .{f.attr}()"
    return None


class BudgetExhausted(Exception):
    pass


class PolicyOracle:
    """Frozen chess-GPT policy behind a metered, per-move quota.

    A memo cache makes repeated queries of the SAME position cheap in wall time,
    but they still cost budget — the quota counts logical LM queries, so the
    fitness comparison stays honest while the tournament runs at a usable speed.
    """

    def __init__(self, gpt, cache_size=4096):
        self._gpt = gpt
        self.quota = 0
        self.used_this_move = 0
        self.total_calls = 0
        self.cache_hits = 0
        self._cache = {}
        self._cache_size = cache_size

    def new_move(self, budget):
        self.quota = budget
        self.used_this_move = 0

    def policy_with_acts(self, board, pgn_prefix="", layer=11):
        """({move: prob}, {move: activation}) — SAME forward pass, SAME 1 unit.

        The activations were already computed by the policy call; returning them
        adds no model compute. Metered identically so no candidate can get extra
        information for free.
        """
        if self.used_this_move >= self.quota:
            raise BudgetExhausted(f"per-move budget {self.quota} exhausted")
        self.used_this_move += 1
        self.total_calls += 1
        return self._gpt.policy_with_acts(board, pgn_prefix=pgn_prefix, layer=layer)

    def policy(self, board, pgn_prefix=""):
        """{move: prob} from the language model. Costs 1 unit of budget."""
        if self.used_this_move >= self.quota:
            raise BudgetExhausted(f"per-move budget {self.quota} exhausted")
        self.used_this_move += 1
        self.total_calls += 1
        key = (board.board_fen(), board.turn, pgn_prefix[-64:])
        hit = self._cache.get(key)
        if hit is not None:
            self.cache_hits += 1
            return dict(hit)
        val = self._gpt.policy(board, pgn_prefix=pgn_prefix)
        if len(self._cache) >= self._cache_size:
            self._cache.clear()
        self._cache[key] = dict(val)
        return val


def load_book(path):
    """Natural openings (UCI move lists) from make_book.py."""
    return [list(m) for m in json.load(open(path))]


def load_gpt(path="models/lichess_16layers.pt"):
    from chessgpt_local import ChessGPT
    return ChessGPT(path)


def pgn_push(pgn, board, move):
    """Extend a Karvonen-format PGN prefix with `move` played on `board`."""
    san = board.san(move)
    if board.turn == chess.WHITE:
        return pgn + f"{board.fullmove_number}.{san} "
    return pgn + f"{san} "


def new_engine():
    last = None
    for p in SF_PATHS:
        try:
            e = chess.engine.SimpleEngine.popen_uci(p)
            e.configure({"Threads": 1, "Hash": 16})
            return e
        except Exception as ex:  # noqa: BLE001
            last = ex
    raise RuntimeError(f"no stockfish: {last}")


def make_openings(n, seed, engine):
    """Balanced random openings as UCI move lists (so PGN context is rebuildable)."""
    rng = random.Random(seed)
    out = []
    while len(out) < n:
        b = chess.Board()
        ok = True
        for _ in range(8):
            moves = list(b.legal_moves)
            if not moves:
                ok = False
                break
            b.push(rng.choice(moves))
        if not ok or b.is_game_over():
            continue
        info = engine.analyse(b, chess.engine.Limit(nodes=20000))
        cp = info["score"].white().score(mate_score=10000)
        if cp is not None and abs(cp) <= 90:   # roughly balanced
            out.append([m.uci() for m in b.move_stack])
    return out


def play_game(player, oracle, budget, engine, rung, opening, player_white):
    """`opening` is a list of UCI moves; we replay it to build board + PGN."""
    board = chess.Board()
    pgn = ";"
    for u in opening:
        mv = chess.Move.from_uci(u)
        pgn = pgn_push(pgn, board, mv)
        board.push(mv)

    while not board.is_game_over(claim_draw=True) and board.ply() < MAX_PLIES:
        is_player = (board.turn == chess.WHITE) == player_white
        if is_player:
            oracle.new_move(budget)
            try:
                mv = player.play(board, pgn)
            except BudgetExhausted:
                return 0.0 if player_white else 1.0, "budget_overrun"
            except Exception:  # noqa: BLE001
                return 0.0 if player_white else 1.0, "crash"
            if mv is None or mv not in board.legal_moves:
                return 0.0 if player_white else 1.0, "illegal"
        else:
            mv = engine.play(board, chess.engine.Limit(nodes=rung)).move
        pgn = pgn_push(pgn, board, mv)
        board.push(mv)

    if not board.is_game_over(claim_draw=True):
        return 0.5, "adjudicated"
    res = board.result(claim_draw=True)
    s = {"1-0": 1.0, "0-1": 0.0}.get(res, 0.5)
    return (s if player_white else 1.0 - s), res


def elo_from_score(s, n):
    s = min(max(s, 0.5 / n), 1 - 0.5 / n)
    return -400 * math.log10(1 / s - 1)


def evaluate(cand_path, budget, games, rungs, seed, gpt=None, verbose=True,
             book=None):
    src = open(cand_path).read()
    hit = scan_source(src)
    if hit:
        return {"error": f"banned construct in candidate: {hit}"}

    spec = importlib.util.spec_from_file_location("cand", cand_path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)

    gpt = gpt or load_gpt()
    oracle = PolicyOracle(gpt)
    engine = new_engine()
    if book:
        rng = random.Random(seed)
        pool = list(book)
        rng.shuffle(pool)
        openings = pool[:max(1, games // 2)]
    else:
        openings = make_openings(max(1, games // 2), seed, engine)

    out = {"candidate": cand_path, "budget": budget, "rungs": {}, "flags": {},
           "openings": "natural" if book else "random8",
           "model": getattr(gpt, "path", "?")}
    t0 = time.time()
    for rung in rungs:
        pts = 0.0
        played = 0
        for i in range(games):
            opening = openings[(i // 2) % len(openings)]
            player = mod.create_player(oracle, budget)
            s, why = play_game(player, oracle, budget, engine, rung,
                               opening, player_white=(i % 2 == 0))
            pts += s
            played += 1
            out["flags"][why] = out["flags"].get(why, 0) + 1
        score = pts / played
        out["rungs"][str(rung)] = {
            "score": score, "games": played,
            "elo": ANCHOR[rung] + elo_from_score(score, played)}
        if verbose:
            print(f"  vs SF@{rung}: {pts}/{played} ({score:.3f}) "
                  f"-> {out['rungs'][str(rung)]['elo']:.0f} Elo", flush=True)
    engine.quit()
    elos = [v["elo"] for v in out["rungs"].values()]
    out["elo"] = sum(elos) / len(elos)
    out["lm_calls_total"] = oracle.total_calls
    out["cache_hits"] = oracle.cache_hits
    out["seconds"] = round(time.time() - t0, 1)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("candidate")
    ap.add_argument("--budget", type=int, default=1)
    ap.add_argument("--games", type=int, default=20)
    ap.add_argument("--rungs", default="32")
    ap.add_argument("--seed", type=int, default=7)
    ap.add_argument("--out", default=None)
    ap.add_argument("--model", default="models/lichess_16layers.pt")
    ap.add_argument("--book", default=None)
    a = ap.parse_args()
    res = evaluate(a.candidate, a.budget, a.games,
                   [int(x) for x in a.rungs.split(",")], a.seed,
                   gpt=load_gpt(a.model),
                   book=load_book(a.book) if a.book else None)
    print(json.dumps(res, indent=2))
    if a.out:
        with open(a.out, "a") as fh:
            fh.write(json.dumps(res) + "\n")


if __name__ == "__main__":
    main()
