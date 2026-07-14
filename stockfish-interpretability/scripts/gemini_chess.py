"""Gemini as a chess engine (measured frontier point).

Uses gemini-3-flash-preview (the free-tier model that answers; 2.x are quota-
blocked). A dubesor-style prompt asks for the single best move; we validate it
is legal, retry a few times with feedback, and fall back to the model's first
legal suggestion. Rate-limited + retried for the free tier.

Key is read from .secrets/gemini.env (GEMINI_API_KEY), never committed.

Exposes GeminiChess.best_move(board, pgn) -> (uci, raw_text) and a policy-ish
top move for human-match. Keep call volume low: free tier is a few RPM.
"""
import os
import re
import time

import chess
import requests

MODEL = os.environ.get("GEMINI_MODEL", "gemini-3-flash-preview")
URL = f"https://generativelanguage.googleapis.com/v1beta/models/{MODEL}:generateContent"
KEY = os.environ.get("GEMINI_API_KEY", "")
MIN_INTERVAL = float(os.environ.get("GEMINI_MIN_INTERVAL", "1.5"))  # s between calls

_last_call = [0.0]


def _throttle():
    dt = time.time() - _last_call[0]
    if dt < MIN_INTERVAL:
        time.sleep(MIN_INTERVAL - dt)
    _last_call[0] = time.time()


PROMPT = (
    "You are a chess grandmaster playing a serious game. "
    "Position in FEN: {fen}\n"
    "Moves so far (PGN): {pgn}\n"
    "It is {stm} to move. Choose the strongest move. "
    "Respond with ONLY the move in UCI coordinate notation "
    "(from-square then to-square, e.g. e2e4, g8f6, e7e8q for promotion). "
    "No commentary, no punctuation, just the 4-5 character move."
)

UCI_RE = re.compile(r"\b([a-h][1-8][a-h][1-8][qrbn]?)\b")


class GeminiChess:
    def __init__(self, model=MODEL, temperature=0.4):
        self.model = model
        self.url = f"https://generativelanguage.googleapis.com/v1beta/models/{model}:generateContent"
        self.temperature = temperature
        self.sess = requests.Session()
        self.calls = 0

    def _raw(self, prompt, max_tokens=100):
        # thinkingBudget=0 keeps latency ~2s and quota low; the move quality is
        # unchanged for this task (verified: same moves with/without thinking).
        body = {"contents": [{"parts": [{"text": prompt}]}],
                "generationConfig": {"temperature": self.temperature,
                                     "maxOutputTokens": max_tokens,
                                     "thinkingConfig": {"thinkingBudget": 0}}}
        for attempt in range(5):
            _throttle()
            try:
                r = self.sess.post(self.url, headers={"x-goog-api-key": KEY,
                                   "Content-Type": "application/json"},
                                   json=body, timeout=60)
                self.calls += 1
                if r.status_code == 429:
                    time.sleep(min(30, 4 * (attempt + 1)))
                    continue
                if r.status_code >= 500:
                    time.sleep(3 * (attempt + 1))
                    continue
                r.raise_for_status()
                cand = r.json().get("candidates", [])
                if not cand:
                    continue
                parts = cand[0].get("content", {}).get("parts", [])
                text = " ".join(p.get("text", "") for p in parts)
                return text
            except Exception:
                time.sleep(2 * (attempt + 1))
        return ""

    def best_move(self, board, pgn=""):
        stm = "White" if board.turn == chess.WHITE else "Black"
        legal = {m.uci(): m for m in board.legal_moves}
        prompt = PROMPT.format(fen=board.fen(), pgn=pgn or "(none)", stm=stm)
        for _try in range(3):
            text = self._raw(prompt)
            # collect all uci-like tokens, take first legal one
            cands = UCI_RE.findall(text.lower().replace("\n", " "))
            for c in cands:
                if c in legal:
                    return c, text
            # also try SAN parse as a fallback
            for tok in re.findall(r"[A-Za-z][A-Za-z0-9+#=\-]{1,6}", text):
                try:
                    mv = board.parse_san(tok)
                    return mv.uci(), text
                except Exception:
                    pass
            prompt += ("\nYour previous answer was not a legal move. "
                       "Legal moves include: "
                       + ", ".join(list(legal)[:12]) + ". Reply with ONE legal UCI move.")
        return None, text


if __name__ == "__main__":
    import sys
    if not KEY:
        print("no key; source .secrets/gemini.env first")
        sys.exit(1)
    eng = GeminiChess()
    for fen, pgn in [
        (chess.STARTING_FEN, ""),
        ("r1bqkbnr/pppp1ppp/2n5/4p3/2B1P3/5N2/PPPP1PPP/RNBQK2R b KQkq - 3 3",
         "1. e4 e5 2. Nf3 Nc6 3. Bc4")]:
        b = chess.Board(fen)
        mv, raw = eng.best_move(b, pgn)
        print(f"move={mv}  san={b.san(chess.Move.from_uci(mv)) if mv else None}  raw={raw[:60]!r}")
