"""Small-LLM verbalizer: turn engine EVIDENCE into faithful English.

The hybrid the user wants: cheap chess strength + engine-grade facts from
Stockfish/chess-GPT/probes, with a tiny general LLM (Qwen2.5-0.5B-Instruct on
CPU) only doing the natural-language surface realization. Because the LLM is
handed the actual numbers (it does not choose the move or invent evals), the
explanation is faithful by construction — unlike Gemini, which picks a move and
then rationalizes (possibly lying).

Evidence bundle (all machine-produced upstream):
  - best move (SAN) + eval (pawns) + win prob
  - candidate lines with SF `Explain` effort % (or chess-GPT policy %)
  - dominant classical eval-term deltas from coach.py (king safety, passed, ...)
  - optional probe readout (board squares the model is most/least sure about)

The LLM is instructed to ONLY restate provided facts. ~0.5B params, CPU,
~100-1000x cheaper than a frontier LLM call.
"""
import sys

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

MODEL_DIR = "models/minicpm5-1b"

SYSTEM = (
    "You are a chess commentator. You will be given MACHINE-VERIFIED FACTS about "
    "a position (the engine's chosen move, its evaluation, candidate lines, and "
    "which positional factors changed). Write 2-3 sentences of natural, "
    "human-friendly commentary that ONLY restates and connects these facts. "
    "Do NOT invent moves, evaluations, or claims not in the facts. Be concise."
)


class Verbalizer:
    def __init__(self, model_dir=MODEL_DIR):
        self.tok = AutoTokenizer.from_pretrained(model_dir, trust_remote_code=True)
        self.model = AutoModelForCausalLM.from_pretrained(
            model_dir, torch_dtype=torch.float32, trust_remote_code=True)
        self.model.eval()

    def verbalize(self, evidence: dict, max_new_tokens=140):
        facts = self._format_evidence(evidence)
        user = SYSTEM + "\n\nFACTS:\n" + facts + "\n\nWrite the commentary."
        # Prefer the tokenizer's own chat template; fall back to MiniCPM's
        # native <用户>...<AI> format (its tokenizer ships no HF template).
        try:
            text = self.tok.apply_chat_template(
                [{"role": "user", "content": user}],
                tokenize=False, add_generation_prompt=True)
        except Exception:
            text = f"<用户>{user}<AI>"
        inp = self.tok(text, return_tensors="pt")
        with torch.no_grad():
            out = self.model.generate(**inp, max_new_tokens=max_new_tokens,
                                      do_sample=False, temperature=None, top_p=None,
                                      pad_token_id=self.tok.eos_token_id)
        gen = self.tok.decode(out[0, inp.input_ids.shape[1]:],
                              skip_special_tokens=True)
        return gen.strip()

    @staticmethod
    def _format_evidence(e):
        lines = []
        if "move" in e:
            lines.append(f"- Engine's move: {e['move']}")
        if "eval_pawns" in e:
            lines.append(f"- Evaluation after the move: {e['eval_pawns']:+.2f} pawns "
                         f"(positive = the side to move is better)")
        if "win_prob" in e:
            lines.append(f"- Expected score for the mover: {e['win_prob']:.0%}")
        if e.get("candidates"):
            cs = "; ".join(f"{c['move']} (search effort {c.get('effort','?')}%, "
                           f"eval {c.get('eval_pawns','?')})" for c in e["candidates"])
            lines.append(f"- Candidate moves the engine weighed: {cs}")
        if e.get("term_deltas"):
            td = "; ".join(f"{k} {v:+.2f}" for k, v in e["term_deltas"].items())
            lines.append(f"- Positional factors that changed (pawns): {td}")
        if e.get("probe_note"):
            lines.append(f"- Model internal-state note: {e['probe_note']}")
        return "\n".join(lines)


if __name__ == "__main__":
    v = Verbalizer()
    # example evidence bundle (as produced by coach.py + Explain patch)
    ev = {
        "move": "c5",
        "eval_pawns": 0.9,
        "win_prob": 0.72,
        "candidates": [
            {"move": "c5", "effort": 76, "eval_pawns": 0.9},
            {"move": "Kf2", "effort": 15, "eval_pawns": 0.4},
            {"move": "Rd1", "effort": 4, "eval_pawns": 0.3},
        ],
        "term_deltas": {"Passed pawn": 0.39, "King safety": 0.08, "Material": -0.06},
        "probe_note": "the model is confident about all 64 squares (board fully tracked)",
    }
    print("EVIDENCE:")
    print(Verbalizer._format_evidence(ev))
    print("\nVERBALIZED:")
    print(v.verbalize(ev))
