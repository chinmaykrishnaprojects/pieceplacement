"""Measure BPE compression on a lichess PGN corpus (tokenizer efficiency).

The chess-GPT is char-level (vocab 32). A BPE tokenizer trained on PGN merges
frequent move-strings into single tokens. This quantifies the win *before*
spending GPU on a retrain: how many fewer tokens per game, i.e. how much cheaper
inference / longer context becomes for the same board coverage.

Trains a small BPE (HF tokenizers) on Karvonen-format PGN text and reports the
char/token compression ratio vs the char-level baseline.
"""
import json
import sys

import chess
import chess.pgn
import pandas as pd


def pgn_corpus_from_prefixes(path, n):
    df = pd.read_csv(path).sample(n, random_state=0)
    return [str(p) for p in df.pgn_prefix if isinstance(p, str) or p == p]


def main():
    n = int(sys.argv[1]) if len(sys.argv) > 1 else 8000
    texts = pgn_corpus_from_prefixes("data/fen_moves_pgn.csv.gz", n)
    total_chars = sum(len(t) for t in texts)

    try:
        from tokenizers import Tokenizer, models, trainers, pre_tokenizers
    except ImportError:
        import subprocess
        subprocess.run([sys.executable, "-m", "pip", "install", "-q", "tokenizers"])
        from tokenizers import Tokenizer, models, trainers, pre_tokenizers

    out = {}
    for vocab_size in (256, 1024, 4096):
        tok = Tokenizer(models.BPE())
        tok.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=False)
        trainer = trainers.BpeTrainer(vocab_size=vocab_size, show_progress=False,
                                      special_tokens=[";"])
        tok.train_from_iterator(texts, trainer)
        total_toks = sum(len(tok.encode(t).ids) for t in texts)
        out[f"vocab_{vocab_size}"] = {
            "chars_per_token": total_chars / total_toks,
            "compression_vs_char": total_chars / total_toks,  # char-level = 1 char/token
            "tokens_per_game_avg": total_toks / len(texts),
        }
        # sample merged tokens (frequent chess substrings)
        vocab = tok.get_vocab()
        merged = sorted([w for w in vocab if len(w) > 2], key=lambda w: vocab[w])[:15]
        out[f"vocab_{vocab_size}"]["sample_tokens"] = [
            w.replace("Ġ", "_") for w in merged]

    char_tokens_per_game = total_chars / len(texts)
    out["char_level_tokens_per_game"] = char_tokens_per_game
    out["n_games"] = len(texts)
    json.dump(out, open("results/bpe_compress.json", "w"), indent=2)
    print(json.dumps(out, indent=2))
    best = out["vocab_4096"]["compression_vs_char"]
    print(f"\nBPE vocab 4096: {best:.2f}x fewer tokens than char-level "
          f"-> ~{best:.1f}x cheaper inference / longer context.")


if __name__ == "__main__":
    main()
