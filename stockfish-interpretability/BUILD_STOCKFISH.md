# Reproducing the Stockfish source build + patches

The C++ work is captured as `results/explain.patch`; the built tree itself is
not committed (34 MB source + binaries). To reproduce on Ubuntu:

```bash
# 1. Stockfish 16 source (last version with the classical eval)
echo "deb-src http://archive.ubuntu.com/ubuntu noble universe" \
  | sudo tee /etc/apt/sources.list.d/src.list
sudo apt-get update && apt-get source stockfish

# 2. Reference build — must print bench signature 2593605
cd stockfish-16/src
make -j4 profile-build ARCH=x86-64-avx2 COMP=gcc
./stockfish bench            # Nodes searched : 2593605

# 3. Apply the Explain patch (per-root-move search effort → candidate cards)
patch -p0 < /path/to/results/explain.patch
make -j4 build ARCH=x86-64-avx2 COMP=gcc
./stockfish bench            # STILL 2593605  → provably 0 Elo, no search change

# 4. Use it
printf 'setoption name Explain value true\nposition startpos\ngo nodes 200000\n' \
  | ./stockfish   # → info string candidate <mv> effort NN% score cp NN
```

`scripts/coach.py` expects a built binary at
`src/stockfish-16/src/stockfish` (for the classical `eval` term table) and the
system `/usr/games/stockfish` for play; adjust the paths at the top if needed.
