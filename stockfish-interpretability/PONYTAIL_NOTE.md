# On the "ponytail" skill

`ponytail` (DietrichGebert/ponytail on GitHub) is a community Claude Code
*skill*, not one installed in this environment, so it could not be invoked via
the Skill tool. Fetched from the web, its core is a 7-rung YAGNI decision
ladder applied before writing any code:

1. Does this need to exist at all? (skip if not)
2. Already in this codebase? (reuse)
3. Stdlib does it? (use it)
4. Native platform feature? (use it)
5. Installed dependency? (use it)
6. One line? (write the one line)
7. Only then: minimum viable implementation.
   — never cutting validation, error handling, security, or accessibility.

Applied here retroactively/going forward: the engines lean on the stock
Stockfish binary and python-chess rather than reimplementing search or PGN
parsing (rungs 3-5); the C++ interpretability patch is ~15 lines reusing the
existing node counter instead of a parallel accounting system (rung 6); the
SPRT harness reuses cutechess-style pairing logic rather than a framework.
