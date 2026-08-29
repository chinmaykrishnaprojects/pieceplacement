---
name: goal
description: Run an autonomous research loop toward a measurable objective — spawn varying subagents that propose candidate solutions, score them on an ungameable harness, and let each generation build on a shared wiki. Use when the user asks to pursue a /goal, run an autoresearch or evolutionary loop, optimize something against a hard metric, or "push the frontier" on a measurable problem.
---

# /goal — autonomous research loop against an ungameable metric

Pursue an objective by generating many candidate solutions, scoring each on a
harness the candidates cannot influence, and feeding results back to the next
generation through a shared wiki. Modeled on autoresearch / evolutionary search:
diversity in proposals, ruthless objectivity in evaluation.

## The one rule that makes this work

**The metric must be ungameable.** If an agent can score well without solving the
problem, it eventually will — not from malice, but because optimization finds the
cheapest path. Before writing any agent prompt, build the evaluator and ask:
*what is the laziest way to score well here, and does it actually achieve the goal?*

A metric is ungameable when:
- The **evaluator owns the environment.** Candidates submit a narrow artifact (one
  function, one file) conforming to an interface. They never control the opponent,
  the test set, the scoring, or the resource meter.
- **Cost is metered by the harness**, never self-reported. Wrap the expensive
  resource in an object that counts and enforces a quota.
- **Outcomes are external.** Games won, tests passed, held-out accuracy — facts
  produced by running the thing, not numbers the candidate prints.
- **Escape hatches are closed.** Scan submitted source for `subprocess`, `socket`,
  network libs, `eval`/`exec`, file reads, or imports of the reference solution.
  Reject unscored on a hit.
- **Final ranking uses held-out data** the agents never saw (a fresh seed, unseen
  test cases), so tuning to the dev set doesn't transfer.

Good fitness functions: games won vs a fixed opponent, held-out benchmark score,
wall-clock/memory under a fixed correctness bar, tests passed on hidden cases.
Bad ones: anything an LLM judges on vibes, anything the candidate self-reports,
anything with a trivial degenerate solution.

## Structure

```
<workdir>/
  arena.py          # the ungameable evaluator — agents NEVER edit this
  selftest.py       # fast mock-based correctness check agents run themselves
  orchestrate.py    # scores candidates, appends to the wiki
  candidates/       # one file per candidate; baseline_*.py is the control
  wiki/
    README.md       # goal, interface, constraints, agent notes (shared memory)
    results.jsonl   # append-only machine-written scores
```

## Procedure

**1. Define the objective and the metric together.** Write the goal as a single
sentence with a number in it. If you cannot state how it is measured, stop and
ask the user — a vague goal produces a vague loop.

**2. Build the harness before any agent runs.** Include:
- a strict candidate interface (one entry point, typed inputs/outputs)
- a resource meter that *enforces* (raises) rather than warns
- forfeit conditions: invalid output, crash, budget overrun
- a source scan for escape hatches
- a **baseline control** representing the current state of the art. Every result
  is reported as a delta against it. Score the baseline first, with enough
  samples to be a real reference, not a pilot.

**3. Build a fast self-test** with a mocked expensive resource so agents can
verify correctness in seconds without contending for CPU. Tell agents explicitly
that passing it proves validity, **not** quality.

**4. Spawn a generation of agents on *different axes*.** Diversity is the point —
3-5 agents, each assigned a genuinely distinct strategy, not variations of one
idea. In each prompt include: read the wiki first; the exact interface; the
forfeit conditions; the assigned axis with a few concrete hooks; the self-test
command; **do not run the expensive evaluator** (the orchestrator does that, and
parallel agents running it would thrash the CPU); and append findings to the wiki
before finishing. Run them in the background, in parallel.

**5. Score everything on the harness yourself**, serially, and append to
`results.jsonl`. Never let an agent report its own fitness.

**6. Feed results forward.** The next generation reads the wiki: what scored what,
and why each agent thought it would work. Instruct them to build on winners or
open a new axis — never to repeat a strategy that already failed.

**7. Validate the winner** at higher sample count on a **held-out seed**, against
the baseline, and report the delta with its uncertainty. A win inside noise is
not a win. For win/loss data, ±1 standard error on a score of *s* over *n* games
is roughly `sqrt(s(1-s)/n)`; converting to Elo, a 20-game sample is worth about
±150 Elo, so treat small gaps as ties and re-run the top two with more games.

**8. Report honestly.** State what beat the baseline, by how much, at what cost,
and what did not work. Negative results are results — record them in the wiki so
the next run doesn't repeat them.

## Cost discipline

The evaluator is usually the expensive part. Before launching a generation,
measure one candidate's evaluation time and multiply: `candidates × samples ×
per-sample cost`. If a generation exceeds the time you have, cut samples for
screening and reserve high-sample runs for finalists. Screen wide and cheap, then
validate narrow and expensive.

## Reporting

Close with a table: candidate, metric vs baseline, cost, and a one-line
description of the mechanism. Then name the single most promising next step.
