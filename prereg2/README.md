# Spike v2: pre-registration (draft until LOCK.json exists and is committed)

**Question.** Code already decodes EC-Lab bytes (v1: 6/6 structure and dtype). Can Nemotron on Nebius Token Factory do the one thing code cannot, namely say *which physical quantity* an unnamed column is, better than every code-only method we know, and abstain when no name fits?

**Why v2.** Spike v1 (2026-10-13) failed: the agents had to rebuild code's byte work and named 0 of 12 columns, with several harness defects (summarised in `docs/history.md`). v2 gives the model only the semantic job, with schema-constrained output, and measures it on a benchmark large enough to carry a claim. An adversarial review of the v2 harness (`results/v2_harness_review.md`) found that the first draft could be passed by code alone. The fixes are described below.

## Benchmark (built by `python -m spike2.tasks`, seed 20261015)

- **Corpus:** 48 public EC-Lab `.mpr` files (yadg and galvani test suites, GPL-3.0, downloaded at run time, never committed). Each is decoded with yadg 7.0.1's column map, which is the answer key; names agree with galvani on every overlapping ID.
- **Task:** one real column is hidden (`TARGET`), and code writes a dossier. The dossier holds:
  - the technique code
  - TARGET's dtype and statistics
  - the strongest generic relations (each known column linearly and as a time integral)
  - the known columns under shuffled aliases `c0..cN` with their parser names
  - 12 sample rows
- **Not in the dossier:** column IDs, file names or file order.
- **Answer:** one canonical name from a closed vocabulary of 114 entries (`spike2/vocab.py`), or `DUPLICATE`, or `ABSTAIN` (no vocabulary name is justified).
  - Pre-registered synonyms score as the same name: `Ewe`/`<Ewe>`/`Ecell`, `I`/`<I>`, `Ece`/`<Ece>`, `control`/`control_V`/`control_I`, `P`/`Pwe`, `Energy charge`/`Energy we charge`, …
  - Units are recorded but not scored, because yadg's units are not always the vendor's.
- **Buckets (test split):**
  - **main, 126 tasks / 43 names:** the primary metric. Non-constant, not identical to another column, and *not settled by name elimination*.
  - **elim, 54:** settled by `spike2/elim.py`, which takes the gap in a family of sibling names that share a word (e.g. the missing harmonic `|Ewe h4|`, or NSD/NSR/THD). No data and no model are needed, so these are reported only.
  - **duplicate, 8**, and **constant, 6:** reported only.
  - **decoys, 26:** ABSTAIN is right.
    - 8 noise f4, 2 noise f8 and 4 random-integer columns inserted by code
    - **10 out-of-vocabulary columns:** a real column whose canonical name is removed from that task's vocabulary. This is the product's real risk, a quantity the vocabulary does not have.
    - 1 constant and 1 exact copy: code flags these, so they are reported only.
- **Sham holes:** every non-oov task also has one name removed from its vocabulary. The name is typical for the technique but absent from the file, so a missing name never betrays an oov decoy. A model answer naming a real EC-Lab name that the task's vocabulary leaves out is final and wrong, with no second chance.
- **Dev split:** 20 main tasks and 3 decoys, from files without drift columns. It is used only to check the harness against the real API before locking.
- **Real drift, honestly stated:** with the faithful little-endian decoding, galvani 0.5.0 lacks only three of the corpus's column IDs:
  - 883/884 (`Energy ce charge/discharge`, gcpl.issue_230): identical in every row to `Energy we charge/discharge`, so DUPLICATE is right
  - 880 (`Energy we`, mb.issue_223): the only genuinely new non-duplicate column, which is in test/main

## Arms

| Arm | Model (Token Factory) | Reasoning | Tools | Buckets on test |
|---|---|---|---|---|
| L0 | Nemotron 3.5 Lightning | off | none | all |
| LT | Nemotron 3.5 Lightning | on | none | all |
| U | Nemotron 3 Ultra | on, medium effort | none | main, decoy |
| LTA | Nemotron 3.5 Lightning | on | up to 6 hypothesis tests (`spike2/dsl.py`: the model writes an expression, code fits TARGET = a·expr + b and reports r²) | main, decoy |
| R | routed | – | LTA's answer if its confidence ≥ 0.7, otherwise U's (computed, no extra calls) | main, decoy |

Code-only baselines (`spike2/baselines.py`, answers committed in `baselines_test.jsonl`):
- **H_prior:** technique prior.
- **H_knn:** supervised kNN on column features, trained leave-one-*source-group*-out (near-copies of one experiment share a group).
- **H_elim:** name elimination.
- **H_combo:** H_elim, otherwise H_knn.

Baselines on test/main, measured before any model call:
- **H_knn and H_combo:** 0.628 macro top-1 (95 % CI 0.52–0.74)
- **H_prior:** 0.465
- **H_elim:** 0 (by construction; it solves 54/54 in the elim bucket)

**Dev outcome (15 Oct):**
- Lightning with thinking on (LT, LTA) ran past its token budget on most dev tasks, so only **L0 and U** run on test. R needs LTA and is dropped.
- Code now enforces the rule that TARGET is not an already-present quantity.
- Dev scores (not decisive): U 8/20, L0 2/20, H_knn 16/20.

## Decision rule (`thresholds.json`)

**GO** if at least one model arm meets all of these on test:
1. macro top-1 on main ≥ **0.60** (each canonical name weighted equally)
2. at least **5 points** above the best baseline (with H_knn at 0.628: **≥ 0.68**). **STRONG GO** if also at least **15 points** above it (**≥ 0.78**), which is the target. Decided by the project owner on 15 Oct, before any model call.
3. hallucination on the 24 scored decoys ≤ **10 %** (at most 2)
4. at least 95 % of main tasks and of scored decoys answered

Only status `ok` counts: a failed main task is wrong, and a failed decoy is a hallucination. **NO-GO:** Rosetta stops and the project owner chooses what to do next.

Reported, not decisive:
- micro and top-3 accuracy, and the paired bootstrap CI of the margin
- abstention rate
- the elim, duplicate and constant buckets, and decoys by kind
- the drift columns
- $ per column and latency
- thinking on vs off, and Lightning vs Ultra

**Reruns:** each task is run once. Only infrastructure errors (status `error`) may be retried, at most 2 attempts. Model failures (`failed`) are final.

## Procedure (Thu 15 Oct)

1. The project owner reviews `thresholds.json` and this file.
2. `python -m spike2.api_check`. A toy dossier, the real prompt and the real 115-value enum schema run on Lightning (thinking off/on) and Ultra (medium effort). Use json_schema only if every configuration answers validly *and* reasoning stays on; otherwise use json_object.
3. **Dev** (≈ $0.4): `python -m spike2.run --arm {L0,LT,U,LTA} --split dev --response-mode <mode>`. Only harness defects may be fixed after this step. Thresholds, tasks and baselines stay unchanged.
4. Run `python -m spike2.tasks` (hashes unchanged) and `python -m spike2.baselines --split test`. Then `python -m spike2.lock --by <name> --response-mode <mode>` and `git commit`. `run.py` refuses the test split until the hashes match and the commit is clean.
5. **Test** (≈ $3.5–4, hard caps L0 $0.2, LT $0.4, U $3.0, LTA $1.6):
   - L0 and LT on all buckets
   - U and LTA with `--buckets main,decoy`
   - `python -m spike2.run --route --split test`, then `python -m spike2.score --split test`
6. Write the decision, numbers and any deviations into `results/`.

One sample per task at temperature 1.0 (top_p 0.95).
