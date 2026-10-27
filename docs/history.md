# How this project got here

Rosetta began as an experiment with language models. It ended as a standalone tool without one, because the measurements pointed that way. Every step was pre-registered or frozen before it was measured. The numbers below are the ones that decided the direction.

## 1. Spike v1: let a language model decode the bytes (failed)

An agent built on NVIDIA Nemotron (3 Ultra and 3.5 Lightning, via Nebius Token Factory) got raw `.mpr` bytes, tools to inspect them, and the task of writing a field specification for drifted EC-Lab files. It also had to name the new columns.

- **Result:** 0 of 12 cases solved. Code alone recovered the byte layout and storage types in 6 of 6 cases.
- **What failed:** naming, the one thing code cannot do. Every model answer used a placeholder name.
- **Independent audit:** five reviewers checked the run. They also found harness defects (module-relative offsets, a strict reply parser, no forced final answer), but none of them changed the outcome.
- **Lesson:** structure is code's job. Test the model only on meaning, with schema-constrained output and a benchmark large enough to carry a claim.

**Erratum from that spike:** in EC-Lab 11.50+ files the column IDs are little-endian `u2` at module offset 6, with `n_columns` at offset 4. Read this way, the drift file `gcpl.issue_230.mpr` contains the new IDs 883 and 884 (Energy ce charge/discharge). They are identical to Energy we charge/discharge in every row.

## 2. Spike v2: name the column, measured (no-go for Nemotron)

The benchmark (`spike2/`, `prereg2/`) hides one real column at a time. Code writes a dossier from the decoded data: statistics, generic relations, the known columns and sample rows. The model must answer with a name from a closed vocabulary, say DUPLICATE, or ABSTAIN.

**Design choices:**
- **Decoys:** 24 scored decoys, made of noise and random integers plus 10 *out-of-vocabulary* columns. In those, the true name is removed from the vocabulary.
- **Sham holes:** every task removes one vocabulary name, so a missing name never gives a decoy away.
- **Code baselines:** a technique prior, name elimination, and a supervised kNN trained leave-one-source-group-out.
- **Harness review:** two rounds of adversarial review before locking. They found, among other things, that the first draft could be passed by name elimination alone.

**Pre-registered rule:** GO needed at least 0.60 macro top-1, at least 5 points over the best code baseline, and at most 10 % of decoys named.

| Arm | Macro top-1 (126 tasks) | Decoys named (24) |
|---|---|---|
| Nemotron 3 Ultra (thinking, medium effort) | 0.411 | 10 |
| Nemotron 3.5 Lightning (no thinking) | 0.054 | 19 |
| code kNN baseline | 0.628 | 24 |

The result was NO-GO. Details: `results/v2_result.md`.

## 3. A stronger model on the same benchmark

Claude Opus 5.5 ran through the Claude Code CLI on the same frozen tasks. Each task was a blind call with no tools and an empty working directory (`bench/claude_cli.py`). It scored **0.969** and named 7 of 24 decoys. The task can therefore be done, and a model can do it well.

## 4. The standalone namer

The aim from here was a tool that needs no model and no API key. Most of what a model knows here is physics, and physics can be written down as code:
- **Relations** check a candidate against the other columns (for example, charge is current times time).
- **Expected unit scales** separate quantities that share a shape, such as mA·h versus C, or a potential versus a current.
- **Storage types** come from the public parser maps.
- **Shape checks** cover clocks, counters, log-spaced frequencies and set points.
- **A statistics fallback** applies only to columns that are not noise. Its guesses are vetoed when a defining relation of the guessed name fails.
- **Abstention** is the answer whenever none of the above supports a name.

It was developed in rounds. Before each round a new hold-out was frozen from public files nobody had looked at. That namer version and Claude Opus 5.5 were then run once on it (`results/open_runs.md`).

| Hold-out | namer correct | namer wrong names | Opus correct | Opus wrong names | Decoys named (namer / Opus) |
|---|---|---|---|---|---|
| 1 (48 files) | 76/79 | 0 | 78/79 | 1 | 4 / 4 of 19 |
| 2 (68 files) | 81/84 | 1 | 83/84 | 0 | 3 / 5 of 19 |
| 3 (18 files) | 50/54 | 0 | 54/54 | 0 | 2 / 2 of 19 |
| 4 (38 files) | 73/86 | 4 | 75/86 | 8 | 1 / 3 of 19 |
| 5 (38 files) | **70/70** | **0** | 69/70 | 1 | **1 / 2** of 19 |
| 6 (57 older files, one column set) | 16/20 | 1 | **20/20** | **0** | **2 / 3** of 16 |
| 7 (594 files from 222 records, 52 names) | 156/193 | 14 | **176/193** | **13** | **1 / 5** of 19 |
| **total** | 522/586 (89.1 %) | **20** | 555/586 (94.7 %) | 23 | **14 / 24** of 130 |

The namer improved between hold-outs. On hold-out 5 it matched or beat Opus on every count. Hold-out 6 is the older files that 0.1.4 made readable. There every namer miss was the current, in files with nothing to confirm it, and two fixes found afterwards bring a rerun (not blind) to 19/20 with no wrong names. Hold-out 7 is by far the broadest set: 594 files from 222 records in about 30 repositories, with 52 names in its tasks. There the namer named 81 % against Opus's 91 %, with about as many wrong names (14 against 13). Hidden one at a time, the namer named 94.4 % of all 6,467 usable columns correctly and 2.1 % wrongly. Over all seven hold-outs the namer named fewer columns than Opus. It gave slightly fewer wrong names and was fooled by far fewer decoys.

**Tried and rejected:** naming "the column that similar files always record and this one lacks". That is what an expert, or a model, often does, and it raised accuracy. It also named 10 of 19 decoys on one hold-out, so it is not used.
