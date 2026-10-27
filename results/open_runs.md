# Open project: runs on the locked v2 benchmark

> Commit ids below refer to the private development history, where each hold-out was frozen before the namer version that was run on it. Each hold-out is reproducible: `python -m holdout.build <set>` rebuilds `data/<set>/tasks.jsonl`, whose SHA-256 is recorded in `holdout/manifest*.json`.

The tasks, dossiers, vocabulary, scorer and baselines are those frozen for spike v2 (`prereg2/LOCK.json`, commit `484d375`). The Claude runs go through the Claude Code CLI (`bench/claude_cli.py`):
- one headless call per task
- an empty working directory
- no tools, no MCP servers and no hooks
- the benchmark's own system prompt
- the same JSON schema as the Nemotron arms

The model cannot see the answer key.

## 2026-10-15: Claude Opus 5.5 (default effort), one sample per task

| Arm | Test/main macro top-1 (126 tasks, 43 names) | Scored decoys named (of 24) | Median latency |
|---|---|---|---|
| **Claude Opus 5.5** | **0.969** (122/126) | 7 (all 10 noise/random: 6 named; out-of-vocabulary: 1 of 10) | 11 s |
| Nemotron 3 Ultra (spike v2) | 0.411 | 10 | 16 s |
| H_knn (code) | 0.628 | 24 | – |
| H_prior (code) | 0.465 | 24 | – |

- **Dev split:** Opus 20/20 and 0/3 decoys named. Nemotron Ultra 8/20, H_knn 16/20.
- **Wrong on main (4):**
  - `Analog IN 2` → Temperature
  - `Analog IN 1` → abstained
  - `dQ` → (Q-Qo)
  - `(Q-Qo)` → Capacity

  Analog inputs carry arbitrary external signals, so they cannot be named from the data.
- **Decoys:** Opus named 6 of 10 noise columns and 1 of 10 out-of-vocabulary columns (`half cycle` → `Ns`). Every named decoy had confidence ≤ 0.55.
- **Exploratory, chosen on the test split, so it needs a fresh hold-out before any claim:** abstaining when confidence ≤ 0.5 gives 118/126 main correct and 1/24 decoys named.
- **Cost:** the CLI reports an API-equivalent cost of about $10.6 for the 152 test tasks. It ran on the Claude subscription.

**Next:** a fresh hold-out set (new files or columns not used in v2) to confirm the confidence rule. Then the product: a CLI or bot that reads a new `.mpr`, decodes the structure with code, names the unknown columns with evidence, and drafts a parser patch.

## 2026-10-15: standalone namer (no language model), first and only run on test so far

`namer/` (commit after `5fc477f`) works in four steps: a constant column is reported as such, an identical column as a duplicate, then name elimination, then the formula library. The library holds physical relations with expected unit scales, storage types from the public parser maps, and shape checks. Columns no relation identifies fall back to a statistics vote, which is never used for noise and must match the quantity's form. It was developed only on columns outside the test split (pool: 364 columns, 0.885 micro / 0.871 macro) and on dev (17/20, 0/3 decoys).

| Arm | Test/main macro top-1 | micro | Scored decoys named (24) |
|---|---|---|---|
| **namer (rules, no LLM)** | **0.837** | 0.857 (108/126) | **5**: 0/14 noise/random, 5/10 out-of-vocabulary |
| Claude Opus 5.5 | 0.969 | 0.968 | 7: 6/14 noise, 1/10 out-of-vocabulary |
| H_knn (code baseline) | 0.628 | 0.635 | 24 |
| Nemotron 3 Ultra | 0.411 | 0.421 | 10 |

- **Of the 18 misses, 7 are abstentions on columns that carry no identifiable physics:** Analog IN 1/2 (5) and Rcmp (2).
- **The rest:**
  - confusions between charge and energy columns
  - one counter
  - `Ece`, `Ns` and `step time`
  - the single genuinely new column, `Energy we`, where the engine abstained
- **Out-of-vocabulary decoys:** when a quantity's name is removed from the vocabulary, the engine names a close relative half the time. That is the main weakness to fix before a hold-out run.

## 2026-10-15: hold-out (48 new public files), each arm run once

The hold-out was frozen at commit `6012f92` before any development round. The namer was frozen at `1bb4ec6`. Each arm was run once, and the task hash was checked before the run.

| Arm | Main macro top-1 (79 tasks, 27 names) | micro | Wrong names on real columns | Abstained | Decoys named (19) |
|---|---|---|---|---|---|
| **namer (no LLM)** | **0.972** | 0.962 (76/79) | **0** | 3 | 4: 0/11 noise/random, 4/8 out-of-vocabulary |
| Claude Opus 5.5 | 0.981 | 0.987 (78/79) | 1 (`|Energy|` → `Energy we discharge`) | 0 | 4: 3/11 noise, 1/8 out-of-vocabulary |
| H_knn (code baseline) | 0.815 | 0.835 | – | – | 19 |
| H_prior (code baseline) | 0.667 | 0.848 | – | – | 19 |

- **Namer, abstained on real columns:** `dq`, `I` and `Capacitance charge`.
- **Overall:** the namer and Opus are close on columns never seen during development. Opus is about 1 point higher on accuracy. The namer never gives a wrong name to a real column and never names noise, but it names half of the out-of-vocabulary columns as a relative.
- **This hold-out is now spent:** further development will use it, and claims will need a new hold-out.

## 2026-10-15: hold-out 2 (68 new public files, frozen at `2c7ddf5`), each arm run once

The namer was frozen at `b6c78c9`. Its development data was the v2 corpus plus the spent hold-out 1.

| Arm | Main top-1 (84 tasks, 21 names) | Wrong names on real columns | Abstained | Decoys named (19) |
|---|---|---|---|---|
| **namer (no LLM)** | **0.964** (81/84) | 1 (`I` → `Ece`) | 2 (`I`, `I Range`) | **3**: 0/11 noise/random, 3/8 out-of-vocabulary |
| Claude Opus 5.5 | 0.988 (83/84) | 0 | 1 (`Rcmp`) | 5: 5/11 noise, 0/8 out-of-vocabulary |
| H_prior (code baseline) | 1.000 | – | – | 19 |
| H_knn (code baseline) | 0.869 | – | – | 19 |

**On this hold-out, "which typical column is missing" alone solves every real column.** The files come from a few experiments with standard column sets, so H_prior scores 1.000. It also names every decoy.

**Both hold-outs together (163 real columns, 38 scored decoys, never seen during development):**

| Arm | Correct | Wrong names on real columns | Decoys named |
|---|---|---|---|
| namer | 157/163 (96.3 %) | 1 | 7/38: 0/22 noise, 7/16 out-of-vocabulary |
| Claude Opus 5.5 | 161/163 (98.8 %) | 1 | 9/38: 8/22 noise, 1/16 out-of-vocabulary |

**Verdict:** the standalone namer is close to Opus without any model. Opus names about 2.5 points more real columns. The namer never names noise, but it names out-of-vocabulary columns as relatives about half the time. Neither is better on every axis yet.

## 2026-10-15: hold-out 3 (18 new public files, Zenodo 17244893, frozen at `5d343a4`), each arm run once

The namer was frozen at `999c481`.

| Arm | Main top-1 (54 tasks, 21 names) | Wrong names on real columns | Abstained | Decoys named (19) |
|---|---|---|---|---|
| namer (no LLM) | 0.917 macro, 0.926 micro (50/54) | 0 | 4 (`half cycle` ×2, `dq`, `Phase(Z)`) | 2: 0/11 noise, 2/8 out-of-vocabulary |
| **Claude Opus 5.5** | **1.000** (54/54) | 0 | 0 | 2: 1/11 noise, 1/8 out-of-vocabulary |
| H_knn | 0.845 | – | – | 19 |
| H_prior | 0.714 | – | – | 19 |

**Opus wins this hold-out.** All of the namer's misses are abstentions, and the next round starts from them.

## 2026-10-15: hold-out 4 (38 new public files, Zenodo 18925051, commercial solid-state batteries; frozen at `5d343a4`), each arm run once

The namer was frozen at `bc323dd`.

| Arm | Main top-1 (86 tasks, 22 names) | Wrong names on real columns | Abstained | Decoys named (19) |
|---|---|---|---|---|
| namer (no LLM) | 0.849 (73/86) | **4** (`(Q-Qo)` → `dQ`) | 9 (Capacitance ×5, `I` ×4) | **1** (random integers) |
| Claude Opus 5.5 | 0.872 (75/86) | 8 | 3 | 3 |

**All four hold-outs, each column never seen in development (303 real columns, 76 scored decoys):**

| Arm | Correct | Wrong names on real columns | Decoys named | All errors |
|---|---|---|---|---|
| namer | 280/303 (92.4 %) | **5** | **10/76** | 33 |
| Claude Opus 5.5 | 290/303 (95.7 %) | 9 | 14/76 | 27 |

**Verdict:** Opus names more columns. The namer is more trustworthy: it gives about half as many wrong names, and it is fooled by fewer decoys.

## 2026-10-15: hold-out 5 (six more CC BY Zenodo records, frozen at `7135d6c`), each arm run once

The namer was frozen at `7d8e38a`.

| Arm | Main top-1 (70 tasks, 25 names) | Wrong names on real columns | Decoys named (19) |
|---|---|---|---|
| **namer (no LLM)** | **1.000** (70/70) | **0** | **1** (out-of-vocabulary) |
| Claude Opus 5.5 | 0.986 (69/70) | 1 (`Analog IN 2` → `Temperature`) | 2 |
| H_prior | 0.786 | – | 19 |
| H_knn | 0.657 | – | 19 |

**On this fresh hold-out the latest namer beats Opus on all three counts.**

**All five hold-outs, each run once, with the namer as frozen before each one (373 real columns, 95 scored decoys):**

| Arm | Correct | Wrong names on real columns | Decoys named |
|---|---|---|---|
| namer (versions 1→5) | 350/373 (93.8 %) | **5** | **11/95** |
| Claude Opus 5.5 | 359/373 (96.2 %) | 10 | 16/95 |

**Development rounds and what each fixed:**
- Glitch rows at step transitions: a robust fit.
- EC-Lab totals that reset per cycle: energy, charge and capacitance.
- Linked techniques that reset Qo.
- Units, so that potentials are not mistaken for currents, charges or powers.
- Counters must be whole numbers.
- The counter-electrode channel.
- Phase angles.
- A clock that resets itself.

**Tried and rejected:** naming "the column that similar files always record" named up to 10/19 decoys.

**What remains unnamed:**
- columns the instrument only processes (capacitance held per segment)
- noise-like DC currents in impedance records
- external analog inputs without a sibling

## 2026-10-16: files attached to parser issues (a check, not a hold-out)

Fourteen `.mpr` files attached to issues in galvani (#42, #64, #67, #73, #74, #87) and yadg (#229, #270) were downloaded at run time and not committed. None of them had been used before. Five (galvani #87) are unreadable by Rosetta, galvani and yadg alike. In each of the other ten, every known column was hidden in turn (`--self-check`).

| namer | Hidden columns | Named correctly | Wrong names |
|---|---|---|---|
| as released (v0.1.0) | 130 | 103 | 2 |
| with the two fixes below | 130 | 105 | 1 |

The fixes were made after looking at these files, so the second row is not a blind result:
- **Ewe named as I in CC-CV cycling** (yadg #270). There the control column holds mA in CC steps and V in CV steps. A potential that sits on the V set point through a long CV hold counted as a current held at its set point. With a mixed `V/mA` control column, that evidence now also needs the column to pass the current-unit check.
- **Ewe-Ece had no relation** (galvani #74). Ewe − Ece now names it. This counts only where subtracting Ece explains the column clearly better than Ewe alone, because Ece is often nearly flat.

Every answer on the five hold-outs and the v2 benchmark is unchanged (`python -m namer.golden check`). Only confidences and alternatives moved. The remaining wrong name is `|Energy|` → `Energy we` (galvani #42).

## 2026-10-27: hold-out 6 (57 older EC-Lab files, data module version 0; frozen at `a2a6905`), namer run once

- **The files:** Zenodo 3631173 and 3631156 (PEM fuel cells, CV and LSV), the same records as hold-out 3. They were fetched then but could not be decoded, so hold-out 3 used only Zenodo 17244893. rosetta-eclab 0.1.4 reads version 0, and the namer had never run on these files. The reader was developed on their byte layout: column IDs and offsets, checked against galvani 0.5.0.
- **Answer key:** there are no vendor exports in these records. Names come from yadg's column map, and values from galvani's version-0 offsets (byte-identical to galvani 0.5.0 on all 57 files).
- **A weak hold-out:** all 57 files share one column set (time, control_V, Ewe, `<I>`, cycle number), likely from one lab, and the technique is not read from version-0 files. The set has 44 tasks: 20 main (4 per name), 6 constant and 18 decoys, of which 16 are scored. Only 5 decoys are out-of-vocabulary, because the set has only five names.
- **Namer** frozen at `a2a6905` (0.1.4 behaviour) and run once after the task hash was checked. The Claude arm (Opus 5.5, default effort, `bench/claude_cli.py`) was run once later the same night on the same tasks. It cost about $3.35 API-equivalent on the subscription.

| Arm | Main top-1 (20 tasks, 5 names) | Wrong names on real columns | Abstained | Decoys named (16) |
|---|---|---|---|---|
| **namer (no LLM)** | **0.800** (16/20) | 1 (`I` → `(Q-Qo)`) | 3 (`I`) | **2** (out-of-vocabulary, both → `(Q-Qo)`) |
| Claude Opus 5.5 | **1.000** (20/20) | **0** | 0 | 3 (noise → `Ece`, confidence ≤ 0.42) |
| H_knn | 0.950 | – | – | 16 |
| H_prior | 0.600 | – | – | 16 |

- time, control, Ewe and cycle number: 16/16 correct. All 6 constant cycle-number columns got ABSTAIN.
- **Every miss is the current `<I>`.** These CV and LSV files hold no charge or power column, and the control column is a potential, so no formula can confirm a current. Three times the statistics suggested `I` but the column lacked its form (shape or storage type), and the namer abstained. Once, in an LSV file, the statistics vote named `(Q-Qo)` (vote share 0.63).
- **Both named decoys came from the same statistics vote:** `<I>` → `(Q-Qo)` with `I` removed from the vocabulary, and `time` → `(Q-Qo)` with `time` removed.
- H_knn scores higher on main because one standard column set suits a nearest-neighbour vote, but it names every decoy. The set is now spent and may be used for development.

**Development round on hold-out 6 (after the run, so not blind).** Three of the four `<I>` misses came from the relation "follows current control". It was tested against `control_V`, a potential set point in CV and LSV, failed and ruled I out. The relation is now testable only when the control's unit allows a current. The same check applies in the other direction: Ewe is no longer tested against a current-only control (`control_I`). A second fix stops the statistics vote from naming `(Q-Qo)` where the known columns hold no current (I, dq or Pwe). EC-Lab computes `(Q-Qo)` from the current, and all 179 public files with `(Q-Qo)` also hold one of them. A rerun (`--label rules-dev`) then scores **19/20, 0 wrong, 1/16 decoys named**. The one abstention is the LSV `<I>`. The named decoy is `time` → `(Q-Qo)` with `time` removed from the vocabulary. Requiring a charge's steps to share the current's sign would catch it, but on that rule the decoy scores 0.75 against 0.88 for a real LSV file, too close to use. Golden changed only in these hold-out 6 answers.

**All six hold-outs, each arm run once (393 real columns, 111 scored decoys):**

| Arm | Correct | Wrong names on real columns | Decoys named |
|---|---|---|---|
| namer (versions 1→6, as frozen) | 366/393 (93.1 %) | **6** | **13/111** |
| Claude Opus 5.5 | 379/393 (96.4 %) | 10 | 19/111 |

## 2026-10-27: hold-out 7 (594 files from 222 records in about 30 repositories; frozen at `81d2d1b`), each arm run once

- **Files:** a broad search over Zenodo (including files inside archives), figshare, Dataverse, Mendeley Data, institutional and domain repositories, GitHub, DataCite, the literature and the web found 255 eligible openly licensed records. Up to three `.mpr` files were taken from each record in a fixed hash order. Every record, licence and file is listed in `holdout/holdout7_sources.json`, and the plan was fixed before any run in `holdout/holdout7_plan.md`. No file is byte-identical to any file used before. One record whose files copied hold-out 4 was dropped. The 594 files decoded with at least 10 rows cover 13 techniques, data module versions 2, 3 and 11, 114 column sets and 81 names.
- **Tasks:** 334, of which 193 main over 52 names (more than twice the names of any earlier hold-out), 108 settled by name elimination (all 108 right), 6 duplicate, 6 constant and 21 decoys, 19 of them scored. The namer was run once on all tasks. Opus was run once on the scored ones, 214 tasks for about $17.30 API-equivalent.

| Arm | Main top-1 macro (52 names) | micro (193) | Wrong names on real columns | Abstained | Decoys named (19) |
|---|---|---|---|---|---|
| namer (no LLM) | 0.750 | 0.808 (156) | 14 | 23 | **1** (out-of-vocabulary `<I>` → `control`, confidence 0.19) |
| **Claude Opus 5.5** | **0.912** | **0.912 (176)** | **13** | 4 | 5 (noise → `Analog IN 1` ×3 and `Ece`; out-of-vocabulary `(Q-Qo)` → `dQ`) |
| H_knn | 0.524 | 0.565 | – | – | 19 |
| H_prior | 0.486 | 0.523 | – | – | 19 |

- **Namer, wrong names:**
  - `Energy we` → `Energy we charge`/`discharge` ×3
  - `Rcmp` → `Ece` ×2
  - `Rdc` and `Rac` → `Re(Z)`
  - `|Ece|` → `-Im(Z)`
  - `-Im(Zce)` → `-Im(Z)`
  - `Ece` → `control`
  - `Energy ce charge` → `Energy we`
  - `Energy ce` → `dq`
  - `dQ` → `(Q-Qo)`
  - `|Energy|` → `Energy we`
- **Namer, abstentions:**
  - Temperature ×4
  - `Cp⁻²`, `Cs⁻²` and `Rwe` ×3 each
  - `Analog IN 1` ×2
  - `Energy ce` ×2
  - and single cases
- **Opus, wrong names:**
  - analog inputs → `Temperature` ×4
  - `Energy we` ↔ `|Energy|` ×3
  - `Ece` → `Ewe-Ece`
  - `Cp` → `Re(C)`
  - `-Im(Zce)` → `-Im(Z)`
  - `z cycle` → `counter inc.`
  - `half cycle` → `cycle number`
  - `Energy ce charge` → `Energy we-ce charge`
- **Secondary, namer only** (`python -m namer.evaluate --sweep holdout7`, run once): **6,467 hidden columns in 594 files over 79 names. 6,103 correct (94.4 %), 139 wrong (2.1 %), 225 abstained (3.5 %), macro 0.833.**
  - Top confusions: `I` → `Ece` ×31 (20 of them by the shape "a second smooth potential next to a known Ewe"), `|Z|` → `-Im(Z)` ×9, `|Ece|` → `-Im(Z)` ×7, `(Q-Qo)` → `dq` ×6, `Rcmp` → `Ece` ×6, `dQ` → `(Q-Qo)` ×6.
  - The wrong names come from 74 of the 222 records, not from a few.

**All seven hold-outs, each arm run once (586 real columns, 130 scored decoys):**

| Arm | Correct | Wrong names on real columns | Decoys named |
|---|---|---|---|
| namer (versions 1→7, as frozen) | 522/586 (89.1 %) | **20** | **14/130** |
| Claude Opus 5.5 | 555/586 (94.7 %) | 23 | 24/130 |

**Verdict:** hold-out 7 is the broadest set so far and the hardest for the namer. Many of its 52 names are rare: Rcmp, harmonics, the counter electrode's impedance, analog inputs. Opus names more columns. Wrong names are now close (20 against 23), and the namer is still fooled by far fewer decoys (14 against 24 of 130). The largest single weakness the sweep shows is a current named `Ece` by the potential shape when nothing in the file confirms a current. Hold-out 7 is now spent, so that is the next development target.
