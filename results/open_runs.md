# Open project: runs on the locked v2 benchmark

> Commit ids before hold-out 6 refer to the private development history; from hold-out 6 on they are commits in this repository. Each hold-out was frozen before the namer version that was run on it, and each is reproducible: `python -m holdout.build <set>` rebuilds `data/<set>/tasks.jsonl`, whose SHA-256 is recorded in `holdout/manifest*.json`.

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

**Tried and rejected:** naming "the column that similar files always record" named up to 10/19 decoys. (The one exception since 0.1.6 is the measured current; see the development round on hold-out 7.)

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

**Development round on hold-out 6 (after the run, so not blind).** Three of the four `<I>` misses came from the relation "follows current control". It was tested against `control_V`, a potential set point in CV and LSV, failed and ruled I out. The relation is now testable only when the control's unit allows a current. The same check applies in the other direction: Ewe is no longer tested against a current-only control (`control_I`). A second fix stops the statistics vote from naming `(Q-Qo)` where the known columns hold no current (I, dq or Pwe). EC-Lab computes `(Q-Qo)` from the current, and all 192 distinct public files with `(Q-Qo)` at the time also held one of them (694 of 694 with hold-out 7). A rerun (`--label rules-dev`) then scores **19/20, 0 wrong, 1/16 decoys named**. The one abstention is the LSV `<I>`. The named decoy is `time` → `(Q-Qo)` with `time` removed from the vocabulary. Requiring a charge's steps to share the current's sign would catch it, but on that rule the decoy scores 0.75 against 0.88 for a real LSV file, too close to use. Golden changed only in these hold-out 6 answers.

**All six hold-outs, each arm run once (393 real columns, 111 scored decoys):**

| Arm | Correct | Wrong names on real columns | Decoys named |
|---|---|---|---|
| namer (versions 1→6, as frozen) | 366/393 (93.1 %) | **6** | **13/111** |
| Claude Opus 5.5 | 379/393 (96.4 %) | 10 | 19/111 |

## 2026-10-27: hold-out 7 (594 files from 221 records in 24 repositories; frozen at `81d2d1b`), each arm run once

- **Files:** a broad search over Zenodo (including files inside archives), figshare, Dataverse, Mendeley Data, institutional and domain repositories, GitHub, DataCite, the literature and the web found 255 openly licensed records that looked eligible. After merging four that were listed under two addresses, 225 had a `.mpr` file that could be read directly or from a zip read in parts, and up to three `.mpr` files were taken from each of them in a fixed hash order. Every chosen record, licence and file is listed in `holdout/holdout7_sources.json`, and the plan was fixed before any run in `holdout/holdout7_plan.md`. No file is byte-identical to any file used before. One record whose files copied hold-out 4 was dropped, and the files of two records could not be downloaded (an unsupported zip compression method, and a dataset page instead of a file), which leaves 222 entries. One dataset is among them twice, under its DOI and its EPrints address (Birmingham eData 1058), so the set has 221 records, and that one supplied 6 files instead of at most 3. The frozen set is kept as it is. The 594 files decoded with at least 10 rows cover 12 techniques (and 66 files whose technique is not read), data module versions 2, 3 and 11, 114 column sets and 81 names.
- **Tasks:** 334, of which 193 main over 52 names (almost twice the 27 of hold-out 1, the most of any earlier hold-out), 108 settled by name elimination (all 108 right), 6 duplicate, 6 constant and 21 decoys, 19 of them scored. The namer was run once on all tasks. Opus was run once on the main and decoy tasks, 214 of them (212 scored), for about $17.30 API-equivalent.

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
  - The wrong names come from 74 of the 221 records, not from a few.

**All seven hold-outs, each arm run once (586 real columns, 130 scored decoys):**

| Arm | Correct | Wrong names on real columns | Decoys named |
|---|---|---|---|
| namer (versions 1→7, as frozen) | 522/586 (89.1 %) | **20** | **14/130** |
| Claude Opus 5.5 | 555/586 (94.7 %) | 23 | 24/130 |

**Verdict:** hold-out 7 is the broadest set so far and the hardest for the namer. Many of its 52 names are rare: Rcmp, harmonics, the counter electrode's impedance, analog inputs. Opus names more columns. Wrong names are now close (20 against 23), and the namer is still fooled by far fewer decoys (14 against 24 of 130). The largest single weakness the sweep shows is a current named `Ece` by the potential shape when nothing in the file confirms a current. Hold-out 7 is now spent, so that is the next development target.

**Development round on hold-out 7 (after the run, so not blind).** In the sweep, the measured current was the weakest column. Of 414 hidden `<I>` columns, 31 were named `Ece` and 47 got ABSTAIN. Most were in impedance records, where DC relations cannot confirm a current, and in potential-controlled files with no charge column. Facts from the 880 distinct readable public files (hold-out 6's 57 files are also stored under `data/holdout3`, and hold-out 3 holds one file twice; each is counted once):
- Every impedance record holds `I` (202/202).
- Every record with a potential set point (`control` in V) holds `I` except one (319/320). The exception, galvani's `bio_logic4.mpr`, stores the current only as the per-point charge `dq`.
- Every impedance record that stores `Ece` also holds a counter-electrode impedance quantity (25/25).

The new rules apply only when the hidden column is the only one missing from the file. The benchmark treats every hidden column so (in 13 hold-out 7 files, 7 of them with a task, columns that yadg does not know are left out as well), and the tool does when a file has one unknown column. With several unknown columns, another of them may be the current.
- Where such a file names no current, a moving column is named `I`. Outside impedance records this needs a file without a charge or power column (`dq`, `(Q-Qo)`, `Pwe`); with one, I's own formulas decide. A column whose size follows the time step is not named `I`: it may be a per-point charge.
- `Ece` is ruled out there, and in impedance records without a counter-electrode quantity.
- If the column can be that missing current but `I` does not fit or is not an allowed name, the namer abstains instead of giving another name.

A review (three reviewers, with each finding checked by another agent) showed that the first version broke the tool's cases with more than one missing column. With the current and other columns unknown, or with `--self-check` on a file whose current is unknown, it named other columns `I`. That is why the rules now need the column to be the only one missing. In copies of public files with those column IDs removed from the reader's map, the tool now answers as it did before the change, or better. One known limit remains: in an old potentiostatic file that stores the current only as `dq`, a hidden `dq` whose size does not follow the time step is named `I`.

Hiding every column of the 880 distinct files in turn (9,040 columns that are not constant or copies):
- correct: 94.7 % → **96.2 %**
- wrong names: 151 → **108**
- abstained: 329 → 237

Every change is an `<I>` column now named correctly (599 `<I>` columns: 459 → 594 correct), and no answer got worse. One wrong name (`|Z|` → `Pwe` in one file) now comes with confidence 0.94 instead of 0.43, because `Ece` no longer competes. On hold-out 7 alone, 94.4 % → 95.7 % are correct and wrong names fall from 139 to 98.

On the hold-out tasks, a rerun (`--label rules-dev`, not blind) names more `<I>` columns than the code just before this fix and adds no wrong name:
- hold-out 4: 82/86 (was 78)
- hold-out 6: 20/20 (was 19)
- hold-out 7: the out-of-vocabulary `<I>` decoy is no longer named (0/19). The set's own `<I>` tasks were already right, so its main score stays 156/193 (macro 0.750).

Golden changed in 29 of 1527 answers:
- 8 changed answer, all to the right one: 7 `<I>` columns named, and the decoy now gets ABSTAIN.
- 18 keep their answer with higher confidence, mostly `|Ewe|` and `|I|` in impedance records, where `Ece` no longer competes.
- 3 out-of-vocabulary `<I>` decoys get ABSTAIN as before, now from the missing-current rule, with the confidence rounded.

**Review before release 0.1.6 (2026-10-28).** Five reviewers checked everything since 0.1.5: code, packaging, scripts and every number in the documents. Each medium or high finding was checked again by another agent that tried to refute it. Fixed before the release:
- **Name elimination with several unknown columns.** It is the oldest rule that infers a column from what the file lacks, and it ignored whether other columns were unknown too. In 12 hold-out 7 impedance and MB files, galvani knows `|Ece h2|` … `|Ece h6|` but not `|Ece h7|`, THD/NSD/NSR `Ece` or some energy columns, and every one of their unknown columns that is not constant or a copy (47 of 56) was named `|Ece h7|` at confidence 0.9. `--self-check` on those files answered `|Ece h7|` for 361 of 594 hidden columns. 0.1.5 does the same. Name elimination, the single-missing-sibling rule, the missing-current rules and the `(Q-Qo)` veto now apply only when one column is unknown. In `--self-check`, elimination still names the slot that hiding a column opens, but never a gap the whole file already has. On the 95 galvani-unknown columns of the 36 public files that have any: names that differ from the answer key 43 → 17 (5 more named columns are unknown to yadg too and keep their names), ABSTAIN 27 → 65, right 17 → 5. Of the 12 right `|Ece h7|` answers, 7 now get ABSTAIN and 5 are named `Pwe` by the statistics vote (confidence 0.30–0.42). `--self-check` on those files: 325 → 603 hidden columns named correctly, wrong names 502 → 90, and no column that was right is now wrong (4 now get ABSTAIN). Golden and the benchmark treat the hidden column as the only one withheld, so their answers do not change.
- **Counts.** The 938 readable files counted hold-out 6's 57 files twice, and hold-out 3 holds one file twice; the corpus figures above are now over the 880 distinct files. Hold-out 7 lists one dataset under two addresses, so it has 221 records, and its files come from 24 repositories, not about 30.
- **Smaller fixes.** A current of a few pA is no longer taken for whole numbers by the missing-current rule. `pytest tests` works in a fresh clone. The hold-out fetch script writes each file under a temporary name first, skips names that would leave the target folder and `.mpr` archive members over 30 MB, and checks that a server really answers range requests. The publish workflow checks that the release tag matches the version.
- Four tests were added: for the time-step exception, for elimination with several unknown columns, for the CLI passing on whether the column is the only unknown one, and for `--self-check` and the file's own gap.

**Several unknown columns: a missing harmonic (2026-10-28 and 29, after 0.1.6).** With several unknown columns, a missing harmonic now goes to the one unknown column that fits the gap's numbered siblings. Those are the known columns that differ from it only in a number, at least two of them with data (`|Ece h2|` … `|Ece h6|` for `|Ece h7|`). The column must have the storage type EC-Lab uses for the quantity, a median magnitude within a factor 10 of the siblings' range, and at most 1 percentage point more negative values than the sibling with the most. No column gets the name if any of these holds:
- a second unknown column fits, constant ones and copies included, or the only one that fits is constant or a copy;
- the siblings' medians lie more than a factor 100 apart. The members of one harmonic family agree within a factor 48.5 in the public files (median 1.9);
- the gap is not the file's only hole in a numbered family, so another unknown column could be one of its siblings. A family is three or more names that differ only in a number, which in the vocabulary means the harmonics. A missing member is a hole once any member is known;
- the reader is not sure where the columns end. That is the case when another column fits in a split of the record that is nearly as plausible, or when another split reads more of the known columns as their own type.

Siblings that differ in a word are other quantities and are not used. THD, NSD and NSR of one channel differ by a factor of 28 in the median channel (112 channels in 50 files), and by up to 94.
- **Reader.** The width search now tries every split of the unexplained bytes, not only the first 200, and scores each column's bytes once per offset. The public file with 8 unknown columns reads in a second or two in a fresh process instead of about 100 s.
  - The unknown columns' plausibility chooses the split, as before. Among the splits within 0.5 of the best, the one that leaves the most known columns readable as their own type wins: known columns can count against a split, never for it.
  - The other splits within 0.5 that leave as many known columns readable are kept as alternatives. The reader adds a note when another split, however implausible, reads more known columns as their own type.
  - It decodes all 932 distinct public `.mpr` files exactly as 0.1.6 does, with galvani and with yadg as the reference. 918 of them have data points; the other 14 have none.
  - **Public test files.** In the 24 test files with data module version 11, the recodings are all two-column ones plus the three-column ones that include an integer-valued f8: 6,260, which both readers can read. A recoding counts as split right when every column gets its own bytes. 829 that 0.1.6 split wrong are now split right, none the other way, 394 are split wrong by both, and 5,037 right by both. The recodings of the 25 files with versions 2 and 3 are left out: they used new IDs that yadg knows (990 to 992, which it reads modulo 256 as NSD Ece, NSR Ewe and NSR I), so no column became unknown. A reviewer's rerun of those files with IDs yadg does not know gave 390 fixed and none broken.
  - **Hold-out files.** 40,509 two-column recodings in the 580 usable hold-out files up to about 1 MB (at most 300 per file): 915 that 0.1.6 split wrong are now split right, none the other way, and 487 are split wrong by both. The scan also required each storage type to match. Counted that way, 847 are fixed, none broken and 1,446 wrong in both: in 959 of the 1,446 the present reader's split is right (0.1.6's too in 891), but an all-zero f4 or f8 column is read as an integer of the same width, which gives the same values.
  - **What got worse.** Trying every split also finds splits beyond the old cap that look more plausible than the right one. That needs six or more unknown columns, since only then are there more than 200 splits. Corrupt columns can also mislead the reader. Reviewers compared the two readers in 17 runs and 41 hard copies:
    - **The runs.** About 126,000 recodings summed over the runs, with 2 to 13 unknown columns where the runs log them, more than a third with five or fewer. The runs' own tallies sum to about 28,500 better, but their checks differ: 15,618 of those count a recoding as right only when each storage type matches too, 9,629 compare values, 3,296 widths and values, and 6 bytes. In one reviewer's search of 14,289 recodings with 6 to 8 unknown columns, 4,991 decoded better and 20 worse, comparing every column's values.
    - **Beyond the cap.** 45 distinct recodings in 35 files, each decoded right by 0.1.6, decoded worse this way: 9 with 6 unknown columns, 32 with 7 and 4 with 8. In each, the right split is among the first 200 and the chosen one beyond them; 4 of the 45 are marked doubtful. In those run through the command line, right names are lost, except in two one-row files whose columns are all constant, and the misread columns can get confident wrong names from other paths, for example `Phase(Z)` named `time` at 1.0.
    - **Corrupt columns,** all in recoded copies of the one file with corrupt harmonic columns. In 7 copies with 3 to 5 unknown columns, the reader takes a split 0.1 less plausible that moves corrupt columns onto readable bytes, because it then reads one more known column as its own type. It misreads 16 to 26 columns (2 of the hard cases below, 5 in a checker's MB_C03 runs). In 3 copies with 7 unknown columns, corrupt columns among the unknown ones lower the right split's score below that of a split the old search never tried (in one, 5.4 for split 129 of 336 against 6.1 for split 260). The reader marks these reads doubtful and misreads 33 to 50 columns. One is from a reviewer's search, 2 from a release checker's scan of 900 random recodings of the file. 0.1.6 split all 10 right but read one or two corrupt columns as integers.
    - **Counting.** Checks that compare each column's values or storage type count such cases as wrong in both; the hard-case check compares bytes, which is how it found 2 of the 10. Runs that used such checks may hide more of both kinds. No public file decodes differently. A first count, from three of the runs and the hard copies, gave 15 regressions beyond the cap, one of them counted twice, and 2 from corrupt columns.
  - **Not blind.** The reader was redesigned on these recodings.
- **Three reviews.** Two or three checkers looked at each version, and a separate agent tried to refute each medium or high finding.
  - **First version.** It had no hole or sibling check and used the old reader. It gave wrong names at confidence 0.9 in constructed cases, all with three or more harmonics unknown. Among the 1.6 million subsets of harmonic columns in the 39 harmonic files that leave a unique harmonic gap, there were 137 read with yadg and 2,191 with the storage types Rosetta's reader guesses. Its simulation had also bypassed the reader. Replayed through the reader, the same scenarios gave 11 wrong names. Nine of them came where `cycle number`, read as a zero u4 plus an f4, shifted the siblings. The hole condition blocks all 137 and all 2,191. A first form of it, which counted a hole only next to two known siblings, let 6 of the 2,191 through.
  - **Second version.** It added the known columns' plausibility to the split's score. A known column that is whole-numbered, mostly zero or corrupt scores less at its own offset than on a neighbour's bytes, so a split that shifted it could win. That made 1 of 53,529 hold-out pair recodings split worse than 0.1.6, and more in targeted searches. The rule also gave 21 wrong names in recoded copies of the one public file with corrupt harmonic columns (holdout7/zenodo_16140857 …MB_C03). The present reader, the sibling check and the doubt check came out of that review.
  - **Present version.** A third review ran 28,100 further trials, up to 12 unknown columns. The rule gave 23,014 names there and none wrong. Its runs flagged all 45 reader regressions beyond the cap, and it identified 2 of the corrupt-column ones among the hard copies. The release review and the check of its fixes sorted the flags and found the other 8 corrupt-column cases: 6 that the third review's runs had listed as wrong in both, and 2 in a new scan.
- **The 36 public files with galvani-unknown columns.** `|Ece h7|` is named right, at confidence 0.9, in all 12 files that hold it (6 PEIS, 1 GEIS and 5 MB files). That includes the 5 where the statistics vote said `Pwe`.
  - Names that differ from the answer key: 17 → 12. ABSTAIN: 65 → 58. Right: 5 → 17.
  - No other answer changes, and `--self-check` is unchanged (603 of 987 right, 90 wrong).
- **Replays through the reader.** Known columns of the public files get new IDs in a copy of the bytes, which `namer.mprfile` then reads.
  - **Simulation scenarios.** These are the scenarios with two or three hidden columns and a harmonic gap: 21,312 in 39 files. 245 were skipped because the file holds two columns under one name. Of the 21,067 replayed, the missing harmonic is named right in 19,923, and its name never goes to a wrong column.
  - **Fresh scenarios.** After the second version, a reviewer drew 61,649 further scenarios in the same files, with two to five hidden columns. None was used in development. 60,651 could be replayed (in 998, two columns share a name), and they give 51,011 right and none wrong.
  - **Whole command line.** On the same 21,067 scenarios, other paths still give 2,134 wrong names, 891 of them at confidence 0.9 or more.
    - 362 of them (98 at 0.9 or more) are on the missing harmonic's own column when the rule gives no name. In 0.1.6 that column got 9,175 (1,681).
    - On the other columns it is 1,772 (793), about as in 0.1.6 (1,812 and 784).
    - Most of the confident ones are a THD, NSD or NSR column called `Ece` by the shape rule for a second potential.
  - **Hard cases from the reviews.**
    - 41 recoded copies of six public files, read with yadg: 24 of MB_C03, 15 of the three operandoEIS files and 2 others, with 3 to 10 unknown columns. They give 76 right and 23 wrong names (0.1.6: 56 and 23); MB_C03 alone gives 28 and 10 (0.1.6: 17 and 9). None of the wrong names comes from the rule.
    - 314 recodings of three operandoEIS files with the galvani reference give no name from the rule.
    - The reviewers' 111 copies give 23 right names from the rule and no wrong one.
  - **Not blind.** The rule was designed on the 12 files above, and its failures were found and fixed on these replays and copies. The next hold-out is the blind test.
- **Regression.** Golden: 1527 answers, 0 differences. The every-column sweep is identical. Eight tests were added (30 pass), and each of 17 deliberate breaks of the rule or the reader fails one of them.

**Several unknown columns: THD, NSD and NSR are not `Ece` (2026-10-29, after 0.1.7).** With several unknown columns, smooth columns got wrong names from three shapes: `Ece` from "a second smooth potential next to a known Ewe" and `Ewe` from the missing-potential shape, often at confidence 0.9 or more, and `Phase(Z)` from the phase shape, below 0.9. Most were THD, NSD and NSR columns (in %), but harmonics, currents and impedance quantities were hit too. Three changes, each resting on facts from the 880 distinct public files:
- **`Ece` needs its channel.** All 25 public impedance records with `Ece` among the 880 distinct files hold all five counter-electrode impedance quantities `|Ece|`, `|Zce|`, `Re(Zce)`, `-Im(Zce)` and `Phase(Zce)`, and so does a 26th public file with 4 rows. One of the 177 records with a frequency column but no `Ece` holds them too. In all 26 of the 880 with `|Zce|`, it equals √(Re(Zce)² + Im(Zce)²) to 0.1 % in every row. `Ece` was already ruled out in an impedance record with none of the five known when the column was the only unknown one. Now that also holds with several unknown columns, unless the other unknown columns could hold the channel: at least five of them, three stored as f4 that fit that relation in 99 % of the rows where the two parts are not both zero. A record counts as an impedance record when its frequency, `|Z|`, `Re(Z)`, `-Im(Z)` or `Phase(Z)` is known: the frequency may be one of the unknown columns, and three public impedance records store none.
- **The phase shape gives way to the relation.** With `Re(Z)` and `-Im(Z)` known, atan2(Im, Re) decides `Phase(Z)`, so the phase shape, which THD, NSD and NSR (in %) also pass, no longer names anything. In the every-column sweep `Phase(Z)` is named by that relation in all 181 such cases.
- **Held between sweeps.** In impedance records with rows outside the sweep, EC-Lab keeps THD, NSD and NSR at their last value there. In the 36 public records with at least 10 such rows, all 81 THD, NSD and NSR columns leave at least 94.6 % of the steps there unchanged, `Ewe` at most 9.5 % (36 columns) and `Ece` at most 7.3 % (19). A column that leaves 90 % or more unchanged is no longer taken for a potential.
- **Replays.** Known columns of the public files get new IDs in a copy of the bytes, which the reader then reads, and the command line names the unknown columns. A recoding is left out when it also changes how another column is read (the IDs are stored next to each other, and yadg reads them one byte off): 282 of the 21,312 harmonic scenarios, 337 of 15,406 development and 345 of 15,556 held-back scenarios.
  - **Held-back scenarios.** 15,556 hidden sets of two to five columns in 563 files, drawn with their own seed before the review and not used for any decision: one or two THD, NSD or NSR columns with others, `Ece`, `Ewe` or `Phase(Z)` with others, and random sets. Most of its files are development files too. 14,888 compared, 51,443 hidden columns. Wrong names at confidence 0.9 or more: 1,464 → 469; at 0.6 to 0.9: 2,414 → 1,188. THD, NSD and NSR named wrong at 0.9 or more: 580 → 19, `Ece` named wrong at 0.9 or more: 1,109 → 113. Right: 30,056 → 30,180, and no answer that 0.1.7 got right became wrong or ABSTAIN. Of the confident wrong names, 896 now get ABSTAIN, 70 the right name and 39 a confidence below 0.6. 10 names that were already wrong, at lower confidence, now reach 0.9: a wrong rival was ruled out. 81 hidden sets occur twice, under two families, and 342 more hide the same columns as a harmonic scenario. Without them: wrong at 0.9 or more 1,440 → 469, THD, NSD and NSR 557 → 19, right 29,402 → 29,524.
  - **Development scenarios.** The same families with another seed, files under 20,000 rows: 14,831 compared. Wrong at 0.9 or more 1,449 → 461 (THD, NSD and NSR: 594 → 24), right 28,937 → 29,059, none lost.
  - **The 21,067 harmonic-gap scenarios of the previous section.** The other columns: 1,772 → 573 wrong names, 793 → 92 of them at 0.9 or more (THD, NSD and NSR: 944 → 77 and 646 → 15). The missing harmonic's own column when the rule gives no name: 362 → 195 (98 → 1). The rule's names do not change. 21 names that were already wrong now reach 0.9: in one file `|Z|` named `Pwe` by |Ewe·I| (0.43 before, because `Ece` competed). Leaving out the unfaithful recodings changes these counts by at most 10.
  - **`Ece` stored twice.** In 9 of the 25 impedance records with `Ece` it is stored under two IDs (`Ece` and `<Ece>`), so the replays above never hide it there. 161 scenarios hide both, using new IDs that neither parser knows, and 33 of them hide the whole counter-electrode channel too. Of the 66 `Ece` columns hidden in those 33, 64 are named `Ece`, as in 0.1.7. No right answer is lost, and THD, NSD and NSR named wrong at 0.9 or more go from 7 to 0.
- **The 36 public files with galvani-unknown columns.** No answer for their unknown columns changes. `--self-check`: one wrong name less (`<I>` was named `Ece` in an impedance file with two unrelated unknown columns), 603 of 987 right, 89 wrong.
- **Review** (workflow wf_288772d2-7c0: code, adversarial-data and mutation checkers; a separate agent tried to refute each medium or high finding; 12 agents). Every finding that went to a verifier was confirmed, five of them at a lower severity than reported.
  - The first version also named `Phase(Z)` from `|Z|` and one part, asin(Im/|Z|) or acos(Re/|Z|) without its sign, and let the phase shape give way whenever one part was known. Where the counter-electrode impedance is small, `Phase(Zwe-ce)` fits asin to r2 0.996, and the prior's tie-break then gave it the name `Phase(Z)`: a right name lost in one file. The free intercept also let an ohm column fit acos (r2 0.98). The relations were removed. On a verifier's 5,937 scenarios they had decided 3,866 answers, 3,860 of them right, so a version that demands an exact fit is worth trying.
  - The first gate only counted the other unknown columns. A parser that lacks THD, NSD and NSR leaves six or nine of them unknown at once, and that switched it off: the relation check replaced the count.
  - A last check (workflow wf_cb34e42f-840, a code and a claims checker, a refuter per medium finding) found that the relation check also took rows where all three columns are zero as fits: in one public file the harmonics are zero in 99 % of the rows, so they passed. Such rows no longer count, and the tolerance comes from the median instead of the largest value, which one misread huge value could inflate. It also found two rules that no test protected (the phase shape's "both known" and `|Z|` or `Phase(Z)` marking an impedance record), and that the check grew with the cube of the unknown columns; a first look at 256 rows now keeps one call at a tenth of a second or less at the size of the largest public impedance record (228,542 rows, up to 11 other unknown columns). A copy of one column next to a negligible one still passes, and then `Ece` stays possible, as in 0.1.7. The rerun of every affected scenario gave the same answers, and its claims checker found no wrong number.
  - Tests were missing for `--self-check` passing its unknown columns on, for `Re(Z)` and `-Im(Z)` marking an impedance record, and for the acos sign. The first two are added, the third went with the relations. Each of 17 deliberate breaks of the three rules now fails a test.
  - Ruling out a wrong candidate lifts the next wrong one (the 10 and 21 cases above). In each, 0.1.7 already gave that name, at lower confidence.
  - The replays skipped `Ece` stored twice and new IDs that yadg knows, and could rename an untouched column. Those are now covered or left out as described above.
- **Tried and not used.** Naming a missing potential only when it is the one unknown column that looks like one. On all 15,168 development scenarios run, with the first version and before unfaithful recodings were left out: for `Ewe` it lost 459 right names, and wrong names at 0.9 or more rose by 30; for `Ece` it lost 212 right names to remove 102 confident wrong `Ece` names, mostly in DC records. A spread test cannot separate `Ewe` from THD, NSD and NSR in pure sweeps either: 33 of the 131 potential-like `Ewe` columns in public impedance records have a coefficient of variation (standard deviation over the absolute mean) at least as large as the smallest among the 41 potential-like THD, NSD and NSR columns (0.15).
- **What remains.** Of the 19 THD, NSD and NSR columns named wrong at 0.9 or more on the held-back scenarios, 10 are called `Ece` with `Ece` itself unknown in a file whose counter-electrode channel is known, 7 are called `Ewe` with `Ewe` unknown too, and 2 fit a relation by chance. galvani and yadg both know `Ece`, `Ewe` and `Phase(Z)`, so these cases come from recodings.
- **Regression.** Golden: 1527 answers, 0 differences. The every-column sweep is identical in all 9,305 columns. 36 tests pass (6 new).
- **Not blind.** The rules were designed on the development and harmonic replays, and the review's own scenarios shaped the final gate. The held-back scenarios played no part in any decision, but they come from the same files. The next hold-out is the blind test.
