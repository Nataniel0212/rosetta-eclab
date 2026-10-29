# Rosetta

**Rosetta names the columns of BioLogic EC-Lab `.mpr` files that your parser does not know. It shows the evidence for each answer, and it says so when the data cannot support a name.**

When the instrument vendor adds a column ID, open-source parsers such as [galvani](https://github.com/echemdata/galvani) stop reading the file until someone works out what the new column is. Rosetta reads the file anyway. It works out the byte width and storage type of each unknown column from the record size, and then identifies the physical quantity from the data. It uses no language model and no network.

```
$ rosetta drift.mpr --self-check
drift.mpr
  technique GCPL, 12057 rows, 79 bytes per record
  1 column ID(s) unknown to galvani: 950; widths solved from the record size (79 bytes)

  column ID 950 (<f8): Q charge or discharge [C or mA·h]  (confidence 0.533)
    evidence: relation: I*dt summed within half cycle: r2 1.0000, scale 0.0002778
    also considered: dQ, step time

  self-check: hid each of 11 known columns in turn; named 11, 11 correctly, 0 wrong, 0 abstained
```

*`drift.mpr` is `gcpl.mpr` from the yadg test suite, with one column ID changed to 950, an ID galvani does not know (see `tests/test_cli.py`).*

## Install

Python 3.10 or later:

```sh
pip install rosetta-eclab
```

Rosetta does not install galvani. If galvani or datalab-org-galvani is installed, "unknown to galvani" means unknown to the galvani you actually have. Without either, Rosetta uses a frozen copy of galvani 0.5.0's column map and says so. The names Rosetta gives never depend on which galvani is installed.

To run the tests or reproduce the benchmark, clone the repository instead:

```sh
git clone https://github.com/Nataniel0212/rosetta-eclab.git
cd rosetta-eclab
pip install ".[test]"
```

## Use

```sh
rosetta FILE.mpr                     # name every column unknown to galvani (the default reference parser)
rosetta FILE.mpr --reference yadg    # ... unknown to yadg instead
rosetta FILE.mpr --self-check        # also hide each known column in turn and report how many come back right
rosetta FILE.mpr --json              # machine-readable output
```

Each unknown column gets one of three answers:
- **NAME:** an EC-Lab quantity, with its unit, a confidence and the evidence, for example the relation that holds and how well it fits.
- **DUPLICATE:** identical in every row to a known column.
- **ABSTAIN:** the data does not support a name. Rosetta prefers this to a guess.

`--self-check` shows how well Rosetta knows *your* kind of file. It hides each known column in turn and tries to name it.

## If galvani stops with "Column ID … is unknown"

In files from EC-Lab 11.50 and later, galvani 0.5.0 reads only the low byte of each column ID. IDs of 256 and above therefore turn into other IDs:

| Real ID | Column | galvani 0.5.0 sees |
|---|---|---|
| 471 | `<Ece>/V` | 215 |
| 438 | `step time/s` | 182 |
| 431–433 | `\|Zwe-ce\|`, `Re(Zwe-ce)`, `-Im(Zwe-ce)` | 175–177 |

The column is often one that galvani already knows. Rosetta reads the full IDs and says so:

```
$ rosetta peis.issue_225.with_ece.mpr
peis.issue_225.with_ece.mpr
  technique PEIS, 32 rows, 112 bytes per record
  galvani 0.5.0 reads only the low byte of each column ID in this file layout, so it misreads 471 as 215, 430 as 174, 431 as 175, 432 as 176, 433 as 177
  every column ID is in galvani's column map; nothing to name
```

The fix was released in datalab-org-galvani 0.6.0 ([galvani-mirror#11](https://github.com/datalab-org/galvani-mirror/issues/11)). Rosetta prints the note when the installed galvani reads the low byte, or when no galvani is installed (it then assumes galvani 0.5.0). IDs that are missing from the map altogether, such as 880 or 883 and 884, are what Rosetta names.

## How it works

1. **Read the file** (`namer/mprfile.py`). Columns the reference parser knows are typed from its column map. For unknown IDs the byte width is solved from the record size, and the storage type is chosen by plausibility. Every split of the record into widths is tried. Among the splits nearly as plausible as the best, the reader takes the one that leaves the most known columns readable as their own type, and it says so when another split would read more of them.
2. **Code checks first.** Constant columns and exact copies of a known column are reported as such. Names that a family of siblings makes obvious are filled in when the column is the only unknown one, for example the missing harmonic `|Ewe h4|`. With several unknown columns, a missing harmonic goes to the one column with the storage type EC-Lab uses for it and the sign and scale of its numbered siblings (`|Ewe h2|`, `|Ewe h3|`, …). It goes to none in these cases:
   - fewer than two of its numbered siblings have data;
   - more than one column fits, or the only one that fits is constant or a copy of a known column;
   - the siblings disagree on a scale;
   - another harmonic of a family the file holds is missing too;
   - the reader is not sure where the columns end.
3. **Physics library** (`namer/library.py`). Every candidate quantity is tested against the file's other columns:
   - **charges:** current × time, reset per half cycle, cycle or sequence
   - **energies and power:** E × I and its integrals
   - **impedance:** |Z| = √(Re² + Im²), the phase angle, Cs and Cp
   - **clocks and step times**, and **counters** stepping by one
   - **set points** that the measured potential or current holds to

   Each test reports r² and the fitted scale, which must match the expected unit conversion. A potential can therefore not pass as a current. Fits are robust to the single glitch rows that real records have at step transitions.
4. **Statistics fallback** (`namer/fallback.py`). A nearest-neighbour vote over column statistics from public files. It is used only for columns that are not noise, and it is vetoed when a defining relation of the guessed quantity fails in the file.
5. **Abstain** when nothing above supports a name.

## How accurate it is

The numbers come from seven hold-out sets of public EC-Lab files: 586 real columns and 130 decoys. The last one alone has 594 files from 221 openly licensed records in 24 repositories. The namer was never developed on any of these files before its run on them. For hold-out 6, the file reader was written against the files' byte layout. Each set was frozen before the namer version that ran on it, and each arm was run once (`results/open_runs.md`).

In every task one real column is hidden and must be named from the data. There are also decoys, where the right answer is not to name the column:
- noise and random integers
- *out-of-vocabulary* columns: real columns whose name the tool is not allowed to use, the way a brand-new quantity would look

| | Rosetta (no model) | Claude Opus 5.5 | best code-only baseline |
|---|---|---|---|
| Real columns named correctly | 522/586 (89.1 %) | 555/586 (94.7 %) | 56.5–100 % per hold-out¹ |
| Wrong names on real columns | **20** | 23 | – |
| Decoys named (should abstain) | **14/130** | 24/130 | 130/130 |
| Last hold-out, 7 (193 columns, 52 names) | 156/193, 14 wrong, **1 decoy named** | **176/193, 13 wrong**, 5 decoys named | 56.5 % |

¹ There are two code-only baselines. A technique prior names the column that is typical for this technique and missing here. A nearest-neighbour vote compares the column with labelled columns from other files. Both are strong when files have standard column sets, but they name every decoy and cannot name a genuinely new column.

**Every column of hold-out 7** (namer only, planned before the run): every usable column of all 594 files that is not constant or a copy of another column was hidden in turn. Of 6,467 columns, Rosetta named 94.4 % correctly, gave a wrong name to 2.1 % and abstained on 3.5 %. The most common mistake was the current named `Ece` when nothing in the file confirmed a current. A fix found afterwards names a missing current from what the file type records, when it is the only column missing. That is not a blind result: with it, 96.2 % of the 9,040 such columns in the 880 distinct public files are named correctly and 1.2 % wrongly, against 94.7 % and 1.7 % before. galvani knows the current in every public file, so outside `--self-check` the fix matters only if a format change hides it.

**Hold-out 6** is 57 older EC-Lab files (data module version 0). Rosetta named 16/20 and Opus 20/20. One fix found afterwards on that set is in 0.1.5, the other in 0.1.6. A rerun with them names 19/20, and 20/20 with the hold-out 7 fix, but that is not a blind result.

The namer improved between hold-outs, so the total includes earlier versions. The history, including the language-model experiments that came first, is in [`docs/history.md`](https://github.com/Nataniel0212/rosetta-eclab/blob/main/docs/history.md).

### Limitations

- **Values EC-Lab computes after the fact** cannot be reconstructed reliably. For example, a capacitance is held per segment.
- **External analog inputs** with no sibling column usually get ABSTAIN.
- **Several unknown columns in one file:** most rules that infer a column from what the file lacks are used only when one column is unknown. There are two exceptions. A missing harmonic such as `|Ece h7|` goes to the one unknown column whose storage type, sign and scale match `|Ece h2|` … `|Ece h6|`, under the conditions in step 2 above. And when `Ewe` is not known, any unknown column that looks like a potential can be named `Ewe`.
  - That names `|Ece h7|` right in all 12 public files that need it.
  - Replays through Rosetta's reader gave two to five known columns of the public files new IDs. In 81,718 such cases the rule named the missing harmonic right in 70,934 and never gave its name to a wrong column.
  - This is not a blind result. The rule was designed on those 12 files, and its failures were found and fixed on the same files and replays.
  - A missing current, or a gap like `THD Ece` whose siblings are other quantities, is left to each column's relations and the statistics vote. More of those get ABSTAIN. The vote can give a wrong name at low confidence, and a shape or a relation can give a confident one.
  - In an impedance file, `Ece` needs a known counter-electrode impedance quantity such as `|Zce|`: every public impedance file with `Ece` holds five of them. That now holds with several unknown columns too, unless the other unknown columns could hold those five: at least five, three of which fit |Zce| = √(Re(Zce)² + Im(Zce)²). In 0.1.7 a THD, NSD or NSR column (in %) next to a known `Ewe` was often named `Ece` with confidence 0.9 or more.
  - In 21,067 of the replayed cases, the other columns got 573 wrong names, 92 of them at confidence 0.9 or more (0.1.7: 1,772 and 793; most of the confident ones were a THD, NSD or NSR column called `Ece`). When the rule gave no name, the harmonic's own column got 195 wrong names (0.1.7: 362; 0.1.6: 9,175).
  - In a further 14,888 replayed cases with two to five hidden columns, held back while the change was made, wrong names at confidence 0.9 or more fell from 1,464 to 469, those on THD, NSD and NSR columns from 580 to 19. No answer that 0.1.7 got right was lost. The cases leave out recodings that also change another column's name. They come from the same files, so this is not a blind result either.
  - Some names that were already wrong now come with confidence 0.9 or more, because a wrong rival was ruled out (10 in the 14,888 cases). A THD, NSD or NSR column can still be named `Ewe` when `Ewe` is unknown too, or `Ece` when `Ece` itself is unknown in a file whose counter-electrode channel is known.
  - `--self-check` still fills the slot that hiding a known column opens, never a gap the file already has.
- **Many unknown columns:** with six or more unknown columns, a record can be split in hundreds of ways. The reader tries them all, which fixes far more decodes than it breaks, but a wrong split can still look more plausible than the right one.
  - In one reviewer's search of 14,289 recodings of public files with 6 to 8 unknown columns, 4,991 decoded better than in 0.1.6 and 20 worse, comparing every column's values. Across all their runs, reviewers found 45 distinct recodings that 0.1.6 decoded right and the reader now decodes wrong, all with 6 to 8 unknown columns.
  - Corrupt columns can mislead the reader too. In copies of the one public file with corrupt harmonic columns, reviewers found 10 recodings, with 3 to 7 unknown columns, that 0.1.6 split right and the reader now splits wrong.
  - The misread columns can get confident wrong names.
  - No public file decodes differently.
- **Two quantities with the same definition in a file** cannot be told apart from the data. In galvanostatic cycling, for example, `dQ` and `Q charge or discharge` reset at the same points. A prior over how common each name is for the technique decides between them, and the alternative is shown.
- **Supported formats:** EC-Lab data-module versions 0, 2, 3, 10 and 11. Version 0 (older EC-Lab) is read with galvani's offsets, because yadg does not read it. All 59 public version-0 files decode byte for byte like galvani 0.5.0, and the two with an EC-Lab text export match it. The file technique is not read from version-0 files, so the namer works without it.

## Reproduce the benchmark

```sh
python scripts/fetch_testdata.py                 # development corpus (yadg + galvani test files)
python scripts/fetch_holdout.py holdout5         # any hold-out; see DATA_SOURCES.md
python -m spike2.tasks                           # v2 benchmark tasks (hashes in prereg2/tasks_manifest.json)
python -m holdout.build holdout5                 # hold-out tasks (hash in holdout/manifest_holdout5.json)
python -m namer.evaluate --split holdout5        # run the namer
python -m spike2.baselines --split holdout5      # code-only baselines
python -m bench.claude_cli --split holdout5 --score rules   # score (Claude runs need the Claude Code CLI)
python -m namer.golden check                     # regression check: every answer must match tests/golden
pytest tests
```

## Data and licence

Rosetta is free software under the **GNU General Public License v3.0 or later** (see `LICENSE`). It uses [yadg](https://github.com/dgbowl/yadg) (GPL), and `spike2/galvani_0.5.0_colmap.json` is a copy of the column map of [galvani](https://github.com/echemdata/galvani) 0.5.0 (GPL-3.0-or-later). Measurement files are downloaded at run time and never committed. Sources, licences and attributions are listed in [`DATA_SOURCES.md`](https://github.com/Nataniel0212/rosetta-eclab/blob/main/DATA_SOURCES.md). Rosetta is not affiliated with BioLogic.
