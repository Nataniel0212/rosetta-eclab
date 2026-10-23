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

*`drift.mpr` is `gcpl.mpr` from the yadg test suite, with one column ID changed to 950, an ID no parser knows (see `tests/test_cli.py`).*

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

The fix was released in datalab-org-galvani 0.6.0 ([galvani-mirror#11](https://github.com/datalab-org/galvani-mirror/issues/11)). Rosetta prints the note only when the installed galvani reads the low byte. IDs that are missing from the map altogether, such as 880 or 883 and 884, are what Rosetta names.

## How it works

1. **Read the file** (`namer/mprfile.py`). Columns the reference parser knows are typed from its column map. For unknown IDs the byte width is solved from the record size, and the storage type is chosen by plausibility.
2. **Code checks first.** Constant columns and exact copies of a known column are reported as such. Names that a family of siblings makes obvious are filled in, for example the missing harmonic `|Ewe h4|`.
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

The numbers come from five hold-out sets of public EC-Lab files from Zenodo, Battery Data Format and navani: 373 real columns and 95 decoys. None of these files were used during development. Each set was frozen before the namer version that ran on it, and each was run once (`results/open_runs.md`).

In every task one real column is hidden and must be named from the data. There are also decoys, where the right answer is not to name the column:
- noise and random integers
- *out-of-vocabulary* columns: real columns whose name the tool is not allowed to use, the way a brand-new quantity would look

| | Rosetta (no model) | Claude Opus 5.5 | best code-only baseline |
|---|---|---|---|
| Real columns named correctly | 350/373 (93.8 %) | 359/373 (96.2 %) | 77–100 % per hold-out¹ |
| Wrong names on real columns | **5** | 10 | – |
| Decoys named (should abstain) | **11/95** | 16/95 | 95/95 |
| Last hold-out (70 columns) | **70/70, 0 wrong, 1 decoy named** | 69/70, 1 wrong, 2 decoys named | 78.6 % |

¹ There are two code-only baselines. A technique prior names the column that is typical for this technique and missing here. A nearest-neighbour vote compares the column with labelled columns from other files. Both are strong when files have standard column sets, but they name every decoy and cannot name a genuinely new column.

The namer improved between hold-outs, so the total includes earlier versions. The history, including the language-model experiments that came first, is in [`docs/history.md`](https://github.com/Nataniel0212/rosetta-eclab/blob/main/docs/history.md).

### Limitations

- **Values EC-Lab computes after the fact** cannot be reconstructed reliably. For example, a capacitance is held per segment.
- **Noise-like DC currents** in impedance records, and **external analog inputs** with no sibling column, usually get ABSTAIN.
- **Two quantities with the same definition in a file** cannot be told apart from the data. In galvanostatic cycling, for example, `dQ` and `Q charge or discharge` reset at the same points. A prior over how common each name is for the technique decides between them, and the alternative is shown.
- **Supported formats:** EC-Lab data-module versions 2, 3, 10 and 11. Very old files (version 0) are not supported yet.

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
