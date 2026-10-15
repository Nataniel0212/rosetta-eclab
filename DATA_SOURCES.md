# Data sources

No measurement data is committed to this repository. The scripts download public files at run time into `data/`, which git ignores:

```sh
python scripts/fetch_testdata.py          # development corpus (yadg + galvani test suites)
python scripts/fetch_holdout.py holdout   # hold-outs 1-5, one at a time
python scripts/fetch_holdout.py holdout2
python scripts/fetch_holdout.py holdout3
python scripts/fetch_holdout.py holdout4
python scripts/fetch_holdout.py holdout5
```

Each run writes a `manifest.json` with SHA-256 checksums next to the files.

## Development corpus

| Source | Licence | Pinned version |
|---|---|---|
| [yadg](https://github.com/dgbowl/yadg), `tests/test_x_eclab/` (EC-Lab `.mpr` files) | GPL-3.0 | commit `61d55fe1fcf6cdd165b09a17ca99e261e1e34f50` |
| [galvani](https://github.com/echemdata/galvani), `tests/testdata/` (Git LFS) | GPL-3.0-or-later | commit `d4a5f444b16eae57d2edfad6df774ead95c92fd6` |

## Hold-outs

These files were never used while developing the namer. Each hold-out was frozen (`holdout/manifest*.json`) before the namer version that was run on it.

| Set | Source | Licence | Authors (Zenodo) |
|---|---|---|---|
| holdout | [Zenodo 12205268](https://zenodo.org/records/12205268): *Supporting data for "How Reference Electrodes Improve Our Understanding of Degradation Processes in Half- and Full-Cell Potassium-Ion Battery Setups"* | CC BY 4.0 | Panasenko, Jeschull et al. (2024) |
| holdout | [Battery Data Format](https://github.com/battery-data-alliance/battery-data-format), `tests/data/mpr` | Apache-2.0 | Battery Data Alliance |
| holdout | [navani](https://github.com/be-smith/navani), `Example_data` | MIT | navani contributors |
| holdout2 | [Zenodo 19222974](https://zenodo.org/records/19222974): *Composition-Dependent Bipolar Membrane Water Electrolysis over Self-Supported Nickel Iron Phosphide Electrodes* | CC BY 4.0 | Guerrero Rodríguez, Kolenko et al. (2026) |
| holdout2 | [Zenodo 15747202](https://zenodo.org/records/15747202): *Peculiarities in Structural Behaviour of Graphite During Anionic Intercalation of PF6-, FSI- and ClO4- at Elevated Temperature* | CC BY 4.0 | Gorbunov, Mikhailova (2025) |
| holdout3 | [Zenodo 17244893](https://zenodo.org/records/17244893): *Electrochemical lithiation of the layered superionic conductors AgCrSe2 and CuCrSe2* (files ≤ 12 MB) | CC BY 4.0 | Moon (2025) |
| holdout3 | [Zenodo 3631173](https://zenodo.org/records/3631173) and [3631156](https://zenodo.org/records/3631156): PEM fuel-cell data. Downloaded, but its data-module version 0 cannot be labelled by the reference decoder, so it is not used | CC BY 4.0 | Pivac, Barbir et al. (2020) |
| holdout4 | [Zenodo 18925051](https://zenodo.org/records/18925051): *Cycling and EIS test data of commercial solid state batteries (TDK Ceracharge)* (files ≤ 16 MB) | CC BY 4.0 | Zhu, Schröder (2026) |
| holdout5 | [Zenodo 10889611](https://zenodo.org/records/10889611): *NiFe-NO3 layered double hydroxide as a novel anode for sodium ion batteries* | CC BY 4.0 | Fortunato (2024) |
| holdout5 | [Zenodo 14809707](https://zenodo.org/records/14809707): *LaPrNiO4+δ nano-columnar thin films as oxygen electrodes for reversible solid oxide cells* | CC BY 4.0 | Panisset, Kreka et al. (2025) |
| holdout5 | [Zenodo 15211416](https://zenodo.org/records/15211416): *Investigation of the Chemical Structure of Fluorinated Diazonium Salts on the Electrografting Behavior and Thin Film Properties* | CC BY 4.0 | Bazin (2025) |
| holdout5 | [Zenodo 15263065](https://zenodo.org/records/15263065): *Dane* | CC BY 4.0 | Bujewska (2025) |
| holdout5 | [Zenodo 15747326](https://zenodo.org/records/15747326): *Temperature Influence on the Staging Process in Carbon Materials during Anionic Intercalation* | CC BY 4.0 | Gorbunov, Mikhailova (2025) |
| holdout5 | [Zenodo 3676445](https://zenodo.org/records/3676445): *Germanium tin alloy nanowires as anode materials for high performance Li-ion batteries* | CC BY 4.0 | Doherty, McNulty et al. (2020) |

## Derived data in this repository

- **`namer/data/stats.json` and `namer/data/stats_features.npy`:** column names and scale-free statistics, computed from the development corpus. The statistics fallback uses them so the tool works without the corpus on disk. Like the corpus, they are GPL-3.0.
- **`tests/golden/answers.jsonl`:** the namer's answers on every benchmark task, used as a regression check. It contains no measurement values.
- **`prereg2/` and `holdout/`:** task manifests (file names, column positions, answer key) and SHA-256 hashes. They contain no measurement values.

## Definitions

Physical definitions of the EC-Lab quantities are paraphrased from the BioLogic EC-Lab Software User's Manual, v10.38 (2014), §2.11.2, §4.3.4.1 and §5.1.5, and from BioLogic Application Note 64 (THD, NSD, NSR). Where the manual was silent, they were checked against the data. This project is not affiliated with BioLogic.
