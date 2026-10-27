# Hold-out 7: plan, fixed before any run (2026-10-27)

## Files
- **Search:** a broad search on 2026-10-26/27 for openly licensed, anonymously downloadable EC-Lab `.mpr` files that the project had never used. It covered Zenodo (including files inside archives), figshare, Dataverse installations, Mendeley Data, OSF, institutional and domain repositories, GitHub and other code hosts, DataCite and OpenAIRE, the literature and the general web. Nine searchers each took one family, a coverage review followed, and three follow-up searches filled its gaps. Searchers only listed files and never ran the namer.
- **Selection rule:** see `holdout7_sources.json`. A record counts if:
  - its licence permits reuse
  - its `.mpr` files are reachable directly or inside a zip that the server can read in parts
  - its files are 30 MB or less
  - its EC-Lab signature is not known to be wrong

  Up to three files are taken from each record, in a fixed hash order. Same-run `.mpt`/`.txt` exports come along. After download, two things were dropped: one record whose files were byte-identical copies of hold-out 4, and one byte-identical copy within the set.
- **The set:** 224 records, 615 `.mpr` files and 100 exports. The tasks use the 594 files that the reference decoder reads with at least 10 rows; the others are empty or unreadable. No file is byte-identical to any file used before.

## Tasks
`python -m holdout.build holdout7` uses the same design as hold-outs 1–6:
- one hidden real column per task
- at most 4 tasks per canonical name
- 6 duplicate and 6 constant columns
- 19 decoys: noise, random integers, out-of-vocabulary, constant and copy

The seed is 20261028. The answer key is yadg 7.0.1's column map. `label_check` in the manifest compares it with the vendor exports where they exist: the answer key's name was found in 43 of 62 export headers, and the rest are naming differences such as harmonics and THD.

The build has 334 tasks: 193 main tasks over 52 names, 108 tasks that name elimination settles (not scored), 6 duplicate, 6 constant and 21 decoys, of which 19 are scored. The `tasks.jsonl` sha256 is `1504d11ec0c043b923e314578095f5f500e39ae28b3533bdc5abcdd9c0cfb6c6`.

## Arms, each run once after the freeze commit
1. **namer** as frozen at the freeze commit. It behaves like main after `fc774a5`: the 0.1.5 release plus the `(Q-Qo)` rule, which has not been released.
2. **Claude Opus 5.5** (default effort) through `bench/claude_cli.py`, on the scored tasks only: `--buckets main,decoy`, 214 tasks.
3. The code-only baselines H_knn and H_prior.

## Measures
- **Primary**, for each arm on the task set:
  - main top-1, macro over names and micro
  - wrong names on real columns
  - abstentions
  - scored decoys named
- **Secondary**, namer only: `python -m namer.evaluate --sweep holdout7`. Every usable column of every usable file is hidden in turn, with the file's other columns known and the whole vocabulary allowed (constant and duplicated columns are skipped). It reports correct, wrong and abstained per name and in total. This measures the wrong-name rate on thousands of columns instead of a few hundred tasks.

Nothing in `namer/` changes between the freeze and these runs. Results go to `results/open_runs.md` whatever they show.
