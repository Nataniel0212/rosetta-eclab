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

## Corrections added after the run (2026-10-28)
The plan above is kept as frozen. A review before release 0.1.6 found these inaccuracies in it:
- **The set:** the 224 records are entries in `holdout7_sources.json`. The files of two could not be downloaded (an unsupported zip compression method, and a dataset page instead of a file), and one dataset is listed twice, under its DOI and its EPrints address (Birmingham eData 1058), so the 615 files come from 221 records and that one supplied 6. Of the 626 listed `.mpr` files, 6 repeat a file name already listed in the same record and were skipped, and 5 belong to the two records that could not be downloaded. Of the 21 files left out of the 594, 20 have fewer than 10 rows and one cannot be decoded. The tasks are drawn from the 594 files, and 182 of them hold a task.
- **Decoys:** there are 21. The 19 scored ones are noise, random integers and out-of-vocabulary columns; the constant and the copy are not scored. Opus ran on the main and decoy tasks, 214 of them, of which 212 are scored.
- **label_check:** of the 19 tasks whose answer-key name is not in the export header, 14 are columns the export does not include (for example harmonics and THD), 4 were matched to another file's export, and one export header could not be read. They are not naming differences.

Later: the `(Q-Qo)` rule ships in 0.1.6, where it applies only when the column is the only unknown one, as every task treats it.
