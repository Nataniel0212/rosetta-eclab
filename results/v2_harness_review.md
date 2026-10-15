> Historical review of the spike-v2 harness, kept as written. Some files it mentions (`spike/…`) belong to the earlier spike-v1 code, which is not part of this repository (see `docs/history.md`).

# Spike v2 harness review (adversarial, offline, 2026-10-14)

No API calls were made. Every number below was reproduced offline from `data/v2/tasks.jsonl`, `runs2/baselines_test.jsonl` and the corpus.

## Defects, ranked

**H1 (high): code alone passes GO.** Sources: `tasks.py:213` (every other column is listed by name), `prompt.md:13` ("x is not a quantity already present") and `baselines.py:99,113` (leave-one-group-out).
- 19 of the 61 test/main names (31% of macro weight) occur in only one source group: harmonics, NSD/NSR/THD and Energy we. Both baselines score 0 on them by construction.
- A rule that uses only the vocabulary strings picks the candidate one token away from at least 2 listed known names. It gives a unique answer on 42/180 tasks, all correct.
- With H_knn as fallback it scores **macro 0.669**: above 0.60 and 23 points over 0.44, with no model.
- **Fix:** add `H_elim` and `H_combo` to `BASELINE_ARMS` (`score.py:29`). Alternatively, make the primary metric the 138 tasks (49 names) elimination cannot solve, where H_knn is 0.548, not 0.44.

**H2 (high): failed runs count as clean on decoys and as completed on main.** Sources: `score.py:43,97,101,137`.
- A failed row has no verdict, so `hallucinated()` is False. Completeness counts failed rows too.
- Reproduced with a synthetic U file: all 20 decoys failed and 10% of main failed → hallucination 0/20, complete=True, **GO**. One decoy answered out of 20 also passes.
- **Fix:** count only `status=="ok"`. Treat failed or missing decoys as hallucinations, or require 20/20. Apply the ≥95% rule to main and decoys separately.

**M1 (medium): about 56 test/main tasks (≈18-20 names, ≈30% of macro weight) cannot be identified from the data:**
- harmonic order: 36 tasks, 12 names
- NSD vs NSR: 12 tasks, 4 names
- THD in mb.issue_149, which has no |E|: 2 tasks. THD is DSL-testable in peis/geis (r²≈0.99).
- Analog IN 1 vs 2: 6 tasks, 2 names

All except one can only be solved by reading the siblings' names. They also look like noise decoys: 21 main tasks meet the same "no relation, |lag-1 autocorrelation| < 0.3" rule as 17/20 decoys. `prompt.md:8` says to ABSTAIN on noise, so the score measures how the model resolves that conflict. **Fix:** collapse these into family names, or report them outside the primary metric.

**M2 (medium): the lock check fails open.** Source: `run.py:50-52`.
- It ignores git's return code: a directory that is not a repo returns `(True,'ok')` (reproduced). The Windows "dubious ownership" error passes the same way.
- It never compares LOCK.json hashes with the files on disk, or `data/v2/tasks.jsonl` with the manifest.
- It leaves out `spike/spec.py` (`corpus.py:97`), `data/external/*.mpr`, the package versions and `runs2/baselines_test.jsonl` (the best baseline is recomputed at score time).
- **Fix:** check the return code, re-hash files and dossiers at startup, and commit the baseline answers to prereg2.

**M3 (medium): the decoys do not test the real risk.** Source: `tasks.py:47`.
- 6 of 20 decoys are free: the code already flags the 3 const decoys as constant and the 3 copies as identical. The real allowance is therefore 2 of 14.
- No decoy is a real quantity that is missing from the vocabulary, which is the product's actual failure mode.
- **Fix:** add out-of-vocabulary decoys (hide a real column and drop its name from that task's enum; ABSTAIN is correct).

**M4 (medium): the drift labels are wrong.** Source: `corpus.py:111-113`, which uses yadg's scrambled BE IDs.
- Under LE decoding, T143 `<Ece>` (471) and T145 `step time` (438) are both in galvani's map. Only 883/884 are new, and both are duplicates.
- So there are **zero genuinely new main columns**, and "names real drift" cannot be claimed.
- Despite `corpus.py:25`, D005 and D017 use a drift file (`tasks.py:199` does not filter).

**M5 (medium): thinking under json_schema is unverified.** If guided decoding starts at the first token, LT, U and LTA cannot reason. `api_check.py:22` uses only a 7-name toy schema. **Fix:** run api_check with `agent.schema()`, assert `reasoning_tokens > 0`, and pre-register a mode for each arm.

**M6 (medium): a constant target confirms every hypothesis.** `dsl.py:157` returns r²=1.0 for any expression (reproduced), which misleads LTA on const decoys. **Fix:** return r²=null and the message "x is constant".

**M7 (medium): the budget cap is soft.** The check before each call (`agent.py:184`) lets 8 workers overshoot. `agent.py:245` prices an unknown model at $0, so a renamed model ID never trips the cap. **Fix:** reject model IDs without a price, and reserve the worst-case cost under the lock.

**Low:**
- L1: `parse_json` (`agent.py:131`) fails on content that has only a closing `</think>`, or two JSON objects (reproduced). Split on the last `</think>` and use `raw_decode`.
- L2: the forced step (`agent.py:293`) still allows `"test"`. Send the answer-only schema.
- L3: the history drops `thought` (`agent.py:318`).
- L4: length cut-offs retry at the same 8192 tokens. Reruns retry failed tasks with no cap (`run.py:41`), which is a forking path; pre-register a cap.
- L5: the vocabulary name `"x"` (`vocab.py:34`) collides with the hidden-column alias.
- L6: the bootstrap treats names as independent (18 come from 3 files), and the margin has no paired test.
- L7: dev covers 13 frequent names. 16 of 18 dev files are also used in test.
- L8: the `TOKEN_FACTORY_BASE_URL` env override can route calls off Token Factory.

## Checked and fine

- **Answer key:** yadg's names match galvani's names for the LE IDs on every overlapping column, apart from synonyms and the unused ID 441. EIS physics gives r²=1.0 in all 9 files.
- **No answer is already a known name:** 0 tasks.
- **No leakage:** no IDs, positions or file names; the aliases are shuffled; no identical column is shared across groups; kNN never trains on the task's own group.
- **Vocabulary and schema:** the vocabulary has 114 names, and the enum matches what `validate_answer` accepts. dq/dQ is the only case collision and is handled.
- **DSL:** there is no exec path, no hang or memory blow-up, and NaN and inf are filtered.
- **Agent loop:** invalid replies stay out of the history, errors are plain language, the budget is shown, parse errors are capped, a 402 stops the run, every call is logged, there is a single writer, and reruns are resumable.
- **Cost:** about 1.6-1.9k system tokens plus a 2.7k dossier (max 4.7k) per call. On test that is about $0.1 for L0, $0.2 for LT, $1.6 for LTA and $2.8 for U ($6.2 at 8192 tokens per call), so `--max-usd 2.6` for U will probably stop early.

## Verdict

**Not ready to lock.** H1 lets a no-model baseline meet GO. H2 lets failed decoy runs pass. M2 means the lock does not lock. Fix those, decide M1 and M3, rerun the baselines and api_check, and then lock.

## Re-review after fixes

Reproduced offline, with no API calls. Running `spike2.score` rewrote `results/v2_score_test.json`.

### Remaining defects

**R1 (high): residual elimination.**
- The 12 NSD/NSR/THD tasks left in main (6 names, 12% of macro weight) are free for a model but score 0 for the baselines (single-group names). They tie in `elim.py` only because |C| and |Y| count as siblings of |Ewe| and |I|.
- If siblings must share a word, the rule is 54/54 correct across the corpus and 12/12 on main. With an H_knn fallback it scores **macro 0.670 against a bar of 0.698**, so GO needs only about 3 points of real skill.
- **Fix:** add that requirement to `one_token_apart` and rebuild. Main becomes 126 tasks over 43 names, H_knn scores 0.624 and the bar rises to 0.774.

**R2 (medium): oov holes are visible.** 9 of 10 test oov decoys can be flagged without looking at the data:
- 6 because a name used in ≥50% of other-group files with the same technique is missing from the vocabulary (I Range, time, |Z|, Ewe, control, Q charge or discharge);
- 3 because a sibling pair is broken in the vocabulary (Capacitance charge, Energy we charge, -Im(Zce)).

Main vocabularies never have a hole, so the cue has zero false positives. **Fix:** give every main task a matched sham hole, or exclude the oov names globally.

**R3 (medium): oov decoys get a free retry.** Outside strict json_schema (or if Token Factory ignores the enum), naming the excluded quantity is rejected as "not in the vocabulary" and the model is asked again. Reproduced: NAME `I Range`, then discarded, then ABSTAIN, which scores clean. **Fix:** keep an excluded-but-known name as the final NAME.

**R4 (medium-low): CRLF in generated prereg files.** `tasks_manifest.json`, `baselines_test.jsonl` and `LOCK.json` are written with CRLF, but git stores LF. In a scratch repo the working-copy hash was `df00…` and the blob/clone hash `bd52…`, while `git status` stayed clean. A clone or checkout makes the lock refuse, and auditors cannot recompute it. **Fix:** write with `newline="\n"`.

**Low:**
- `score._read` lets a later ok row override a failed one, so "failed is final" is not enforced when scoring.
- LOCK coverage is unchecked: an empty `files` map passes.
- Package versions are recorded but never checked.
- LTA's `.mpr` inputs are not verified against `data/external/manifest.json` (they match now).
- `api_check` skips the step schema and reads a missing `reasoning_tokens` as "thinking off".
- The prompt prose names |Z| and time even when the vocabulary omits them.
- Expressions near 1e200 silently return r²=0.

### Verified fixed

- **H2:** all decoys failed gives 24/24 hallucinations and no GO; 1 of 24 decoys answered gives no GO; 10% of main failed is incomplete; 3 of 24 decoys named fails.
- **Lock:** outside a repo it refuses (git exit 128). File hashes, `tasks.jsonl` and LOCK tracking are checked.
- **Reproducibility:** the rebuilt dossiers and recomputed baselines match the committed hashes and files.
- **Elimination on new main:** the original rule settles 0 tasks. Variants (≥1 sibling, dtype, prior tie-break) score 0.23–0.50, below H_knn's 0.548.
- **M4:** the drift columns are 880 (T049, main) and 883/884 (duplicates), and dev has no drift files.
- **M6, M7:** fixed; the host is asserted.
- **Lows:** `</think>` and two-object replies are handled, the forced step uses the answer-only schema, `thought` is kept, and the alias is `TARGET`.
- **Mocks:** the oracle and random mocks run end to end.

### Verdict

**Almost ready.** Fix R1–R3 (and R4 before the lock commit), rebuild the tasks and baselines, then lock.
