# Spike v2 result (2026-10-15)

**Decision: NO-GO.** Pre-registered at commit `484d375` (`prereg2/LOCK.json`, combined sha256 `3a0f8888…`) before the first test-split call. Raw scores are in `results/v2_score_test.json`.

| Arm | Macro top-1 on test/main (126 tasks, 43 names) | 95 % CI | Decoys named (24 scored) | $ per task | Median latency |
|---|---|---|---|---|---|
| U: Nemotron 3 Ultra, thinking on, medium effort | **0.411** | 0.30–0.53 | 10 (42 %) | 0.0143 | 15.6 s |
| L0: Nemotron 3.5 Lightning, thinking off | 0.054 | 0.02–0.10 | 19 (79 %) | 0.0006 | – |
| H_knn: code, supervised kNN, leave-one-source-group-out | **0.628** | 0.52–0.74 | (always names) | 0 | – |
| H_prior: code, technique prior | 0.465 | 0.36–0.58 | (always names) | 0 | – |

- **Bar:** GO needed ≥ 0.68 (H_knn + 5 points) and ≤ 10 % decoys named. Ultra is **22 points below** the best code baseline (paired 95 % CI −36 to −7) and named 10 of 24 decoys: 6 of 10 out-of-vocabulary columns and 4 of 8 noise columns.
- **Where Ultra does better than code:** 8 of 43 names. These are mostly impedance quantities that a formula ties together (`|Z|`, `Re/−Im(Zce)`, `Re/−Im/|Zwe-ce|`), plus `step time` and `Energy we charge`.
- **Where it loses:** 23 names, mostly confusions within a family. Charge columns (`(Q-Qo)`, `dQ`, `dq`, `Q charge or discharge`, `Q charge`), counters (`Ns`, `half cycle`, `cycle number`) and control vs measured current.
- **The one genuinely new column** in the corpus (`Energy we`, ID 880) was named `Energy we discharge`, which is wrong.
- **Cost of the test:** U $2.18, L0 $0.13. Dev and api_check added about $0.47.

**Conclusion.** On this benchmark, Nemotron cannot reliably name instrument columns from data, abstain when no name fits, or beat a simple supervised classifier. Rosetta's TI claim does not hold, and the pre-registered rule stops the project. What remains useful:
- the harness and its lessons (name elimination, source-group leakage, sham vocabulary holes, the lock)
- the narrower observation that Ultra does well when a closed-form relation identifies the column
