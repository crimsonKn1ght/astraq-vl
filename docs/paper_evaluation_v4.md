# Paper evaluation v4

Version 4 is the executable successor to the incomplete v3 DeepSDO protocol. It
does not train or tune any checkpoint. It writes only below
`eval_runs/paper_eval_v4`, `datasets/paper_eval_v4`, and
`checkpoints/paper_eval_v4`; v2 and v3 evidence remain unchanged.

## Reason for the revision

During the frozen v3 smoke run, Qwen3-VL-4B generated 512 completion tokens for
`deepsdo_test_0036` without emitting an assistant EOS token. The retained output
was coherent and non-repetitive but ended mid-sentence, so this was a genuine
ceiling hit rather than a decoding or EOS-detection error. Because v3 requires
zero token-cap hits, its `original_512` condition cannot proceed to definitive
full inference.

V4 changes only the ceiling for the unchanged original prompt:

| Condition | Prompt | Ceiling | Role |
|---|---|---:|---|
| `original_1024` | `Describe this solar image.` | 1024 | primary continuity |
| `concise_256` | `Describe this solar image in one concise sentence.` | 256 | prompt robustness |

Both conditions retain greedy decoding, one beam, no retrieval, no few-shot
examples, no quantization, and mandatory natural termination. A response that
reaches its ceiling remains a technical failure. Do not increase a ceiling
inside v4; create another protocol revision if a cap is reached.

## RunPod workflow

From a clean checkout of `evaluation/paper-eval-v4-suite`:

```bash
cd /workspace/astraq-vl
export HF_HUB_DISABLE_XET=1
```

Run each stage in order:

```bash
bash scripts/runpod/run_paper_eval.sh preflight \
  --protocol configs/paper_eval_v4.yaml \
  --suites deepsdo --models all --conditions all

bash scripts/runpod/run_paper_eval.sh prepare \
  --protocol configs/paper_eval_v4.yaml \
  --suites deepsdo --models all --conditions all

bash scripts/runpod/run_paper_eval.sh download \
  --protocol configs/paper_eval_v4.yaml \
  --suites deepsdo --models all --conditions all

bash scripts/runpod/run_paper_eval.sh smoke \
  --protocol configs/paper_eval_v4.yaml \
  --suites deepsdo --models all --conditions all --resume

bash scripts/runpod/run_paper_eval.sh run \
  --protocol configs/paper_eval_v4.yaml \
  --suites deepsdo --models all --conditions all --resume
```

Smoke uses a deterministic stratified cohort for every model and condition.
`token_cap` is accepted only as smoke plumbing evidence; it remains a full-run
failure. If any smoke row reports `token_cap`, stop and revise the protocol;
do not start the full run. Before analysis, every model/condition completion
report must contain 102 successful rows, no missing, failed, extra, or duplicate
rows, and no token-cap hits. The expected generation total is 1,020 rows.

## Analysis and blinded review

```bash
bash scripts/runpod/run_paper_eval.sh analyze \
  --protocol configs/paper_eval_v4.yaml \
  --suites deepsdo --models all --conditions all

bash scripts/runpod/run_paper_eval.sh audit-prepare \
  --protocol configs/paper_eval_v4.yaml --suites deepsdo
```

Two reviewers independently complete `reviewer_1.csv` and `reviewer_2.csv`.
Summarize the review:

```bash
bash scripts/runpod/run_paper_eval.sh audit-summarize \
  --protocol configs/paper_eval_v4.yaml --suites deepsdo
```

If the first pass creates `adjudication.csv`, complete it and rerun
`audit-summarize`. Packaging remains blocked until the factuality audit is
complete:

```bash
bash scripts/runpod/run_paper_eval.sh package \
  --protocol configs/paper_eval_v4.yaml \
  --suites deepsdo --models all --conditions all

sync
```

## Reporting rules

- Report the two prompt conditions separately.
- Label v3 as an incomplete protocol diagnostic, not a scored comparison.
- Preserve v2 as the preliminary shared-128-token result.
- Report CIDEr with significant digits and its one-reference/length caveat.
- Do not interpret reference overlap as factual correctness.
- Report all termination and length diagnostics and the blinded factuality audit.
