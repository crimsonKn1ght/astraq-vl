# Paper evaluation v3

Version 3 is an immutable follow-up to the completed v2 evaluation. It never trains or tunes a checkpoint, and it writes only below `eval_runs/paper_eval_v3`, `datasets/paper_eval_v3`, and `checkpoints/paper_eval_v3`. Existing v2 evidence remains unchanged.

## Frozen DeepSDO design

DeepSDO has 102 official test records and five sequentially loaded models: AstraQ-VL Stage 1, AstraQ-VL Stage 2, AstroLLaVA, Qwen3-VL-4B-Instruct, and InternVL3.5-4B-Instruct. InternVL is pinned to commit `dbf19f04a6a6f2fb15821cc7ae738430a3580cf5` and uses its official 448-pixel dynamic tiling with at most 12 tiles.

The two conditions are independent generation contracts:

| Condition | Prompt | Cap | Role |
|---|---|---:|---|
| `original_512` | `Describe this solar image.` | 512 | primary continuity |
| `concise_256` | `Describe this solar image in one concise sentence.` | 256 | prompt robustness |

A row that terminates at `max_new_tokens` is a technical failure. Do not raise a cap inside the same protocol; create a new protocol revision and rerun the complete affected condition.

## RunPod requirements

Use persistent storage with one NVIDIA GPU reporting at least 22,000 MiB VRAM, compute capability 8.x or 9.x, 32 GiB RAM, and 80 GiB free disk. Models run sequentially and are not quantized. The wrapper creates seeded Python 3.11.15 environments, verifies every pinned package, installs Java for METEOR when possible, and records disk, quota, Git, package, GPU, and Hugging Face transfer state.

From a clean clone on persistent storage:

```bash
cd /workspace/astraq-vl

HF_HUB_DISABLE_XET=1 \
bash scripts/runpod/run_paper_eval.sh preflight \
  --protocol configs/paper_eval_v3.yaml \
  --suites deepsdo --models all --conditions all

HF_HUB_DISABLE_XET=1 \
bash scripts/runpod/run_paper_eval.sh prepare \
  --protocol configs/paper_eval_v3.yaml \
  --suites deepsdo --models all --conditions all

HF_HUB_DISABLE_XET=1 \
bash scripts/runpod/run_paper_eval.sh download \
  --protocol configs/paper_eval_v3.yaml \
  --suites deepsdo --models all --conditions all

HF_HUB_DISABLE_XET=1 \
bash scripts/runpod/run_paper_eval.sh smoke \
  --protocol configs/paper_eval_v3.yaml \
  --suites deepsdo --models all --conditions all --resume

HF_HUB_DISABLE_XET=1 \
bash scripts/runpod/run_paper_eval.sh run \
  --protocol configs/paper_eval_v3.yaml \
  --suites deepsdo --models all --conditions all --resume
```

The expected generation total is 1,020 rows: 102 records × 5 models × 2 conditions. Every model/condition completion report must show 102 successful rows, no missing/failed/extra rows, and no token-cap failures.

Smoke uses a deterministic stratified cohort for each model and condition. A resumed
smoke invocation reuses completed cohort members rather than advancing to different
benchmark rows. A `token_cap` satisfies only this smoke plumbing gate and remains a
technical failure in the complete evaluation.

## Analysis and blinded review

Run automatic scoring after generation:

```bash
bash scripts/runpod/run_paper_eval.sh analyze \
  --protocol configs/paper_eval_v3.yaml \
  --suites deepsdo --models all --conditions all
```

Prepare the frozen 30-image, 150-caption primary-condition audit:

```bash
bash scripts/runpod/run_paper_eval.sh audit-prepare \
  --protocol configs/paper_eval_v3.yaml --suites deepsdo
```

Two reviewers independently complete `reviewer_1.csv` and `reviewer_2.csv` using `CODEBOOK.md`. Then run:

```bash
bash scripts/runpod/run_paper_eval.sh audit-summarize \
  --protocol configs/paper_eval_v3.yaml --suites deepsdo
```

If reviewers disagree, the first summarize pass creates `adjudication.csv`. Complete it and rerun `audit-summarize`. Packaging is blocked until `factuality_audit/summary.json` reports `complete: true`.

Finally:

```bash
bash scripts/runpod/run_paper_eval.sh package \
  --protocol configs/paper_eval_v3.yaml \
  --suites deepsdo --models all --conditions all

sync
```

The package command verifies report/protocol hashes, condition-specific record and prediction fingerprints, factuality-audit completion, and private/public bundle checksums. Use `--resume` after an interrupted smoke or full generation run; never combine fingerprint directories manually.

## Reporting rules

- Report the two prompt conditions separately.
- CIDEr remains the continuity metric but is printed with significant digits; its near-floor magnitude and one-reference length sensitivity must be stated.
- METEOR, ROUGE-L, and BLEU are secondary.
- Predeclared paired comparisons use 10,000 bootstrap replicates and Holm-adjusted two-sided bootstrap p-values per metric and condition.
- Stage 1 versus Stage 2 is secondary on DeepSDO. Claims must describe observed dataset-specific competitiveness and cannot assume that AstraQ wins.
