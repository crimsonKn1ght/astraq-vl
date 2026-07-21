# Invalid / superseded evaluation results

This note records evaluation artifacts that must **not** be used for reported
comparisons, and why. It exists so stale numbers are never silently carried into
tables or the manuscript.

## AstroLLaVA reference on the internal 2% held-out split — INVALID

**Artifact:** `astrollava-reference-full-heldout-eval-v1.zip`
(predictions + metrics for the `astrollava_reference` label on
`datasets/astrollava_llava/test.json`).

**Invalid for two independent reasons:**

1. **Faulty output-token slicing.** These predictions were generated before the
   decode fix (commit `b773be2`, "Fix AstroLLaVA reference decode to stop
   truncating generated answer prefix"). The `OfficialLLaVABackend` unconditionally
   stripped `input_ids.shape[1]` tokens from the output, which deletes the leading
   tokens of each answer and empties completions shorter than the prompt. The
   affected run has 31 unscored/empty rows (`n=3240` vs the split's `3271`), a
   direct symptom of the bug. Decoding is now handled by
   `decode_utils.strip_prompt_if_echoed`, covered by `tests/test_decode.py`.

2. **Training-data lineage overlap.** The AstroLLaVA reference model was trained on
   the same AstroLLaVA data lineage as this held-out split, so an in-domain
   comparison against it is not a clean held-out comparison. Per the review,
   AstroLLaVA is **omitted** from the internal in-domain comparison entirely and is
   evaluated **only** on the external benchmark (DeepSDO, then AstroVLBench).

**Action:** The internal 2% comparison reports **AstraQ-VL Stage 1 vs Stage 2
only**. A corrected AstroLLaVA run appears only in the external-benchmark results.

## Still valid

`astraq-vl-stage1-full-heldout-eval-v1.zip`, `astraq-vl-stage2-full-heldout-eval-v1.zip`,
and `qwen2_5-vl-7b-full-heldout-eval-v1.zip` use correct decode paths (marker-split
for AstraQ-VL, prompt-length slice for Qwen). They remain usable, pending the
invalid-response-accounting change (score empties as failures rather than dropping
them) that will re-run scoring for a consistent denominator across models.
