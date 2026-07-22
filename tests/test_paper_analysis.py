from __future__ import annotations

import copy
import tempfile
import unittest

from eval.paper.analysis import (
    _apply_holm_adjustment,
    _astro_hierarchy_intervals,
    _astro_paired_hierarchy_interval,
    _deepsdo_completion_diagnostics,
    analysis_run_fingerprint,
    paired_metric_intervals,
    score_caption_rows,
)
from eval.paper.protocol import PaperProtocol
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


class PaperAnalysisTests(unittest.TestCase):
    def test_holm_adjustment_is_monotone_within_metric(self) -> None:
        rows = [
            {"metric": "cider", "headline_comparison": True, "two_sided_p_value": 0.01},
            {"metric": "cider", "headline_comparison": True, "two_sided_p_value": 0.03},
            {"metric": "cider", "headline_comparison": True, "two_sided_p_value": 0.2},
        ]
        _apply_holm_adjustment(rows)
        self.assertEqual(
            [row["holm_adjusted_p_value"] for row in rows],
            [0.03, 0.06, 0.2],
        )

    def test_deepsdo_completion_diagnostics_report_lengths_and_caps(self) -> None:
        diagnostics = _deepsdo_completion_diagnostics(
            {
                "model": [
                    {
                        "response": "A complete caption.",
                        "reference": "A caption.",
                        "completion_token_count": 4,
                        "termination_reason": "eos",
                        "token_cap_hit": False,
                    },
                    {
                        "response": "unfinished",
                        "reference": "Another caption.",
                        "completion_token_count": 8,
                        "termination_reason": "max_new_tokens",
                        "token_cap_hit": True,
                    },
                ]
            },
            "original_512",
        )[0]
        self.assertEqual(diagnostics["token_cap_hits"], 1)
        self.assertEqual(diagnostics["completion_tokens_median"], 6)
        self.assertEqual(diagnostics["possible_unfinished_outputs"], 1)
    def test_later_astro_lock_does_not_rehash_internal_deep_analysis_run(self) -> None:
        protocol = PaperProtocol.load(ROOT / "configs" / "paper_eval_v2.yaml")
        changed = copy.deepcopy(protocol.data)
        changed["datasets"]["astrovlbench"]["locked_revision"] = "f" * 40
        modified = PaperProtocol(protocol.path, changed)
        record_hashes = {"internal": "1" * 64, "deepsdo": "2" * 64}
        with tempfile.TemporaryDirectory() as tmp:
            first = analysis_run_fingerprint(
                protocol,
                ("internal", "deepsdo"),
                record_hashes,
                tmp,
                repo_root=ROOT,
            )
            second = analysis_run_fingerprint(
                modified,
                ("internal", "deepsdo"),
                record_hashes,
                tmp,
                repo_root=ROOT,
            )
        self.assertEqual(first, second)

    def test_caption_scoring_keeps_invalid_row_in_denominator(self) -> None:
        protocol = PaperProtocol.load(ROOT / "configs" / "paper_eval_v2.yaml")
        rows = [
            {"id": "a", "status": "ok", "response": "bright sun", "reference": "bright sun"},
            {"id": "b", "status": "empty", "response": "", "reference": "solar flare"},
        ]
        scored, summary = score_caption_rows(rows, protocol, paper_mode=False, include_sbert=False)
        self.assertEqual(len(scored), 2)
        self.assertEqual(summary["n"], 2)
        self.assertEqual(summary["valid_response_rate"], 0.5)
        self.assertEqual(scored[1]["rouge_l"], 0.0)

    def test_pairing_rejects_different_record_ids(self) -> None:
        protocol = PaperProtocol.load(ROOT / "configs" / "paper_eval_v2.yaml")
        protocol.data["statistics"]["bootstrap_replicates"] = 10
        with self.assertRaisesRegex(Exception, "record IDs differ"):
            paired_metric_intervals(
                [{"id": "a", "image_id": "x", "token_f1": 1.0}],
                [{"id": "b", "image_id": "x", "token_f1": 0.0}],
                ["token_f1"], protocol, cluster_key="image_id",
                left_label="left", right_label="right", suite="internal", task="qa",
            )

    def test_astro_hierarchy_bootstrap_keeps_repeated_views_clustered(self) -> None:
        protocol = PaperProtocol.load(ROOT / "configs" / "paper_eval_v2.yaml")
        protocol.data["statistics"]["bootstrap_replicates"] = 20
        task_keys = (
            "task1", "task2.first", "task2.nvss", "task3", "task4",
            "task5.q1", "task5.q2", "task5.q3",
        )
        rows = []
        for index, task_key in enumerate(task_keys):
            source = "radio" if task_key.startswith("task2") else (
                "spectrum" if task_key.startswith("task5") else task_key
            )
            rows.append(
                {
                    "id": f"row-{index}",
                    "task_key": task_key,
                    "source_object_id": source,
                    "allowed_labels": ["A", "B"],
                    "reference_label": "A",
                    "prediction": "A",
                }
            )
        intervals = _astro_hierarchy_intervals(rows, protocol, "fixture")
        project = next(
            row for row in intervals if row["task_key"] == "hierarchy.project_summary"
        )
        self.assertEqual(project["estimate"], 0.5)
        paired = _astro_paired_hierarchy_interval(
            rows, rows, protocol, "left", "right"
        )
        self.assertEqual(paired["estimate"], 0.0)
        self.assertEqual(paired["lower"], 0.0)
        self.assertEqual(paired["upper"], 0.0)


if __name__ == "__main__":
    unittest.main()
