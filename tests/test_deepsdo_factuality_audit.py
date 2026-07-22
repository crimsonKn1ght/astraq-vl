from __future__ import annotations

import unittest

from scripts.deepsdo_factuality_audit import _stratified_sample, _weighted_kappa


class DeepSDOFactualityAuditTests(unittest.TestCase):
    def test_stratified_sample_is_deterministic_and_covers_groups(self) -> None:
        records = [
            {
                "id": f"row-{index}",
                "topic_stratum": f"topic-{index % 3}",
                "channel": f"channel-{index % 2}",
                "collapsed_modality": f"modality-{index % 2}",
            }
            for index in range(40)
        ]
        first = _stratified_sample(records, 30, 42)
        second = _stratified_sample(records, 30, 42)
        self.assertEqual([row["id"] for row in first], [row["id"] for row in second])
        self.assertEqual(len({row["id"] for row in first}), 30)
        self.assertEqual({row["topic_stratum"] for row in first}, {"topic-0", "topic-1", "topic-2"})

    def test_weighted_kappa_handles_agreement_and_disagreement(self) -> None:
        self.assertEqual(_weighted_kappa([0, 1, 2], [0, 1, 2]), 1.0)
        self.assertLess(_weighted_kappa([0, 1, 2], [2, 1, 0]), 0.0)


if __name__ == "__main__":
    unittest.main()
