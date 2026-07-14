#!/usr/bin/env python3
"""Organize the AstraQ-VL Hub repos and publish their exact model cards.

The script is dry-run by default. It is safe to run before or after the first
round of repository organization: every destination has one or more accepted
legacy source paths, and files already at their final paths are skipped.

Examples:
    python scripts/organize_hf_repos.py
    python scripts/organize_hf_repos.py --apply
    python scripts/organize_hf_repos.py --repo stage1 --apply
    python scripts/organize_hf_repos.py --apply --direct
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

from huggingface_hub import (
    CommitOperationAdd,
    CommitOperationCopy,
    CommitOperationDelete,
    HfApi,
)


ROOT = Path(__file__).resolve().parents[1]


@dataclass(frozen=True)
class RepoPlan:
    name: str
    repo_id: str
    model_card: Path
    # final path -> accepted old paths, newest first
    targets: dict[str, tuple[str, ...]]


STAGE1_TARGETS = {
    "checkpoints/standard/astraq-vl-stage1-ep1.zip": (
        "astraq-vl-stage1-ep1.zip",
    ),
    "checkpoints/standard/astraq-vl-stage1-ep2.zip": (
        "astraq-vl-stage1-ep2.zip",
    ),
    "checkpoints/standard/astraq-vl-stage1-ep3.zip": (
        "astraq-vl-stage1-ep3.zip",
    ),
    "checkpoints/legacy-no-heldout/astraq-vl-stage1-legacy-1epoch-no-heldout-all-checkpoints.zip": (
        "no-heldout-checkpoints/astraq-vl-stage1-legacy-1epoch-no-heldout-all-checkpoints.zip",
    ),
    "checkpoints/legacy-no-heldout/astraq-vl-stage1-legacy-1epoch-no-heldout-checkpoint-1300.zip": (
        "no-heldout-checkpoints/astraq-vl-stage1-legacy-1epoch-no-heldout-checkpoint-1300.zip",
    ),
    "evaluations/full-heldout/astraq-vl-stage1-full-heldout-eval-v1.zip": (
        "astraq-vl-stage1-full-heldout-eval-v1.zip",
    ),
    # The final directory name makes the limited Phase 0 scope visible in the
    # Hub file tree, even before a reader opens README.md.
    "evaluations/phase0-captions-only/phase0_stage1_ep1_results.zip": (
        "evaluations/phase0/phase0_stage1_ep1_results.zip",
        "phase0/phase0_stage1_ep1_results.zip",
    ),
    "evaluations/phase0-captions-only/phase0_stage1_ep2_results.zip": (
        "evaluations/phase0/phase0_stage1_ep2_results.zip",
        "phase0/phase0_stage1_ep2_results.zip",
    ),
    "evaluations/phase0-captions-only/phase0_stage1_results.zip": (
        "evaluations/phase0/phase0_stage1_results.zip",
        "phase0_stage1_results.zip",
    ),
    "metrics/stage1-training-curve/stage1_training_curve.csv": (
        "stage1_training_curve/stage1_training_curve.csv",
    ),
    "metrics/stage1-training-curve/stage1_training_curve.json": (
        "stage1_training_curve/stage1_training_curve.json",
    ),
    "metrics/stage1-training-curve/stage1_training_curve.png": (
        "stage1_training_curve/stage1_training_curve.png",
    ),
}


STAGE2_TARGETS = {
    "evaluations/full-heldout/astraq-vl-stage2-full-heldout-eval-v1.zip": (
        "astraq-vl-stage2-full-heldout-eval-v1.zip",
    ),
    "evaluations/phase0/phase0_stage2_results.zip": (
        "phase0_stage2_results.zip",
    ),
    "metrics/astraq-vl-stage2-metrics.zip": (
        "astraq-vl-stage2-metrics.zip",
    ),
    "metrics/eval-loss-curve/eval_loss_curve.png": (
        "eval_loss_curve.png",
    ),
    "metrics/eval-loss-curve/eval_loss_curve.zip": (
        "eval_loss_curve.zip",
    ),
}


PLANS = {
    "stage1": RepoPlan(
        name="stage1",
        repo_id="grKnight/astraq-vl-stage1",
        model_card=ROOT / "MODEL_CARD.md",
        targets=STAGE1_TARGETS,
    ),
    "stage2": RepoPlan(
        name="stage2",
        repo_id="grKnight/astraq-vl-stage2",
        model_card=ROOT / "MODEL_CARD_STAGE2.md",
        targets=STAGE2_TARGETS,
    ),
}


def repo_files(api: HfApi, repo_id: str) -> set[str]:
    return {
        item.path
        for item in api.list_repo_tree(repo_id, recursive=True)
        if item.__class__.__name__ == "RepoFile"
    }


def build_operations(
    plan: RepoPlan,
    files: set[str],
) -> tuple[list[object], list[str]]:
    operations: list[object] = []
    report: list[str] = []

    for destination, candidates in plan.targets.items():
        present_sources = [path for path in candidates if path in files]

        if destination in files:
            if present_sources:
                raise RuntimeError(
                    f"{plan.repo_id}: destination and legacy source both exist: "
                    f"{destination!r}, {present_sources!r}. Resolve this manually."
                )
            report.append(f"KEEP  {destination}")
            continue

        if len(present_sources) != 1:
            raise RuntimeError(
                f"{plan.repo_id}: expected exactly one source for {destination!r}; "
                f"found {present_sources!r} among {candidates!r}"
            )

        source = present_sources[0]
        operations.extend(
            [
                CommitOperationCopy(
                    src_path_in_repo=source,
                    path_in_repo=destination,
                ),
                CommitOperationDelete(path_in_repo=source),
            ]
        )
        report.append(f"MOVE  {source} -> {destination}")

    if not plan.model_card.is_file():
        raise FileNotFoundError(f"Model card not found: {plan.model_card}")

    operations.append(
        CommitOperationAdd(
            path_in_repo="README.md",
            path_or_fileobj=str(plan.model_card),
        )
    )
    report.append(f"WRITE README.md <- {plan.model_card.name}")
    return operations, report


def selected_plans(selection: str) -> Iterable[RepoPlan]:
    if selection == "both":
        return PLANS.values()
    return (PLANS[selection],)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--repo",
        choices=("stage1", "stage2", "both"),
        default="both",
        help="Repository to process (default: both).",
    )
    parser.add_argument(
        "--apply",
        action="store_true",
        help="Create commits. Without this flag, only print the plan.",
    )
    parser.add_argument(
        "--direct",
        action="store_true",
        help="Commit directly to main instead of opening pull requests.",
    )
    args = parser.parse_args()

    api = HfApi()

    for plan in selected_plans(args.repo):
        info = api.model_info(plan.repo_id)
        files = repo_files(api, plan.repo_id)
        operations, report = build_operations(plan, files)

        print(f"\n{plan.repo_id} @ {info.sha}")
        for line in report:
            print(f"  {line}")

        if not args.apply:
            continue

        result = api.create_commit(
            repo_id=plan.repo_id,
            repo_type="model",
            operations=operations,
            commit_message="Organize artifacts and refresh model card",
            commit_description=(
                "Group checkpoints, evaluations, and metrics; label caption-only "
                "Phase 0 artifacts explicitly; and update every model-card path."
            ),
            parent_commit=info.sha,
            create_pr=not args.direct,
        )
        print(f"  CREATED {result.pr_url or result.commit_url}")

    if not args.apply:
        print("\nDry run only. Re-run with --apply to create pull requests.")


if __name__ == "__main__":
    main()
