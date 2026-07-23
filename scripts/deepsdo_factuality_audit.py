"""Prepare and summarize the frozen blinded DeepSDO factuality audit."""

from __future__ import annotations

import csv
import hashlib
import json
import random
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, Iterable, Mapping, Sequence

from eval.paper.artifacts import read_jsonl, sha256_file, write_json_atomic
from eval.paper.bootstrap import bootstrap_ci
from eval.paper.protocol import PaperProtocol


DIMENSIONS = (
    "factual_correctness",
    "unsupported_details",
    "relevant_coverage",
    "completeness_coherence",
)


def _records_path(data_root: Path, condition_id: str) -> Path:
    return data_root / "deepsdo" / "conditions" / condition_id / "records.jsonl"


def _stratified_sample(
    records: Sequence[Mapping[str, Any]], count: int, seed: int
) -> list[Dict[str, Any]]:
    groups: Dict[tuple[str, str, str], list[Dict[str, Any]]] = defaultdict(list)
    for record in records:
        key = (
            str(record.get("topic_stratum") or "unknown"),
            str(record.get("channel") or "unknown"),
            str(record.get("collapsed_modality") or "unknown"),
        )
        groups[key].append(dict(record))
    rng = random.Random(seed)
    for values in groups.values():
        rng.shuffle(values)
    selected: list[Dict[str, Any]] = []
    ordered_keys = sorted(groups)
    while len(selected) < count and ordered_keys:
        next_keys = []
        for key in ordered_keys:
            if groups[key] and len(selected) < count:
                selected.append(groups[key].pop())
            if groups[key]:
                next_keys.append(key)
        ordered_keys = next_keys
    if len(selected) != count:
        raise RuntimeError(f"Could select only {len(selected)} of {count} audit images")
    return selected


def _audit_id(seed: int, record_id: str, model_label: str) -> str:
    payload = f"{seed}:{record_id}:{model_label}".encode("utf-8")
    return hashlib.sha256(payload).hexdigest()[:20]


def _write_csv(path: Path, rows: Sequence[Mapping[str, Any]], fields: Sequence[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(fields), lineterminator="\n")
        writer.writeheader()
        writer.writerows([{field: row.get(field, "") for field in fields} for row in rows])


def _read_csv(path: Path) -> list[Dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as stream:
        return [dict(row) for row in csv.DictReader(stream)]


def prepare_audit(
    protocol: PaperProtocol,
    data_root: Path,
    output_root: Path,
    repo_root: Path,
) -> Path:
    config = protocol.data.get("factuality_audit") or {}
    condition_id = str(config["condition"])
    seed = int(config.get("seed", 42))
    sample_size = int(config.get("images", 30))
    records_path = _records_path(data_root, condition_id)
    if not records_path.is_file():
        raise SystemExit(f"Missing primary-condition records: {records_path}")
    records = read_jsonl(records_path)
    selected = _stratified_sample(records, sample_size, seed)
    records_hash = sha256_file(records_path)
    by_model: Dict[str, Dict[str, Mapping[str, Any]]] = {}
    prediction_hashes: Dict[str, str] = {}
    for model_label in protocol.selected_models("deepsdo"):
        model_root = protocol.model_output_dir(
            "deepsdo",
            model_label,
            records_hash,
            repo_root,
            output_root,
            condition_id,
        )
        predictions_path = model_root / "predictions.jsonl"
        if not predictions_path.is_file():
            raise SystemExit(f"Missing completed predictions for audit: {predictions_path}")
        predictions = read_jsonl(predictions_path)
        if len(predictions) != len(records):
            raise SystemExit(
                f"{model_label} has {len(predictions)} predictions; expected {len(records)}"
            )
        by_model[model_label] = {str(row["id"]): row for row in predictions}
        prediction_hashes[model_label] = sha256_file(predictions_path)

    audit_root = output_root / "factuality_audit"
    reviewer_paths = [audit_root / f"reviewer_{index}.csv" for index in (1, 2)]
    if any(path.is_file() for path in reviewer_paths):
        raise SystemExit(
            "Reviewer sheets already exist; refusing to overwrite possible completed work"
        )
    private_key: Dict[str, Any] = {}
    base_rows = []
    for record in selected:
        record_id = str(record["id"])
        for model_label in protocol.selected_models("deepsdo"):
            audit_id = _audit_id(seed, record_id, model_label)
            prediction = by_model[model_label][record_id]
            private_key[audit_id] = {
                "record_id": record_id,
                "model_label": model_label,
                "condition_id": condition_id,
            }
            base_rows.append(
                {
                    "audit_id": audit_id,
                    "image_id": record.get("image_id") or record_id,
                    "image_path": record.get("image_path"),
                    "reference": record.get("reference"),
                    "candidate": prediction.get("response"),
                    **{dimension: "" for dimension in DIMENSIONS},
                    "notes": "",
                }
            )
    fields = (
        "audit_id",
        "image_id",
        "image_path",
        "reference",
        "candidate",
        *DIMENSIONS,
        "notes",
    )
    for reviewer_index, path in enumerate(reviewer_paths, 1):
        rows = list(base_rows)
        random.Random(seed + reviewer_index).shuffle(rows)
        _write_csv(path, rows, fields)
    write_json_atomic(audit_root / "private_key.json", private_key)
    write_json_atomic(
        audit_root / "sample_manifest.json",
        {
            "protocol_sha256": protocol.fingerprint,
            "condition_id": condition_id,
            "seed": seed,
            "sample_images": sample_size,
            "candidate_rows": len(base_rows),
            "models": len(by_model),
            "records_file_sha256": records_hash,
            "prediction_file_sha256": prediction_hashes,
            "selected_records": [
                {
                    "id": row["id"],
                    "topic_stratum": row.get("topic_stratum"),
                    "channel": row.get("channel"),
                    "collapsed_modality": row.get("collapsed_modality"),
                }
                for row in selected
            ],
        },
    )
    (audit_root / "CODEBOOK.md").write_text(
        """# DeepSDO blinded factuality audit codebook

Review the image, official reference, and anonymous candidate independently. Do not attempt to infer the model.

- `factual_correctness`: 0 = major factual error, 1 = mixed/uncertain, 2 = factually supported.
- `unsupported_details`: 0 = no unsupported detail, 1 = minor unsupported detail, 2 = major hallucinated or unsupported claim.
- `relevant_coverage`: 0 = misses the main content, 1 = partial coverage, 2 = covers the main visible/reference content.
- `completeness_coherence`: 0 = incomplete/incoherent, 1 = understandable with a defect, 2 = complete and coherent.

Use only integer scores 0, 1, or 2. Add a short note for every 0 score or major unsupported-detail score of 2. Reviewers work independently. Differences are resolved only in the separate adjudication sheet produced by `audit-summarize`.
""",
        encoding="utf-8",
    )
    print(f"Prepared {len(base_rows)} blinded captions in {audit_root}")
    return audit_root


def _score(value: str, context: str) -> int:
    try:
        score = int(value)
    except ValueError as exc:
        raise SystemExit(f"Missing or invalid score for {context}") from exc
    if score not in {0, 1, 2}:
        raise SystemExit(f"Score must be 0, 1, or 2 for {context}")
    return score


def _weighted_kappa(left: Sequence[int], right: Sequence[int]) -> float:
    if len(left) != len(right) or not left:
        raise ValueError("Kappa requires equal non-empty rating vectors")
    observed = sum(abs(a - b) / 2 for a, b in zip(left, right)) / len(left)
    left_counts = [left.count(value) / len(left) for value in range(3)]
    right_counts = [right.count(value) / len(right) for value in range(3)]
    expected = sum(
        left_counts[a] * right_counts[b] * abs(a - b) / 2
        for a in range(3)
        for b in range(3)
    )
    return 1.0 if expected == 0 and observed == 0 else 1.0 - observed / expected if expected else 0.0


def summarize_audit(protocol: PaperProtocol, output_root: Path) -> Path:
    audit_root = output_root / "factuality_audit"
    sample_manifest = json.loads(
        (audit_root / "sample_manifest.json").read_text(encoding="utf-8")
    )
    if sample_manifest.get("protocol_sha256") != protocol.fingerprint:
        raise SystemExit("Audit sample manifest does not match the current protocol")
    reviewer_rows = [_read_csv(audit_root / f"reviewer_{index}.csv") for index in (1, 2)]
    reviewers = [
        {str(row["audit_id"]): row for row in rows} for rows in reviewer_rows
    ]
    if set(reviewers[0]) != set(reviewers[1]):
        raise SystemExit("Reviewer sheets do not contain identical audit IDs")
    private_key = json.loads((audit_root / "private_key.json").read_text(encoding="utf-8"))
    if set(private_key) != set(reviewers[0]):
        raise SystemExit("Private audit key does not match the reviewer sheet IDs")
    scored: Dict[str, Dict[str, int]] = {}
    disagreements = []
    agreement: Dict[str, float] = {}
    for dimension in DIMENSIONS:
        left = [_score(reviewers[0][audit_id][dimension], f"reviewer 1/{audit_id}/{dimension}") for audit_id in sorted(reviewers[0])]
        right = [_score(reviewers[1][audit_id][dimension], f"reviewer 2/{audit_id}/{dimension}") for audit_id in sorted(reviewers[0])]
        agreement[dimension] = _weighted_kappa(left, right)
    for audit_id in sorted(reviewers[0]):
        values = {}
        for dimension in DIMENSIONS:
            left = _score(reviewers[0][audit_id][dimension], f"reviewer 1/{audit_id}/{dimension}")
            right = _score(reviewers[1][audit_id][dimension], f"reviewer 2/{audit_id}/{dimension}")
            if left != right:
                disagreements.append(
                    {
                        "audit_id": audit_id,
                        "dimension": dimension,
                        "reviewer_1": left,
                        "reviewer_2": right,
                        "adjudicated_score": "",
                        "adjudication_notes": "",
                    }
                )
            else:
                values[dimension] = left
        scored[audit_id] = values
    adjudication_path = audit_root / "adjudication.csv"
    if disagreements and not adjudication_path.is_file():
        _write_csv(
            adjudication_path,
            disagreements,
            ("audit_id", "dimension", "reviewer_1", "reviewer_2", "adjudicated_score", "adjudication_notes"),
        )
        write_json_atomic(
            audit_root / "summary.json",
            {"complete": False, "reason": "adjudication_required", "disagreements": len(disagreements), "agreement": agreement},
        )
        print(f"Created {adjudication_path}; complete every adjudicated_score and rerun audit-summarize")
        return audit_root / "summary.json"
    if disagreements:
        adjudicated = {
            (row["audit_id"], row["dimension"]): _score(
                row["adjudicated_score"],
                f"adjudication/{row['audit_id']}/{row['dimension']}",
            )
            for row in _read_csv(adjudication_path)
        }
        expected = {(row["audit_id"], row["dimension"]) for row in disagreements}
        if set(adjudicated) != expected:
            raise SystemExit("Adjudication rows do not exactly match reviewer disagreements")
        for (audit_id, dimension), value in adjudicated.items():
            scored[audit_id][dimension] = value

    config = protocol.data.get("factuality_audit") or {}
    replicates = int(config.get("bootstrap_replicates", 10000))
    seed = int(config.get("seed", 42))
    model_rows = []
    for model_label in protocol.selected_models("deepsdo"):
        audit_ids = [audit_id for audit_id, key in private_key.items() if key["model_label"] == model_label]
        for dimension in DIMENSIONS:
            values = [scored[audit_id][dimension] for audit_id in audit_ids]
            interval = bootstrap_ci(values, n_resamples=replicates, seed=seed)
            model_rows.append(
                {"model": model_label, "dimension": dimension, **interval.as_dict()}
            )
    _write_csv(
        audit_root / "model_scores.csv",
        model_rows,
        tuple(model_rows[0]) if model_rows else (),
    )
    summary = {
        "complete": True,
        "protocol_sha256": protocol.fingerprint,
        "condition_id": config.get("condition"),
        "reviewers": 2,
        "candidate_rows": len(scored),
        "disagreements": len(disagreements),
        "agreement_weighted_kappa": agreement,
        "model_scores": model_rows,
    }
    write_json_atomic(audit_root / "summary.json", summary)
    lines = ["# DeepSDO blinded factuality audit", "", f"Reviewed captions: {len(scored)}", "", "## Agreement", ""]
    lines.extend(f"- {dimension}: weighted kappa {value:.4f}" for dimension, value in agreement.items())
    lines.extend(["", "The model mapping was concealed during independent review; disagreements were adjudicated before aggregation.", ""])
    (audit_root / "summary.md").write_text("\n".join(lines), encoding="utf-8")
    print(f"Completed factuality audit summary in {audit_root}")
    return audit_root / "summary.json"


__all__ = ["prepare_audit", "summarize_audit"]
