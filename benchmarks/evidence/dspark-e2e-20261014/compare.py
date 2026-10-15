# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Score qualified fixed-length serving runs against native Engine Replay."""

import argparse
import csv
import hashlib
import json
import math
import statistics
from collections import Counter, defaultdict
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

CONTRACT = "dspark-e2e-client-v1"
METRICS = ("mean_ttft_ms", "mean_tpot_ms", "output_throughput_tok_s")


def positive(value):
    return type(value) in (int, float) and math.isfinite(value) and value > 0


def index_runs(records):
    indexed = {}
    for record in records:
        key = (record["cell_id"], record["repetition"])
        if key in indexed:
            raise ValueError(f"duplicate run key {key}; select the intended attempt explicitly")
        indexed[key] = record
    return indexed


def qualification_error(measured, predicted, expected_requests, total_gpus):
    if measured is None:
        return "missing_measurement"
    if measured.get("status") != "success":
        return "measurement_" + measured.get("status", "invalid")
    if measured.get("qualification", {}).get("eligible") is not True:
        return "measurement_unqualified"
    if predicted is None:
        return "missing_prediction"
    if predicted.get("status") != "success":
        return "prediction_" + predicted.get("status", "invalid")
    if predicted.get("qualification", {}).get("eligible") is not True:
        return "prediction_unqualified"
    if predicted.get("predictor") != "aisimulate.engine_replay":
        return "unsupported_predictor"
    identity = measured.get("identity_sha256", "")
    if len(identity) != 64 or any(c not in "0123456789abcdef" for c in identity):
        return "missing_identity"
    if identity != predicted.get("identity_sha256"):
        return "identity_mismatch"
    for run in (measured, predicted):
        if run.get("metrics_contract") != CONTRACT:
            return "metric_contract_mismatch"
        if run.get("expected_requests") != expected_requests or run.get("total_gpus") != total_gpus:
            return "matrix_mismatch"
        metrics = run.get("metrics", {})
        if metrics.get("completed_requests") != expected_requests or metrics.get("failed_requests") != 0:
            return "incomplete_requests"
    return None


def compare(cells, measurements, predictions):
    """A point is one configuration/repetition, never one streaming chunk."""
    measured_index, predicted_index = index_runs(measurements), index_runs(predictions)
    cell_ids = [cell["cell_id"] for cell in cells]
    if len(set(cell_ids)) != len(cell_ids):
        raise ValueError("duplicate cell_id in matrix")
    expected_keys = {
        (cell["cell_id"], repetition) for cell in cells for repetition in range(1, cell.get("repetitions", 3) + 1)
    }
    unexpected = (measured_index.keys() | predicted_index.keys()) - expected_keys
    if unexpected:
        raise ValueError(f"runs outside selected matrix: {sorted(unexpected)}")
    rows, errors, variability = [], defaultdict(list), defaultdict(list)
    counts = Counter()
    requests = Counter()
    scored_cells = set()
    for cell in cells:
        expected = cell["expected_requests"]
        gpus = cell["total_gpus"]
        if type(expected) is not int or expected <= 0 or type(gpus) is not int or gpus <= 0:
            raise ValueError("matrix needs positive expected_requests and total_gpus")
        for repetition in range(1, cell.get("repetitions", 3) + 1):
            key = (cell["cell_id"], repetition)
            measured, predicted = measured_index.get(key), predicted_index.get(key)
            reason = qualification_error(measured, predicted, expected, gpus)
            status = reason or "paired"
            counts[status] += 1
            requests["planned"] += expected
            for side, run in (("measured", measured), ("predicted", predicted)):
                if run:
                    for field in ("completed_requests", "failed_requests"):
                        value = run.get("metrics", {}).get(field)
                        if type(value) is int and value >= 0:
                            requests[f"{side}_{field}"] += value
            if reason is None:
                requests["paired"] += expected
                scored_cells.add(cell["cell_id"])
            for metric in METRICS:
                y = measured.get("metrics", {}).get(metric) if measured else None
                p = predicted.get("metrics", {}).get(metric) if predicted else None
                metric_status = reason or ("scored" if positive(y) and positive(p) else "invalid_metric")
                signed = 100 * (p - y) / y if metric_status == "scored" else None
                row = {
                    "cell_id": key[0],
                    "model": cell.get("model_key"),
                    "deployment": cell.get("deployment"),
                    "num_speculative_tokens": cell.get("num_speculative_tokens"),
                    "acceptance": cell.get("acceptance"),
                    "sampling_model": predicted.get("acceptance", {}).get("sampling_model") if predicted else None,
                    "repetition": repetition,
                    "measurement_run_id": measured.get("run_id") if measured else None,
                    "prediction_run_id": predicted.get("run_id") if predicted else None,
                    "metric": metric,
                    "status": metric_status,
                    "measured": y,
                    "predicted": p,
                    "ape_percent": abs(signed) if signed is not None else None,
                    "signed_error_percent": signed,
                    "total_gpus": gpus,
                    "measured_per_gpu": y / gpus if metric == METRICS[2] and positive(y) else None,
                    "predicted_per_gpu": p / gpus if metric == METRICS[2] and positive(p) else None,
                    "measurement_reasons": measured.get("qualification", {}).get("reasons", []) if measured else [],
                    "prediction_reasons": predicted.get("qualification", {}).get("reasons", []) if predicted else [],
                }
                rows.append(row)
                if signed is not None:
                    errors[metric].append(signed)
                    variability[(key[0], metric)].append((y, p))
    aggregates = {}
    for metric in METRICS:
        values = errors[metric]
        aggregates[metric] = {
            "scored_runs": len(values),
            "unscored_runs": len(expected_keys) - len(values),
            "mape_percent": statistics.mean(map(abs, values)) if values else None,
            "signed_bias_percent": statistics.mean(values) if values else None,
        }
    variations = []
    for (cell_id, metric), values in sorted(variability.items()):
        item = {"cell_id": cell_id, "metric": metric, "paired_repetitions": len(values)}
        for side, series in zip(("measured", "predicted"), zip(*values, strict=True), strict=True):
            item[side + "_mean"] = statistics.mean(series)
            item[side + "_sample_stddev"] = statistics.stdev(series) if len(series) > 1 else None
        variations.append(item)
    groups = defaultdict(list)
    dimensions = ("model", "deployment", "num_speculative_tokens", "acceptance", "sampling_model", "metric")
    for row in rows:
        groups[tuple(row[field] for field in dimensions)].append(row)
    stratified = []
    for key, points in groups.items():
        values = [point["signed_error_percent"] for point in points if point["status"] == "scored"]
        stratified.append(
            {
                **dict(zip(dimensions, key, strict=True)),
                "available_runs": len(points),
                "scored_runs": len(values),
                "mape_percent": statistics.mean(map(abs, values)) if values else None,
                "signed_bias_percent": statistics.mean(values) if values else None,
            }
        )
    report = {
        "metrics_contract": CONTRACT,
        "status": "measured, no threshold" if any(errors.values()) else "pending evidence",
        "configurations": {
            "planned": len(cells),
            "with_qualified_pairs": len(scored_cells),
        },
        "runs": {"planned": len(expected_keys), **counts},
        "requests": dict(requests),
        "sessions": None,
        "aggregates": aggregates,
        "stratified": stratified,
        "variability": variations,
        "aggregation": "Unweighted mean of per-run APE; not error of averages or request-weighted error.",
        "scope": "Fixed-length serving E2E only; no kernel, chunk-gap, FPM or host-overhead accuracy claim.",
    }
    return rows, report


def plot_pairs(rows, output):
    fig, axes = plt.subplots(1, len(METRICS), figsize=(12, 4))
    for axis, metric in zip(axes, METRICS, strict=True):
        pairs = [row for row in rows if row["metric"] == metric and row["status"] == "scored"]
        if pairs:
            observed = [row["measured"] for row in pairs]
            predicted = [row["predicted"] for row in pairs]
            axis.scatter(observed, predicted, alpha=0.7)
            bounds = (min(observed + predicted), max(observed + predicted))
            axis.plot(bounds, bounds, linestyle="--", color="gray")
        else:
            axis.text(0.5, 0.5, "No qualified pairs", ha="center", transform=axis.transAxes)
        axis.set(title=metric, xlabel="Measured", ylabel="Engine Replay")
    fig.tight_layout()
    fig.savefig(output, dpi=160)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--matrix", required=True, type=Path)
    parser.add_argument("--measurements", type=Path, nargs="*", default=[])
    parser.add_argument("--predictions", type=Path, nargs="*", default=[])
    parser.add_argument("--output-dir", required=True, type=Path)
    args = parser.parse_args()
    matrix = json.loads(args.matrix.read_text())
    rows, report = compare(
        matrix["cells"],
        [json.loads(path.read_text()) for path in args.measurements],
        [json.loads(path.read_text()) for path in args.predictions],
    )
    report["sources"] = [
        {"path": str(path), "sha256": hashlib.sha256(path.read_bytes()).hexdigest()}
        for path in [args.matrix, *args.measurements, *args.predictions]
    ]
    args.output_dir.mkdir(parents=True, exist_ok=True)
    (args.output_dir / "report.json").write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")
    if rows:
        with (args.output_dir / "paired.csv").open("w", newline="") as stream:
            writer = csv.DictWriter(stream, fieldnames=list(rows[0]), lineterminator="\n")
            writer.writeheader()
            writer.writerows(rows)
    plot_pairs(rows, args.output_dir / "measured-vs-replay.png")
    print(
        json.dumps(
            {key: report[key] for key in ("status", "configurations", "runs", "aggregates")},
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
