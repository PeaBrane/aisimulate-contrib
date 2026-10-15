# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Collect exact-token closed-loop requests against an already-qualified endpoint.

This original client uses the public vLLM tokenization/completions protocol.
InferenceX's fixed-length client was inspected, but does not retain the exact
input/output token IDs required for this replay audit. No upstream code is copied.
"""

import argparse
import concurrent.futures
import hashlib
import json
import random
import statistics
import time
import urllib.request
from pathlib import Path


def fetch(url, payload=None, timeout=3600):
    data = None if payload is None else json.dumps(payload).encode()
    request = urllib.request.Request(url, data=data, headers={"Content-Type": "application/json"})
    return urllib.request.urlopen(request, timeout=timeout)


def metric_snapshot(urls):
    snapshots = {}
    for url in urls:
        with fetch(url, timeout=30) as response:
            snapshots[url] = response.read().decode()
    return snapshots


def exact_prompts(url, model, length, count, seed):
    text = " ".join(
        f"Experiment {i}: compare the measured time of the system and explain the result clearly." for i in range(128)
    )
    with fetch(url + "/tokenize", {"model": model, "prompt": text, "add_special_tokens": False}) as response:
        tokens = json.load(response)["tokens"]
    if not tokens:
        raise ValueError("tokenizer returned an empty token pool")
    tokens = tokens * ((length + len(tokens)) // len(tokens) + 1)
    rng = random.Random(seed)
    starts = [rng.randint(0, len(tokens) - length) for _ in range(count)]
    return [tokens[start : start + length] for start in starts]


def one_request(endpoint, model, prompt, output_length, request_id, epoch_ns, temperature):
    payload = {
        "model": model,
        "prompt": prompt,
        "max_tokens": output_length,
        "ignore_eos": True,
        "temperature": temperature,
        "seed": 42,
        "stream": True,
        "return_token_ids": True,
        "stream_options": {"include_usage": True},
    }
    start = time.perf_counter_ns()
    row = {
        "request_id": request_id,
        "arrival_ns": start - epoch_ns,
        "prompt_token_ids": prompt,
        "expected_output_length": output_length,
    }
    tokens, events = [], []
    usage = None
    try:
        with fetch(endpoint + "/v1/completions", payload) as response:
            for line in response:
                if not line.startswith(b"data:"):
                    continue
                content = line[5:].strip()
                if content == b"[DONE]":
                    break
                item = json.loads(content)
                if "error" in item:
                    raise ValueError(str(item["error"]))
                usage = item.get("usage") or usage
                for choice in item.get("choices", []):
                    ids = choice.get("token_ids") or []
                    if ids:
                        now = time.perf_counter_ns()
                        events.append({"elapsed_ns": now - start, "token_count": len(ids)})
                        tokens.extend(ids)
        if len(tokens) != output_length:
            raise ValueError(f"expected {output_length} explicit output IDs, received {len(tokens)}")
        if usage is None or usage.get("completion_tokens") != output_length:
            raise ValueError(f"server completion usage does not match output IDs: {usage}")
        if usage.get("prompt_tokens") != len(prompt):
            raise ValueError(f"server prompt usage does not match exact input length: {usage}")
        row.update(
            status="success",
            ttft_ms=events[0]["elapsed_ns"] / 1e6,
            tpot_ms=(events[-1]["elapsed_ns"] - events[0]["elapsed_ns"]) / 1e6 / (len(tokens) - 1)
            if len(tokens) > 1
            else None,
        )
    except Exception as error:
        row.update(status="failed", error=f"{type(error).__name__}: {error}")
    row.update(
        output_token_ids=tokens,
        stream_events=events,
        usage=usage,
        completion_ns=time.perf_counter_ns() - epoch_ns,
    )
    return row


def run_requests(endpoint, model, prompts, output_length, concurrency, temperature, prefix):
    epoch = time.perf_counter_ns()
    with concurrent.futures.ThreadPoolExecutor(max_workers=concurrency) as executor:
        futures = [
            executor.submit(
                one_request,
                endpoint,
                model,
                prompt,
                output_length,
                f"{prefix}-{index}",
                epoch,
                temperature,
            )
            for index, prompt in enumerate(prompts)
        ]
        rows = [future.result() for future in futures]
    return rows, (time.perf_counter_ns() - epoch) / 1e9


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--matrix", type=Path, required=True)
    parser.add_argument("--cell-id", required=True)
    parser.add_argument("--identity", type=Path, required=True)
    parser.add_argument("--endpoint", required=True)
    parser.add_argument(
        "--tokenizer-endpoint",
        help="Worker tokenization URL when the P/D proxy has no /tokenize route",
    )
    parser.add_argument("--metrics-url", action="append", required=True)
    parser.add_argument("--model", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--repetition", type=int, choices=[1, 2, 3], required=True)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--temperature", type=float, default=1.0)
    parser.add_argument("--smoke", action="store_true")
    args = parser.parse_args()
    cell = next(row for row in json.loads(args.matrix.read_text())["cells"] if row["cell_id"] == args.cell_id)
    identity = json.loads(args.identity.read_text())
    args.output.mkdir(parents=True, exist_ok=False)
    expected = 1 if args.smoke else cell["expected_requests"]
    warmup_count = 1 if args.smoke else cell["warmup_requests"]
    prompts = exact_prompts(
        args.tokenizer_endpoint or args.endpoint,
        args.model,
        cell["input_length"],
        expected + warmup_count,
        args.seed + args.repetition,
    )
    identity["client_source_sha256"] = hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
    identity["workload_generator"] = "tokenized-experiment-pool-v1"
    identity.update(
        cell=cell,
        seed=args.seed + args.repetition,
        temperature=args.temperature,
        workload_sha256=hashlib.sha256(json.dumps(prompts).encode()).hexdigest(),
    )
    identity_hash = hashlib.sha256(json.dumps(identity, sort_keys=True).encode()).hexdigest()
    (args.output / "identity.json").write_text(json.dumps(identity, indent=2) + "\n")
    with (args.output / "workload.jsonl").open("w") as out:
        for index, prompt in enumerate(prompts[warmup_count:]):
            out.write(
                json.dumps(
                    {
                        "request_id": f"measure-{index}",
                        "prompt_token_ids": prompt,
                        "output_length": cell["output_length"],
                    }
                )
                + "\n"
            )
    warmup, _ = run_requests(
        args.endpoint,
        args.model,
        prompts[:warmup_count],
        cell["output_length"],
        cell["concurrency"],
        args.temperature,
        "warmup",
    )
    (args.output / "warmup.jsonl").write_text("".join(json.dumps(row) + "\n" for row in warmup))
    if any(row["status"] != "success" for row in warmup):
        raise RuntimeError("warmup failed; preserved warmup records, profiling not started")
    before = metric_snapshot(args.metrics_url)
    rows, duration = run_requests(
        args.endpoint,
        args.model,
        prompts[warmup_count:],
        cell["output_length"],
        cell["concurrency"],
        args.temperature,
        "measure",
    )
    after = metric_snapshot(args.metrics_url)
    (args.output / "request_records.jsonl").write_text("".join(json.dumps(row) + "\n" for row in rows))
    (args.output / "metrics-before.json").write_text(json.dumps(before, indent=2))
    (args.output / "metrics-after.json").write_text(json.dumps(after, indent=2))
    successful = [row for row in rows if row["status"] == "success"]
    reasons = list(identity.get("qualification", {}).get("reasons", ["runtime qualification not supplied"]))
    if not identity.get("qualification", {}).get("eligible", False) and not reasons:
        reasons.append("runtime qualification is not eligible")
    if args.smoke:
        reasons.append("smoke request count; not a production measurement")
    if len(successful) != expected:
        reasons.append("some measured requests failed")
    summary = {
        "cell_id": args.cell_id,
        "repetition": args.repetition,
        "run_id": args.output.name,
        "identity_sha256": identity_hash,
        "metrics_contract": "dspark-e2e-client-v1",
        "total_gpus": cell["total_gpus"],
        "expected_requests": expected,
        "status": "success" if len(successful) == expected else "failed",
        "qualification": {"eligible": not reasons, "reasons": reasons},
        "request_records": "request_records.jsonl",
        "duration_s": duration,
        "acceptance_evidence": ["metrics-before.json", "metrics-after.json"],
        "metrics": {
            "mean_ttft_ms": statistics.mean(r["ttft_ms"] for r in successful) if successful else None,
            "mean_tpot_ms": statistics.mean(r["tpot_ms"] for r in successful) if successful else None,
            "output_throughput_tok_s": sum(len(r["output_token_ids"]) for r in successful) / duration,
            "completed_requests": len(successful),
            "failed_requests": expected - len(successful),
        },
    }
    (args.output / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary))
    if len(successful) != expected:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
