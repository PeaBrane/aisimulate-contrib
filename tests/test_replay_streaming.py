# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Public opt-in observation controls; no GPU, performance data, or network."""

from __future__ import annotations

import json

import pytest
import yaml

from aisimulate import main as cli
from aisimulate.cli_args import build_parser
from aisimulate.compiler import prediction_to_replay_spec
from aisimulate.config.cli import CorePredictionConfig
from aisimulate.predict import run_prediction
from aisimulate.runner import EngineReplayRunnerFactory, InvalidRunnerError
from aisimulate.sweeper.replay import ReplayOutputRequirements


def config(*, backend="vllm"):
    return {
        "engine": {
            "model": "example/model",
            "hardware": "h200_sxm",
            "backend": backend,
            "context_length": 4096,
            "workers": {
                "aggregated": {
                    "timing": {"type": "fixed", "prefill_ms": 1.0, "decode_ms": 1.0},
                    "kv_cache": {"capacity": {"type": "fixed", "blocks": 128}},
                }
            },
        }
    }


class Runtime:
    def __init__(self):
        self.payload = None

    def run_replay_json(self, payload):
        self.payload = json.loads(payload)
        return json.dumps({"duration_ms": 1.0, "output_throughput_tok_s": 1.0, "completed_requests": 1})


@pytest.mark.parametrize("invalid", [0, -1, True, 1.5, "20", float("nan")])
def test_output_interval_requires_positive_integer(invalid):
    with pytest.raises(ValueError, match="stream_interval must be a positive integer"):
        ReplayOutputRequirements(stream_interval=invalid)


@pytest.mark.parametrize("interval", [None, 1, 20])
def test_output_interval_reaches_json_wrapper_without_changing_engine_spec(interval):
    runtime = Runtime()
    spec = prediction_to_replay_spec(CorePredictionConfig.model_validate(config()))
    factory = EngineReplayRunnerFactory(runtime=runtime)
    baseline = factory.create(0)
    baseline.run(spec)
    original = runtime.payload
    baseline.close()
    runner = factory.create(1)
    runner.run(spec, output_requirements=ReplayOutputRequirements(stream_interval=interval))
    observed = runtime.payload
    runner.close()
    if interval is not None:
        assert observed.pop("stream_interval") == interval
    else:
        assert "stream_interval" not in observed
    assert observed == original


def test_non_vllm_runner_rejects_interval_before_calling_native_runtime():
    runtime = Runtime()
    spec = prediction_to_replay_spec(CorePredictionConfig.model_validate(config(backend="sglang")))
    runner = EngineReplayRunnerFactory(runtime=runtime).create(0)
    with pytest.raises(InvalidRunnerError, match="native vLLM text replay"):
        runner.run(spec, output_requirements=ReplayOutputRequirements(stream_interval=20))
    assert runtime.payload is None
    runner.close()


@pytest.mark.parametrize("stack,execution_mode", [("dynamo", "offline"), ("engine", "online")])
def test_public_prediction_rejects_unsupported_stream_observation_before_runner(stack, execution_mode):
    with pytest.raises(ValueError, match="offline engine prediction"):
        run_prediction(
            CorePredictionConfig.model_validate(config()),
            stack=stack,
            execution_mode=execution_mode,
            runner_factory=object(),
            output_requirements=ReplayOutputRequirements(stream_interval=20),
        )


@pytest.mark.parametrize("value", ["0", "-1", "1.5"])
def test_cli_rejects_invalid_interval(value):
    with pytest.raises(SystemExit):
        build_parser().parse_args(["predict", "-c", "unused.yaml", "--stream-interval", value])


def test_predict_stream_interval_reaches_runtime(tmp_path, monkeypatch, capsys):
    path = tmp_path / "prediction.yaml"
    path.write_text(yaml.safe_dump(config()))
    runtime = Runtime()
    monkeypatch.setattr(cli, "resolve_runner_factory", lambda _: EngineReplayRunnerFactory(runtime=runtime))
    assert (
        cli.main(
            [
                "predict",
                "-c",
                str(path),
                "--stream-interval",
                "20",
                "--output-dir",
                str(tmp_path / "out"),
                "--format",
                "json",
            ]
        )
        == 0
    )
    assert runtime.payload["stream_interval"] == 20
    assert json.loads(capsys.readouterr().out)["completed_requests"] == 1


def test_native_sdk_retains_stream_view_without_explicit_raw_report():
    from aisimulate import _runtime

    raw = config()
    raw["traffic"] = {
        "source": {"type": "synthetic", "input_tokens": 8, "output_tokens": 45},
        "load": {"type": "concurrency", "concurrency": 1},
        "stop": {"requests": 2},
    }
    spec = prediction_to_replay_spec(CorePredictionConfig.model_validate(raw))
    runner = EngineReplayRunnerFactory(runtime=_runtime, determinism="canonical_v1").create(0)
    try:
        baseline = runner.run(spec, output_requirements=ReplayOutputRequirements(include_raw_report=True))
        observed = runner.run(spec, output_requirements=ReplayOutputRequirements(stream_interval=20))
    finally:
        runner.close()
    original = dict(baseline.metadata["native_report"])
    actual = dict(observed.metadata["native_report"])
    stream = actual.pop("stream_observation")
    assert stream["completed_requests_with_content"] == 2
    assert stream["content_chunks"] == 8
    assert stream["request_mean_itl"]["mean_ms"] == pytest.approx(original["mean_tpot_ms"])
    for name in ("wall_time_ms", "processed_tokens_per_s", "processed_output_tokens_per_s"):
        original.pop(name)
        actual.pop(name)
    assert actual == original


@pytest.mark.parametrize("backend,extra", [("sglang", []), ("vllm", ["--online"]), ("vllm", ["--stack", "dynamo"])])
def test_unsupported_stream_observation_preserves_existing_output(tmp_path, monkeypatch, backend, extra):
    path = tmp_path / "prediction.yaml"
    path.write_text(yaml.safe_dump(config(backend=backend)))
    output = tmp_path / "output"
    output.mkdir()
    result = output / "prediction.json"
    result.write_text("preserve previous result")
    runtime = Runtime()
    monkeypatch.setattr(cli, "resolve_runner_factory", lambda _: EngineReplayRunnerFactory(runtime=runtime))
    with pytest.raises(SystemExit) as error:
        cli.main(
            [
                "predict",
                "-c",
                str(path),
                "--stream-interval",
                "20",
                "--output-dir",
                str(output),
                "--overwrite",
                *extra,
            ]
        )
    assert error.value.code == 2
    assert result.read_text() == "preserve previous result"
    assert runtime.payload is None
