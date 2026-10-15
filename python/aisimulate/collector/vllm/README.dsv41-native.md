# V4.1 native DSpark observation

This experimental collector runs real text requests through vLLM and observes
actual eager execution sites. It never constructs attention metadata, synthetic
KV state, or substitute kernels. The initial scope is B200 TP4/EP1, fixed K5,
adaptive verification disabled, 1024 input tokens, and 16 output tokens.

The source pin is [vLLM ac9126e58aa7bbab1856ba6593ba4d5003fea516](https://github.com/vllm-project/vllm/tree/ac9126e58aa7bbab1856ba6593ba4d5003fea516).
Its Apache-2.0 source is copyright contributors to the vLLM project. These
collector modules are original adapters using the pinned public worker-extension
and request APIs. `dsv41_native_sources.json` hashes the installed source files;
the worker rejects a different implementation before installing observation hooks.

## Run

Use the campaign's exact image and already available checkpoint. The caller must
provide a JSON array of 1024 real token IDs and a new shared output directory.
Run from `python/aisimulate` with this checkout on `PYTHONPATH`:

```bash
python -m collector.vllm.collect_dsv41_native \
  --model /models/DeepSeek-V4.1-Flash \
  --prompt-tokens /evidence/prompt-token-ids.json \
  --image-digest sha256:3d17635c5aa0340e2f2502c067ce4b587d8e7dc678195e90b71142490dc6d71c \
  --repetitions 5 \
  --output /evidence/v41-native-sites
```

The executable validates the checkpoint config hash, disables external Engram
offload and prefix caching, and warms two completed requests. It observes four
separate passes: layer, attention, mHC, and Engram. Each pass runs five fresh
requests by default (20 measured requests total). The synthetic acceptance
length is 3.51 committed tokens per round. Native query widths and absolute
positions are recorded from each invocation; expected values never replace them.
The output length allows a full K5 target verification round to occur. A runtime
that does not execute the required shape remains missing coverage.

These are forward-site measurements. The Python wrappers and extra observation
work change host execution, so request latency from this run must not become an
end-to-end validation point. CUDA events bracket the selected native call.
Positions are cloned after the ending event so reused native buffers cannot
rewrite the evidence. One site class per pass avoids timing nested observation
events as part of a parent module. Native internal streams and collectives stay
unchanged; inclusive scope and reduction ownership remain explicit.

## Source contracts

All paths below are relative to the immutable vLLM source pin:

| Contract | Source |
| --- | --- |
| Public worker RPC and extension mixin | `vllm/entrypoints/llm.py:572`, `vllm/v1/worker/worker_base.py:289` |
| Target/draft layer numbering and native attention call | `vllm/models/deepseek_v41/attention.py:287`, `:332`, `:659` |
| Native query boundaries and prefill/decode counts | `vllm/v1/attention/backends/mla/sparse_swa.py:182` |
| SWA metadata consumption by attention | `vllm/models/deepseek_v41/nvidia/flashmla.py:129` |
| Layer positions and shifted mHC ownership | `vllm/models/deepseek_v41/nvidia/model.py:452` |
| Fused prior post/current pre/RMSNorm and optional reduction | `vllm/models/deepseek_v41/nvidia/model.py:557`, `:599` |
| Eager mHC stream behavior | `vllm/models/deepseek_v41/nvidia/model.py:475` |
| Noncausal draft and native context insertion | `vllm/models/deepseek_v41/nvidia/dspark.py:1`, `:173` |

## Outputs and remaining gates

`receipt.json` retains launch identity, source hashes, request completions and
errors. `raw/rank-N.json` contains all observed calls, request boundaries,
positions, query widths, sequence counts, causal identity and native dispatch.
Each rank keeps failed or unsupported observations. No rank averages or fastest
rank selection occur in this stage.

The records are **not importable timing rows**. The current consumer's SGLang
component geometry does not represent vLLM's fused mHC/RMSNorm/communication
ownership. Parent/child inclusive costs must never be summed. The vLLM consumer
needs an explicit stage/site mapping and matching sequence/query/causal keys.
Final collapse and draft context preparation occur outside the observed layer
sites and remain a separate coverage requirement. No numerical accuracy or
complete performance-model coverage is claimed by a successful observation run.

The collector is experimental and is not added to default model population or
the op registry before the consumer and native smoke are qualified. Its explicit
entry point stages this bounded source audit; it does not emit perf Parquet or
modify existing measurements.
