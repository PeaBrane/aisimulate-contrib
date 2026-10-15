# V4.1 native DSpark observation

This experimental collector runs real text requests through vLLM and observes
actual eager execution sites. It never constructs attention metadata, synthetic
KV state, or substitute kernels. The initial scope is B200 TP4/EP1, fixed K5,
adaptive verification disabled, 1024 input tokens, and 16 output tokens.

The base source pin is [vLLM ac9126e58aa7bbab1856ba6593ba4d5003fea516](https://github.com/vllm-project/vllm/tree/ac9126e58aa7bbab1856ba6593ba4d5003fea516).
Its Apache-2.0 source is copyright contributors to the vLLM project. These
collector modules are original adapters using the pinned public worker-extension
and request APIs. `dsv41_native_sources.json` hashes the installed source files;
the worker rejects a different implementation before installing observation hooks.
The image `sha256:3d17635c5aa0340e2f2502c067ce4b587d8e7dc678195e90b71142490dc6d71c`
contains a decoder replay graph patch beyond that base revision. The manifest
pins the installed model, replay layers, model state and replay graph manager.
The observer requires graph mode `NONE` and rejects an active replay graph;
only the audited eager branch retains the measured per-layer boundaries.
The upstream/installed hash comparison is retained in the campaign's
`v41-installed-source-audit.json` evidence. This is not graph-mode qualification.

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
offload and prefix caching, and warms two completed requests. It observes seven
separate passes: input preparation, layer, finalization, draft context preparation,
attention, mHC, and Engram. Each pass runs five fresh requests by default
(35 measured requests total). The synthetic acceptance
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

Paths below refer to the base source pin, except the four replay patch files
identified above, whose installed hashes are pinned to the image:

| Contract | Source |
| --- | --- |
| Public worker RPC and extension mixin | `vllm/entrypoints/llm.py:572`, `vllm/v1/worker/worker_base.py:289` |
| Target/draft layer numbering and native attention call | `vllm/models/deepseek_v41/attention.py:287`, `:332`, `:659` |
| Native query boundaries and prefill/decode counts | `vllm/v1/attention/backends/mla/sparse_swa.py:182` |
| SWA metadata consumption by attention | `vllm/models/deepseek_v41/nvidia/flashmla.py:129` |
| Layer positions and shifted mHC ownership | `vllm/models/deepseek_v41/nvidia/model.py:452` |
| Fused prior post/current pre/RMSNorm and optional reduction | `vllm/models/deepseek_v41/nvidia/model.py:557`, `:599` |
| Eager mHC stream behavior | `vllm/models/deepseek_v41/nvidia/model.py:475` |
| Bounded replay gather/scatter and restored full rows | `vllm/models/deepseek_v41/decoder_replay_layers.py:46` |
| Eager versus inner replay graph selection | Installed `vllm/models/deepseek_v41/nvidia/model_state.py:242`, `:479`; `nvidia/decoder_replay_cudagraph.py` |
| Root input and finalization boundaries | `vllm/models/deepseek_v41/nvidia/model.py:851`, `:1005`, `:1074` |
| Noncausal draft and native context insertion | `vllm/models/deepseek_v41/nvidia/dspark.py:1`, `:173` |

## Outputs and remaining gates

`receipt.json` retains launch identity, source hashes, request completions and
errors. `raw/rank-N.json` contains all observed calls, request boundaries,
positions, query widths, sequence counts, causal identity and native dispatch.
Each rank keeps failed or unsupported observations. No rank averages or fastest
rank selection occur in this stage.

The finalizer emits `dsv41_native_stage_perf.parquet` only when all four ranks
agree on each invocation's exact identity, all 40 target and three draft layers
are present, and both input/finalization boundaries plus draft context preparation
were observed. It takes the slowest rank per invocation, then the median of
repeated identical physical queries. Auxiliary attention/mHC/Engram observations
remain diagnostic; they are never summed with inclusive layer costs. Failed or
partial observations remain in the raw files and cannot publish a table.

The physical key includes native site, layer/owner, config identity, TP/EP,
decoder replay, observed 528-byte MXFP8 SWA layout, phase, sequence count,
query width, causality and absolute past-KV position. Rows preserve actual
attention dispatch, cache dtype/record width, fused-reduction flags, source pin,
image digest, eager policy, sample count and rank count. Configuration identity
is canonical sorted HF JSON excluding only SDK-inferred top-level `quant_algo`,
`kv_cache_quant_algo` and `quant_dynamic`; original quantization declarations and
all model geometry remain included. The consumer rejects incompatible effective
GEMM/MoE/FMHA/cache overrides.

Native layers own shifted previous-layer mHC finalization, current pre/RMSNorm,
attention, FFN and their fused reductions. Input preparation ends at entry to the
first layer; it replaces embedding costs and includes native Engram hash setup.
Finalization begins immediately after the last layer returns and ends when the
model returns. This includes the bounded replay scatter and target collapse/norm;
its request identity uses full model input rows, not the bounded late-layer rows.
Draft final norm runs in its logits method and remains a separate approximation.
Draft context preparation spans native auxiliary projection through KV insertion.
No observation changes serving dispatch or constructs attention metadata.

The initial consumer requires exact queries. It neither extrapolates nor borrows
SGLang local-compute rows. Acceptance-driven replay can request other absolute
KV positions; these remain coverage gaps. Mixed prefill/decode native batches
and heterogeneous prefill batches also require additional physical identities.
A standalone homogeneous prefill retains its whole native stage graph.
A follow-up grid may interpolate
KV only within observed dispatch/compression boundaries, keeping site, sequence
count, query width, causal mode, cache layout and runtime exact. No such wider
coverage is claimed by this initial collector. The steady-state draft cost uses
an explicit full-acceptance KV-position bound; startup uses the actual prompt+K
query extent. Acceptance sampling remains independent from that cost assumption.

The collector remains experimental and is not added to default model population
or the op registry before a successful native GPU smoke. Its table has a separate
consumer; it never changes existing measurements or backend labels. A successful
collection establishes only the recorded operator coverage, not serving accuracy.
