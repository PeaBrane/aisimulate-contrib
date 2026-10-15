# Pro0813 DSpark attention and cost audit

Scope: source audit of vLLM `ac9126e58aa7bbab1856ba6593ba4d5003fea516`,
Pro0813 checkpoint `72e1d3230f6c080a530b0a1d46f8eb4602340597`, TP8/EP1,
fixed K6, adaptive verification disabled. No GPU jobs or latency fitting were
performed in this audit. The original `smoke-pro-eager/` prediction is retained;
its observed/predicted difference is not qualified accuracy evidence.

## Corrected source-proven errors

1. **One draft layer instead of three.** Pro0813 declares the legacy MTP field
   `num_nextn_predict_layers=1`, but native V4 DSpark reads `n_mtp_layers or 3`.
   The model had incorrectly fallen back through the legacy field. DSpark now
   builds three layers independently of legacy MTP. V4.1 retains its separate
   runtime fallback, which explicitly permits `num_nextn_predict_layers`.
   This correction also aligns cost/KV capacity with the already three-layer
   transfer geometry.
2. **Old context projection shape and missing copied weights.** Native context
   preparation uses one replicated `MergedColumnParallelLinear(7168, [512]*3)`.
   The old model charged separate per-layer `7168 -> (1536+512)` projections.
   It now charges one `7168 -> 1536` projection, excluding unused Q work.
   The duplicated context WKV allocation adds `7168*1536` FP8 weight bytes per
   rank to the existing draft inventory. This is separate from backbone WKV.
3. **Missing attention output reduction.** The existing vLLM module collector
   stubs TP all-reduce/all-gather while constructing local partition shapes.
   Pinned TP8/EP1 serving executes row-parallel `wo_b` output reduction. Neither
   the module lookup nor the graph previously supplied that cost. The DSpark
   graph now adds one existing `CustomAllReduce` per target layer (61) and draft
   layer (3), at the corresponding query-token count. This correction is limited
   to vLLM TP8/EP1. The generic collective table remains an approximation to the
   runtime's selected communication implementation.
4. **Proxy compressor weights and missing stack heads.** Draft SWA has no
   compressor, so its HCA timing proxy's compressor matrices and FP32 position
   parameters are excluded from resident draft weights. The stack's final
   FP32 mHC collapse and BF16 main/final RMSNorm weights are counted once.
   The existing replicated BF16 Markov matrices match the runtime. Optional
   confidence-head tensors and quantization metadata still require a checkpoint
   allocation receipt; this correction does not claim complete memory closure.

Source references:

- [V4 DSpark construction, WKV duplication and preparation](https://github.com/vllm-project/vllm/blob/ac9126e58aa7bbab1856ba6593ba4d5003fea516/vllm/models/deepseek_v4/nvidia/dspark.py), lines 73–147 and192–223.
- [Attention projections](https://github.com/vllm-project/vllm/blob/ac9126e58aa7bbab1856ba6593ba4d5003fea516/vllm/models/deepseek_v4/attention.py), lines 248–284.
- [Sequence-parallel gate and decoder ownership](https://github.com/vllm-project/vllm/blob/ac9126e58aa7bbab1856ba6593ba4d5003fea516/vllm/models/deepseek_v4/nvidia/model.py), lines 1180–1213.
- Local measurement boundary: `collector/vllm/collect_dsv4_attn.py`,
  `_tp_simulation` and `run_dsv4_module` (collectives replaced with identity;
  `use_cuda_graph=True`).

## Remaining contracts and coverage gaps

The pinned target has 30 CSA layers (compression 4) and 31 HCA layers
(compression 128). Its three draft layers are pure SWA (native ratio 1, checkpoint
ratio 0), with no compressor or indexer. TP8 has 16 local query heads; the selected
FlashMLA decode implementation pads them to 64. Startup logs register
`DEEPSEEK_SPARSE_SWA/FLASHMLA_SPARSE_DSV4`, block 256, and
`FLASHINFER_TRTLLM_MXFP4_MXFP8` MoE. These logs identify configured backends, not a
per-invocation kernel trace.

The checked B200 vLLM 0.24 generation tables contain 1544 rows each for CSA/HCA;
every row has `isl=1`. They record local heads, TP, batch and total length, but
have no independent request-count/query-width/causal axes. The consumer also
collapses TP identity. Native verification actually classifies K+1 queries as
decode, flattens each query into a sparse attention row, and shares scheduler
metadata by layer type. Thus flattened token-basis lookup is a useful
approximation, not proof of exact seven-query serving coverage.

Draft attention still substitutes HCA with capped KV `128+K`. Actual noncausal
SWA uses padded index width `ceil((128+K)/64)*64`, which is 192 at K6, and has no
compressed branch. HCA latency remains a proxy error; its nonexistent compressor
weights have been removed from the draft inventory. This audit did not
manufacture a SWA row or claim exact memory closure.

Additional missing or approximate work includes auxiliary hidden-state
reconstruction/mean at the three target taps, hyper-connection head collapse,
final RMSNorm, logits collection, Markov embedding/logit additions/sampling,
and serving host/launch overhead. Existing mHC measurements already include two
sites per layer; multiplying those costs by two again would be wrong. Their
standalone pre/post decomposition differs from the pinned fused post/pre/norm
boundaries. None of these costs was inferred from the smoke latency gap.

The eager smoke cannot validate graph-basis operation timing: existing attention
collection uses CUDA graphs, whereas this serving run disables them. Preserve
eager/graph identity separately and collect graph serving results before judging
the numerical model. Kernel-launch and scheduler host time require separate
measurement, not an operation multiplier fitted to E2E latency.

## Minimal missing measurement contract

Collect actual pinned native forwards with checkpoint weights, TP8/EP1, C1 and
C4, FP8 KV and the runtime's resolved projection/MoE policies:

| Site | Query width | Causality | Required ownership |
|---|---:|---|---|
| Target CSA/HCA prefill | 1024 | causal | Complete attention module, including output reduction, or explicitly local compute plus separately observed reduction |
| Target CSA/HCA verify | 7 | causal | Real per-query sparse indices; shared planner ownership recorded once per layer type |
| Draft SWA proposal | 6 | noncausal | Actual192-column SWA indices, padded 64 heads, no compressor/indexer |
| Context preparation | 1024 initially;7 in steady rounds | N/A | Main projection/norm, fused 1536-wide WKV, per-layer KV norm/RoPE/cache insertion |
| Boundary and sampling work | 1 request and 6 draft positions | N/A | Target taps, draft head collapse/norm/logits, sequential Markov stages; avoid overlap with measured layer scopes |

Retain request count, query width, actual positions/KV frontier, local and padded
heads, causal flag, compression/SWA role, cache layout/block size, graph policy,
runtime/image/config hashes, all-rank timing and selected kernel identity.
Cover the 1024-output E2E KV interval with an operator-only margin; interpolation
may vary KV within one physical identity, never query width or causality.

Validation: the complete DSpark scheme suite passed 28 tests, including new
Pro depth/projection/collective/KV/residency and V4.1 fallback regressions. New
GPU coverage and qualified accuracy remain pending. The corrected unqualified
smoke prediction is retained beside the original in
`smoke-pro-eager/prediction-pro-fix/`; the same workload and acceptance produce
76.64 ms TTFT and 2.43 ms TPOT, versus measured 325.57 ms and 29.22 ms.
