// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-FileCopyrightText: Copyright contributors to the vLLM project
// SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Optional token-delivery observations, not an HTTP or detokenization simulator.
//! Behavioral references (independent implementation; no upstream code copied):
//! vLLM b3563fc65ae0f5359802593d78e7ea097e1fed31, output_processor.py;
//! AgentX/AIPerf 754356e9a39acc6cc6afb242d123bb57c3fb6f75,
//! inter_token_latency_metric.py, request_latency_metric.py, and ttft_metric.py.
//! Modified semantic adaptation by NVIDIA; no upstream implementation copied.
//! https://github.com/vllm-project/vllm/blob/b3563fc65ae0f5359802593d78e7ea097e1fed31/vllm/v1/engine/output_processor.py
//! https://github.com/SemiAnalysisAI/agentx-harness/tree/754356e9a39acc6cc6afb242d123bb57c3fb6f75/src/aiperf/metrics/types
//! See the repository-root THIRD_PARTY_NOTICES.md for source/license details.

use std::num::NonZeroUsize;

use serde::Serialize;

use super::{StreamingDistribution, TraceDistributionStats};

/// Tokens delivered together at a modeled output boundary, without host costs.
#[derive(Debug, Clone, PartialEq, Serialize)]
pub struct StreamContentChunk {
    pub at_ms: f64,
    pub tokens: usize,
}

/// Optional per-request observations. Engine timestamps remain separate.
#[derive(Debug, Clone, Serialize)]
pub struct StreamRequestObservation {
    pub chunks: Vec<StreamContentChunk>,
    pub output_tokens: usize,
    pub first_content_ms: Option<f64>,
    pub last_content_ms: Option<f64>,
    pub ttft_ms: Option<f64>,
    pub ttst_ms: Option<f64>,
    pub e2e_latency_ms: Option<f64>,
    /// One average for this request, not a chunk interval or pooled token gap.
    pub request_mean_itl_ms: Option<f64>,
}

impl StreamRequestObservation {
    pub(super) fn rebase(&mut self, origin_ms: f64) {
        for chunk in &mut self.chunks {
            chunk.at_ms -= origin_ms;
        }
        for time in [&mut self.first_content_ms, &mut self.last_content_ms]
            .into_iter()
            .flatten()
        {
            *time -= origin_ms;
        }
    }
}

#[derive(Debug, Clone, Serialize)]
pub struct StreamLatencyDistribution {
    pub samples: u64,
    #[serde(flatten)]
    pub distribution: TraceDistributionStats,
}

/// Explicitly opt-in, modeled content-delivery metrics for completed, admitted
/// requests. All distributions use the collector's bounded DDSketch (0.1%).
#[derive(Debug, Clone, Serialize)]
pub struct StreamObservationReport {
    pub stream_interval: usize,
    pub boundary: &'static str,
    pub completed_requests_with_content: usize,
    pub output_tokens: usize,
    pub content_chunks: usize,
    pub ttft: StreamLatencyDistribution,
    pub ttst: StreamLatencyDistribution,
    pub e2e_latency: StreamLatencyDistribution,
    pub request_mean_itl: StreamLatencyDistribution,
    pub token_gap_itl: StreamLatencyDistribution,
    pub inter_chunk_latency: StreamLatencyDistribution,
}

#[derive(Debug)]
pub(super) struct StreamObservationCollector {
    pub(super) interval: NonZeroUsize,
    completed_requests: usize,
    output_tokens: usize,
    content_chunks: usize,
    ttft: StreamingDistribution,
    ttst: StreamingDistribution,
    e2e: StreamingDistribution,
    request_mean_itl: StreamingDistribution,
    token_gaps: StreamingDistribution,
    chunk_gaps: StreamingDistribution,
}

impl StreamObservationCollector {
    pub(super) fn new(interval: NonZeroUsize) -> Self {
        Self {
            interval,
            completed_requests: 0,
            output_tokens: 0,
            content_chunks: 0,
            ttft: StreamingDistribution::default(),
            ttst: StreamingDistribution::default(),
            e2e: StreamingDistribution::default(),
            request_mean_itl: StreamingDistribution::default(),
            token_gaps: StreamingDistribution::default(),
            chunk_gaps: StreamingDistribution::default(),
        }
    }

    pub(super) fn observe(
        &mut self,
        arrival_ms: f64,
        times: &[f64],
        successful_terminal_ms: Option<f64>,
        include_in_distributions: bool,
    ) -> StreamRequestObservation {
        let mut chunks = Vec::new();
        let mut pending = 0;
        // Native replay reports all tokens from an SD output at one completion
        // timestamp. Keep that whole burst atomic: never split at N tokens.
        for burst in times.chunk_by(|left, right| left == right) {
            pending += burst.len();
            if chunks.is_empty() || pending >= self.interval.get() {
                chunks.push(StreamContentChunk {
                    at_ms: burst[0],
                    tokens: pending,
                });
                pending = 0;
            }
        }
        // A successful final output flushes a short tail. Cancellation, failure,
        // or a replay cutoff is not a successful vLLM finish response.
        if pending > 0
            && let Some(terminal_ms) = successful_terminal_ms
        {
            chunks.push(StreamContentChunk {
                at_ms: terminal_ms,
                tokens: pending,
            });
        }
        let first_content_ms = chunks.first().map(|chunk| chunk.at_ms);
        let last_content_ms = chunks.last().map(|chunk| chunk.at_ms);
        let output_tokens: usize = chunks.iter().map(|chunk| chunk.tokens).sum();
        let ttst_ms = chunks.first().and_then(|first| {
            if first.tokens > 1 {
                Some(0.0)
            } else {
                chunks.get(1).map(|second| second.at_ms - first.at_ms)
            }
        });
        let observation = StreamRequestObservation {
            output_tokens,
            first_content_ms,
            last_content_ms,
            ttft_ms: first_content_ms.map(|first| first - arrival_ms),
            ttst_ms,
            e2e_latency_ms: last_content_ms.map(|last| last - arrival_ms),
            request_mean_itl_ms: first_content_ms
                .zip(last_content_ms)
                .and_then(|(first, last)| {
                    (output_tokens > 1).then(|| (last - first) / (output_tokens - 1) as f64)
                }),
            chunks,
        };
        if include_in_distributions && !times.is_empty() {
            self.completed_requests += 1;
            self.output_tokens += observation.output_tokens;
            self.content_chunks += observation.chunks.len();
            for (distribution, value) in [
                (&mut self.ttft, observation.ttft_ms),
                (&mut self.ttst, observation.ttst_ms),
                (&mut self.e2e, observation.e2e_latency_ms),
                (&mut self.request_mean_itl, observation.request_mean_itl_ms),
            ] {
                if let Some(value) = value {
                    distribution.add(value);
                }
            }
            for chunks in observation.chunks.windows(2) {
                let gap = chunks[1].at_ms - chunks[0].at_ms;
                self.chunk_gaps.add(gap);
                self.token_gaps.add(gap);
            }
            for chunk in &observation.chunks {
                for _ in 1..chunk.tokens {
                    self.token_gaps.add(0.0);
                }
            }
        }
        observation
    }

    pub(super) fn finish(&self) -> StreamObservationReport {
        let distribution = |source: &StreamingDistribution| StreamLatencyDistribution {
            samples: source.count,
            distribution: source.finish(),
        };
        StreamObservationReport {
            stream_interval: self.interval.get(),
            boundary: "modeled_content_delivery_without_host_or_network_cost",
            completed_requests_with_content: self.completed_requests,
            output_tokens: self.output_tokens,
            content_chunks: self.content_chunks,
            ttft: distribution(&self.ttft),
            ttst: distribution(&self.ttst),
            e2e_latency: distribution(&self.e2e),
            request_mean_itl: distribution(&self.request_mean_itl),
            token_gap_itl: distribution(&self.token_gaps),
            inter_chunk_latency: distribution(&self.chunk_gaps),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn collector(interval: usize) -> StreamObservationCollector {
        StreamObservationCollector::new(NonZeroUsize::new(interval).unwrap())
    }

    #[test]
    fn first_output_is_immediate_and_short_successful_tail_flushes() {
        // Hand oracle: 45 one-token outputs at 1..45 ms, interval 20.
        // First output is immediate, then 20-token, 20-token, 4-token chunks.
        let mut collector = collector(20);
        let times: Vec<_> = (1..=45).map(f64::from).collect();
        let observed = collector.observe(0.0, &times, Some(45.0), true);
        assert_eq!(
            observed.chunks,
            [(1.0, 1), (21.0, 20), (41.0, 20), (45.0, 4)]
                .map(|(at_ms, tokens)| StreamContentChunk { at_ms, tokens })
        );
        assert_eq!(observed.ttft_ms, Some(1.0));
        assert_eq!(observed.ttst_ms, Some(20.0));
        assert_eq!(observed.e2e_latency_ms, Some(45.0));
        assert_eq!(observed.request_mean_itl_ms, Some(1.0));
        let report = collector.finish();
        assert_eq!(report.token_gap_itl.samples, 44);
        assert!((report.token_gap_itl.distribution.mean_ms - 1.0).abs() < 1e-12);
        assert_eq!(report.inter_chunk_latency.samples, 3);
        assert!((report.inter_chunk_latency.distribution.mean_ms - 44.0 / 3.0).abs() < 1e-12);
    }

    #[test]
    fn speculative_bursts_are_atomic_even_when_crossing_threshold() {
        // Hand oracle: the first 4-token burst is immediate. Two 3-token
        // bursts cross interval 5 together; final 2-token burst flushes.
        let mut collector = collector(5);
        let observed = collector.observe(
            0.0,
            &[10., 10., 10., 10., 20., 20., 20., 30., 30., 30., 40., 40.],
            Some(40.0),
            true,
        );
        assert_eq!(
            observed.chunks,
            [(10.0, 4), (30.0, 6), (40.0, 2)]
                .map(|(at_ms, tokens)| StreamContentChunk { at_ms, tokens })
        );
        assert_eq!(observed.ttst_ms, Some(0.0));
        assert_eq!(observed.request_mean_itl_ms, Some(30.0 / 11.0));
        assert_eq!(collector.finish().token_gap_itl.samples, 11);
    }

    #[test]
    fn cancellation_and_cutoff_do_not_synthesize_successful_tail() {
        let mut collector = collector(20);
        let observed = collector.observe(0.0, &[1., 2., 3.], None, false);
        assert_eq!(observed.output_tokens, 1);
        assert_eq!(observed.last_content_ms, Some(1.0));
        assert_eq!(observed.request_mean_itl_ms, None);
        assert_eq!(collector.finish().completed_requests_with_content, 0);
    }

    #[test]
    fn one_output_and_empty_response_have_no_itl_sample() {
        let mut collector = collector(20);
        for times in [&[][..], &[1.0][..]] {
            let observed = collector.observe(0.0, times, Some(1.0), true);
            assert_eq!(observed.request_mean_itl_ms, None);
        }
        let report = collector.finish();
        assert_eq!(report.completed_requests_with_content, 1);
        assert_eq!(report.request_mean_itl.samples, 0);
        assert_eq!(report.token_gap_itl.samples, 0);
    }

    #[test]
    fn request_mean_distribution_is_not_token_weighted() {
        let mut collector = collector(1);
        collector.observe(0., &[1., 11.], Some(11.), true);
        collector.observe(0., &[1., 31., 61., 91.], Some(91.), true);
        // Two request means (10,30) versus four gaps (10,30,30,30).
        let report = collector.finish();
        assert_eq!(report.request_mean_itl.samples, 2);
        assert_eq!(report.request_mean_itl.distribution.mean_ms, 20.0);
        assert_eq!(report.token_gap_itl.samples, 4);
        assert_eq!(report.token_gap_itl.distribution.mean_ms, 25.0);
    }
}
