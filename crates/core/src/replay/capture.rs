// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

use serde::{Deserialize, Serialize};

/// Seed used by placement implementations that support canonical worker
/// selection. Random replay never receives this seed.
pub const CANONICAL_SELECTOR_SEED: u64 = 0xd1a0_5eed;

/// Identity and selection semantics for one replay execution.
#[derive(Clone, Copy, Debug, Default, Eq, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum ReplayDeterminism {
    /// Preserve normal runtime randomness: fresh random request IDs, so synthetic
    /// outputs vary between runs, and no canonical selector seed.
    #[default]
    Random,
    /// Use ordinal request UUIDs and the canonical selector seed.
    CanonicalV1,
}

impl ReplayDeterminism {
    /// Return a selector seed only when canonical replay was explicitly
    /// requested. This prevents deterministic benchmark policy from leaking
    /// into ordinary simulations.
    pub const fn selector_seed(self) -> Option<u64> {
        match self {
            Self::Random => None,
            Self::CanonicalV1 => Some(CANONICAL_SELECTOR_SEED),
        }
    }
}

/// Replay-owned controls for optional detailed capture.
#[derive(Clone, Copy, Debug, Default, Eq, PartialEq, Serialize, Deserialize)]
pub struct ReplayCaptureOptions {
    /// Optional modeled content-output batching. Does not delay engine work,
    /// terminals, or load generation; the first output and final tail flush.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub stream_interval: Option<std::num::NonZeroUsize>,
    #[serde(default)]
    pub capture_per_request: bool,
    #[serde(default)]
    pub capture_lifecycle_evidence: bool,
    #[serde(default)]
    pub capture_canonical_evidence: bool,
    #[serde(default)]
    pub determinism: ReplayDeterminism,
}

impl ReplayCaptureOptions {
    /// Canonical evidence references request records, so enabling it implies
    /// per-request capture even if the caller did not request a separate
    /// per-request JSONL file.
    pub const fn effective_per_request(self) -> bool {
        self.capture_per_request || self.capture_canonical_evidence
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn stream_interval_is_positive_and_absent_by_default() {
        let defaults = serde_json::to_value(ReplayCaptureOptions::default()).unwrap();
        assert!(defaults.get("stream_interval").is_none());
        for value in [
            serde_json::json!(0),
            serde_json::json!(-1),
            serde_json::json!(true),
            serde_json::json!(1.5),
        ] {
            assert!(
                serde_json::from_value::<ReplayCaptureOptions>(serde_json::json!({
                    "stream_interval": value,
                }))
                .is_err()
            );
        }
        let options: ReplayCaptureOptions = serde_json::from_value(serde_json::json!({
            "stream_interval": 20,
        }))
        .unwrap();
        assert_eq!(options.stream_interval.unwrap().get(), 20);
    }
}
