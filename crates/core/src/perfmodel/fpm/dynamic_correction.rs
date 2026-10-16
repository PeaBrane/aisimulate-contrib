// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Online correction with observation-driven, expanding log-space bounds.
//!
//! A balanced local sample set supplies bucket medians; a separate recent FIFO
//! supplies the global fallback, including outside the observed bounds. Samples
//! can occur in both sets. Their combined slot budget is bounded, and samples
//! expire after twice that budget in admitted observations. All rebucketing and
//! statistics happen on admission, so prediction neither scans samples nor
//! changes geometry.

use std::collections::{BTreeMap, VecDeque};

use super::options::ForwardPassPerfOptions;
use super::samples::integer_sqrt;

const SHRINKAGE_OBSERVATIONS: f64 = 20.0;
type BucketKey = [usize; 2];

#[derive(Clone, Copy, Debug, PartialEq)]
struct Observation {
    admission: usize,
    log_x: [f64; 2],
    ratio: f64,
}

#[derive(Clone, Debug, PartialEq)]
pub(crate) struct DynamicCorrection {
    ndim: usize,
    bins: [usize; 2],
    log_bounds: Option<[[f64; 2]; 2]>,
    primary: Vec<Observation>,
    recent: VecDeque<Observation>,
    primary_capacity: usize,
    recent_capacity: usize,
    max_age: usize,
    next_admission: usize,
    min_observations: usize,
    lower_factor: Option<f64>,
    upper_factor: Option<f64>,
    fallback: f64,
    factors: BTreeMap<BucketKey, f64>,
}

impl DynamicCorrection {
    pub(crate) fn new(options: &ForwardPassPerfOptions, ndim: usize) -> Self {
        assert!((1..=2).contains(&ndim));
        let shape = options.bucket_shape.unwrap_or_else(|| {
            let width = integer_sqrt(options.bucket_count);
            [width, width]
        });
        let bins = if ndim == 1 {
            [
                options
                    .bucket_shape
                    .map_or(options.bucket_count, |shape| shape[0] * shape[1]),
                1,
            ]
        } else {
            shape
        };
        // Reserve a quarter for recent evidence without making a valid
        // workload-kind readiness threshold unreachable at small budgets.
        let recent_capacity = (options.max_observations / 4).min(
            options
                .max_observations
                .saturating_sub(options.min_observations),
        );
        Self {
            ndim,
            bins,
            log_bounds: None,
            primary: Vec::new(),
            recent: VecDeque::new(),
            primary_capacity: options.max_observations - recent_capacity,
            recent_capacity,
            max_age: options.max_observations.saturating_mul(2),
            next_admission: 0,
            min_observations: options.min_observations,
            lower_factor: options.min_faster_correction_factor,
            upper_factor: options.max_slower_correction_factor,
            fallback: 1.0,
            factors: BTreeMap::new(),
        }
    }

    pub(crate) fn add_observation(&mut self, x: Vec<f64>, observed_ms: f64, base_ms: f64) {
        let Some(log_x) = self.log_coordinates(&x) else {
            return;
        };
        if !observed_ms.is_finite() || observed_ms <= 0.0 || !base_ms.is_finite() || base_ms <= 0.0
        {
            return;
        }
        let ratio = observed_ms / base_ms;
        // Reject overflow and underflow before clipping. Invalid evidence must
        // not grow bounds, advance sample age, or contaminate the fallback.
        if !ratio.is_finite() || ratio <= 0.0 {
            return;
        }
        let observation = Observation {
            admission: self.next_admission,
            log_x,
            ratio: self.project(ratio),
        };
        self.next_admission = self.next_admission.saturating_add(1);
        let oldest = self.next_admission.saturating_sub(self.max_age);
        self.primary.retain(|row| row.admission >= oldest);
        while self
            .recent
            .front()
            .is_some_and(|row| row.admission < oldest)
        {
            self.recent.pop_front();
        }
        self.primary.push(observation);
        if self.recent_capacity > 0 {
            self.recent.push_back(observation);
            if self.recent.len() > self.recent_capacity {
                self.recent.pop_front();
            }
        }

        let bounds = self
            .log_bounds
            .get_or_insert([[log_x[0]; 2], [log_x[1]; 2]]);
        for axis in 0..self.ndim {
            bounds[axis][0] = bounds[axis][0].min(log_x[axis]);
            bounds[axis][1] = bounds[axis][1].max(log_x[axis]);
        }
        self.rebuild_factors();
    }

    pub(crate) fn correction_factor_for(&self, x: &[f64]) -> f64 {
        let Some(log_x) = self.log_coordinates(x) else {
            return 1.0;
        };
        self.bucket_key(log_x)
            .and_then(|key| self.factors.get(&key).copied())
            .unwrap_or(self.fallback)
    }

    /// Number of distinct retained observations, excluding duplicate FIFO slots.
    pub(crate) fn observation_count(&self) -> usize {
        // The recent FIFO contains every admitted ID from its oldest entry to
        // the newest. Primary rows before that interval are the only extras.
        self.recent.front().map_or(self.primary.len(), |oldest| {
            self.recent.len()
                + self
                    .primary
                    .iter()
                    .take_while(|row| row.admission < oldest.admission)
                    .count()
        })
    }

    pub(crate) fn is_ready(&self) -> bool {
        self.primary.len() >= self.min_observations || self.recent_is_ready()
    }

    pub(crate) fn ready_bucket_count(&self) -> usize {
        self.factors.len()
    }

    pub(crate) fn correction_factors(&self) -> Vec<f64> {
        let mut factors = self.factors.values().copied().collect::<Vec<_>>();
        if self.recent_is_ready() {
            factors.push(self.fallback);
        }
        factors
    }

    fn recent_is_ready(&self) -> bool {
        self.recent.len() >= self.min_observations
    }

    fn log_coordinates(&self, x: &[f64]) -> Option<[f64; 2]> {
        if x.len() != self.ndim || x.iter().any(|value| !value.is_finite() || *value < 0.0) {
            return None;
        }
        let mut log_x = [0.0; 2];
        for (target, value) in log_x.iter_mut().zip(x) {
            *target = value.ln_1p();
        }
        Some(log_x)
    }

    fn bucket_key(&self, log_x: [f64; 2]) -> Option<BucketKey> {
        let bounds = self.log_bounds?;
        let mut key = [0; 2];
        for axis in 0..self.ndim {
            let [lo, hi] = bounds[axis];
            if log_x[axis] < lo || log_x[axis] > hi {
                return None;
            }
            if hi > lo {
                key[axis] = (((log_x[axis] - lo) / (hi - lo) * self.bins[axis] as f64) as usize)
                    .min(self.bins[axis] - 1);
            }
        }
        Some(key)
    }

    fn project(&self, factor: f64) -> f64 {
        let factor = self.lower_factor.map_or(factor, |lower| factor.max(lower));
        self.upper_factor.map_or(factor, |upper| factor.min(upper))
    }

    fn rebuild_factors(&mut self) {
        let mut buckets: BTreeMap<BucketKey, Vec<f64>> = BTreeMap::new();
        for row in &self.primary {
            let key = self
                .bucket_key(row.log_x)
                .expect("admitted row is in bounds");
            buckets.entry(key).or_default().push(row.ratio);
        }
        if self.primary.len() > self.primary_capacity {
            // BTreeMap order and strict size comparison choose the
            // lexicographically first fullest bucket. Primary admission order
            // survives rebucketing, so its first matching row is the oldest.
            let mut fattest = ([0; 2], 0);
            for (&key, values) in &buckets {
                if values.len() > fattest.1 {
                    fattest = (key, values.len());
                }
            }
            let index = self
                .primary
                .iter()
                .position(|row| self.bucket_key(row.log_x) == Some(fattest.0))
                .expect("fullest bucket contains an observation");
            self.primary.remove(index);
            let values = buckets.get_mut(&fattest.0).expect("bucket exists");
            values.remove(0);
            if values.is_empty() {
                buckets.remove(&fattest.0);
            }
        }

        self.fallback = if self.recent_is_ready() {
            median(&mut self.recent.iter().map(|row| row.ratio).collect::<Vec<_>>())
        } else {
            1.0
        };
        self.factors.clear();
        if self.primary.len() < self.min_observations {
            return;
        }
        for (key, mut values) in buckets {
            let support = values.len() as f64;
            let estimate = median(&mut values);
            let weight = support / (support + SHRINKAGE_OBSERVATIONS);
            let factor = (weight * estimate.ln() + (1.0 - weight) * self.fallback.ln()).exp();
            // A convex combination in log-space lies between its endpoints.
            // Enforce this property at representable extremes, where exp can
            // otherwise round to zero or infinity when bounds are disabled.
            let factor = factor.clamp(estimate.min(self.fallback), estimate.max(self.fallback));
            self.factors.insert(key, self.project(factor));
        }
    }
}

fn median(values: &mut [f64]) -> f64 {
    debug_assert!(!values.is_empty());
    values.sort_unstable_by(f64::total_cmp);
    let middle = values.len() / 2;
    if values.len() % 2 == 1 {
        values[middle]
    } else {
        let sum = values[middle - 1] + values[middle];
        if sum.is_finite() {
            sum / 2.0
        } else {
            values[middle - 1] / 2.0 + values[middle] / 2.0
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn options(capacity: usize, minimum: usize) -> ForwardPassPerfOptions {
        ForwardPassPerfOptions {
            max_observations: capacity,
            min_observations: minimum,
            ..Default::default()
        }
    }

    fn close(actual: f64, expected: f64) {
        assert!(
            (actual - expected).abs() <= 2e-14 * expected.abs().max(1.0),
            "{actual} != {expected}"
        );
    }

    #[test]
    fn default_budget_and_hand_computed_log_shrinkage() {
        let mut correction = DynamicCorrection::new(&ForwardPassPerfOptions::default(), 1);
        assert_eq!(correction.primary_capacity, 48);
        assert_eq!(correction.recent_capacity, 16);
        assert_eq!(correction.max_age, 128);
        for _ in 0..20 {
            correction.add_observation(vec![1.0], 20.0, 10.0);
        }
        for _ in 0..16 {
            correction.add_observation(vec![1000.0], 5.0, 10.0);
        }
        // Twenty local ratios of 2 have equal shrinkage weight to the
        // sixteen recent global ratios of 0.5: sqrt(2 * 0.5) = 1.
        close(correction.correction_factor_for(&[1.0]), 1.0);
        close(correction.correction_factor_for(&[1000.0]), 0.5);
        close(correction.correction_factor_for(&[1e9]), 0.5);
        // An unoccupied in-range bucket also uses the recent global median.
        close(correction.correction_factor_for(&[30.0]), 0.5);
        assert_eq!(correction.observation_count(), 36);
        assert_eq!(correction.ready_bucket_count(), 2);
        assert_eq!(correction.correction_factors().len(), 3);
    }

    #[test]
    fn bounds_grow_and_old_coordinates_are_rebucketed() {
        let mut correction = DynamicCorrection::new(&options(64, 1), 2);
        correction.add_observation(vec![0.0, 0.0], 1.0, 1.0);
        correction.add_observation(vec![15.0, 15.0], 2.0, 1.0);
        assert_eq!(correction.bucket_key([15_f64.ln_1p(); 2]), Some([3, 3]));
        correction.add_observation(vec![65535.0, 65535.0], 1.0, 1.0);
        // ln(16) / ln(65536) = 1/4, so the old upper endpoint moves to
        // bucket one instead of remaining in bucket three.
        assert_eq!(correction.bucket_key([15_f64.ln_1p(); 2]), Some([1, 1]));
        assert!(correction.factors.contains_key(&[1, 1]));
        assert_eq!(correction.primary.len(), 3);
        assert_eq!(correction.observation_count(), 3);
        let bounds = correction.log_bounds.unwrap();
        assert_eq!(bounds[0], [0.0, 65535_f64.ln_1p()]);
        assert_eq!(bounds[1], bounds[0]);
    }

    #[test]
    fn balanced_eviction_uses_lexicographic_ties_and_oldest_admission() {
        let mut correction = DynamicCorrection::new(&options(4, 1), 1);
        // Primary capacity is three. Equal counts in the lower and upper
        // buckets evict the lower bucket's oldest row, despite the upper
        // bucket containing the globally oldest row.
        for x in [100.0, 0.0, 100.0, 0.0] {
            correction.add_observation(vec![x], 1.0, 1.0);
        }
        assert_eq!(
            correction
                .primary
                .iter()
                .map(|row| row.admission)
                .collect::<Vec<_>>(),
            [0, 2, 3]
        );
        correction.add_observation(vec![100.0], 1.0, 1.0);
        assert_eq!(
            correction
                .primary
                .iter()
                .map(|row| row.admission)
                .collect::<Vec<_>>(),
            [2, 3, 4]
        );
        assert_eq!(correction.observation_count(), 3);
    }

    #[test]
    fn count_includes_recent_evidence_evicted_from_primary_without_duplicates() {
        let mut correction = DynamicCorrection::new(&options(4, 1), 1);
        for x in [100.0, 1000.0, 10000.0, 0.0] {
            correction.add_observation(vec![x], 2.0, 1.0);
        }
        // Four singleton buckets tie. The newest sample is alone in the
        // lowest bucket, so it is evicted locally and kept only by the FIFO.
        assert_eq!(
            correction
                .primary
                .iter()
                .map(|row| row.admission)
                .collect::<Vec<_>>(),
            [0, 1, 2]
        );
        assert_eq!(correction.recent.front().unwrap().admission, 3);
        assert_eq!(correction.observation_count(), 4);
        assert_eq!(correction.correction_factor_for(&[0.0]), 2.0);
    }

    #[test]
    fn expiry_removes_rare_old_evidence_without_shrinking_bounds() {
        let mut correction = DynamicCorrection::new(&options(4, 1), 1);
        correction.add_observation(vec![0.0], 2.0, 1.0);
        for _ in 0..7 {
            correction.add_observation(vec![100.0], 1.0, 1.0);
        }
        assert!(correction.primary.iter().any(|row| row.admission == 0));
        let bounds = correction.log_bounds;
        correction.add_observation(vec![100.0], 1.0, 1.0);
        assert!(correction.primary.iter().all(|row| row.admission > 0));
        assert_eq!(correction.log_bounds, bounds);
        close(correction.correction_factor_for(&[0.0]), 1.0);
    }

    #[test]
    fn invalid_evidence_and_prediction_leave_all_state_unchanged() {
        let mut correction = DynamicCorrection::new(&options(64, 1), 2);
        correction.add_observation(vec![1.0, 100.0], 2.0, 1.0);
        let saved = correction.clone();
        for (x, target, base) in [
            (vec![1.0], 1.0, 1.0),
            (vec![-1.0, 1.0], 1.0, 1.0),
            (vec![f64::NAN, 1.0], 1.0, 1.0),
            (vec![f64::INFINITY, 1.0], 1.0, 1.0),
            (vec![1e9, 1e9], f64::NAN, 1.0),
            (vec![1e9, 1e9], f64::INFINITY, 1.0),
            (vec![1e9, 1e9], 1.0, f64::INFINITY),
            (vec![1e9, 1e9], 1.0, f64::NAN),
            (vec![1e9, 1e9], 0.0, 1.0),
            (vec![1e9, 1e9], 1.0, 0.0),
            (vec![1e9, 1e9], -1.0, 1.0),
            (vec![1e9, 1e9], 1.0, -1.0),
            (vec![1e9, 1e9], f64::MAX, f64::MIN_POSITIVE),
            (vec![1e9, 1e9], f64::MIN_POSITIVE, f64::MAX),
        ] {
            correction.add_observation(x, target, base);
            assert_eq!(correction, saved);
        }
        for x in [&[1.0, 100.0][..], &[1e9, 1e9], &[0.0, 0.0]] {
            assert!(correction.correction_factor_for(x).is_finite());
            assert_eq!(correction, saved);
        }
        for x in [&[f64::NAN, 1.0][..], &[1.0], &[-1.0, 1.0]] {
            assert_eq!(correction.correction_factor_for(x), 1.0);
            assert_eq!(correction, saved);
        }
    }

    #[test]
    fn readiness_uses_primary_count_and_fallback_uses_its_own_support() {
        let mut correction = DynamicCorrection::new(&options(16, 5), 1);
        for _ in 0..4 {
            correction.add_observation(vec![100.0], 2.0, 1.0);
        }
        assert!(!correction.is_ready());
        assert_eq!(correction.ready_bucket_count(), 0);
        assert_eq!(correction.correction_factor_for(&[100.0]), 1.0);
        correction.add_observation(vec![100.0], 2.0, 1.0);
        assert!(correction.is_ready());
        // Fallback capacity four cannot reach min_observations five. Five
        // local samples give weight 5/(5+20) = 1/5 toward ratio two.
        close(
            correction.correction_factor_for(&[100.0]),
            1.148698354997035,
        );
        assert_eq!(correction.correction_factor_for(&[101.0]), 1.0);
    }

    #[test]
    fn factor_summary_includes_ready_fallback_extrema_without_extra_local_bucket() {
        let mut correction = DynamicCorrection::new(&ForwardPassPerfOptions::default(), 1);
        for _ in 0..48 {
            correction.add_observation(vec![100.0], 0.5, 1.0);
        }
        for _ in 0..16 {
            correction.add_observation(vec![100.0], 2.0, 1.0);
        }
        // Local retention still has 32 faster observations and 16 slower
        // observations, whereas all recent observations have ratio two.
        assert!(correction.correction_factor_for(&[100.0]) < 1.0);
        assert_eq!(correction.correction_factor_for(&[101.0]), 2.0);
        let factors = correction.correction_factors();
        assert_eq!(factors.len(), 2);
        assert_eq!(factors.iter().copied().fold(0.0, f64::max), 2.0);
        assert_eq!(correction.ready_bucket_count(), 1);

        let mut unsupported = DynamicCorrection::new(&options(16, 5), 1);
        for _ in 0..5 {
            unsupported.add_observation(vec![100.0], 2.0, 1.0);
        }
        assert_eq!(unsupported.correction_factors().len(), 1);
        assert_eq!(unsupported.ready_bucket_count(), 1);
    }

    #[test]
    fn ratios_are_clipped_before_median_and_custom_shapes_are_respected() {
        let mut configured = options(64, 1);
        configured.bucket_shape = Some([2, 3]);
        let mut correction = DynamicCorrection::new(&configured, 2);
        correction.add_observation(vec![0.0, 0.0], 0.001, 1.0);
        correction.add_observation(vec![100.0, 100.0], 1000.0, 1.0);
        assert_eq!(correction.bins, [2, 3]);
        assert_eq!(correction.bucket_key([100_f64.ln_1p(); 2]), Some([1, 2]));
        assert_eq!(correction.primary[0].ratio, 0.5);
        assert_eq!(correction.primary[1].ratio, 2.0);
        assert_eq!(correction.fallback, 1.25);
        assert_eq!(DynamicCorrection::new(&configured, 1).bins, [6, 1]);
    }

    #[test]
    fn smallest_budgets_remain_bounded_and_can_learn() {
        for capacity in 1..=4 {
            let mut correction = DynamicCorrection::new(&options(capacity, 1), 1);
            for _ in 0..20 {
                correction.add_observation(vec![5.0], 1.5, 1.0);
                assert!(correction.primary.len() + correction.recent.len() <= capacity);
                assert!(correction.observation_count() <= capacity);
                assert!(correction.correction_factor_for(&[5.0]) > 1.0);
            }
            assert!(correction.is_ready());
        }
        let mut correction = DynamicCorrection::new(&options(5, 5), 1);
        for _ in 0..5 {
            correction.add_observation(vec![5.0], 1.5, 1.0);
        }
        assert_eq!(correction.primary_capacity, 5);
        assert_eq!(correction.recent_capacity, 0);
        assert!(correction.is_ready());
        assert!(correction.correction_factor_for(&[5.0]) > 1.0);
    }

    #[test]
    fn unbounded_extreme_finite_ratios_remain_finite_and_positive() {
        let mut configured = options(16, 1);
        configured.min_faster_correction_factor = None;
        configured.max_slower_correction_factor = None;
        for ratio in [f64::MAX, f64::from_bits(1)] {
            let mut correction = DynamicCorrection::new(&configured, 1);
            for _ in 0..2 {
                correction.add_observation(vec![1.0], ratio, 1.0);
            }
            for factor in [
                correction.fallback,
                correction.correction_factor_for(&[1.0]),
                correction.correction_factor_for(&[1e9]),
            ] {
                assert!(factor.is_finite() && factor > 0.0);
                assert_eq!(factor, ratio);
            }
        }
    }
}
