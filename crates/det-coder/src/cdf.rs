use det_num::{exp_f32, sum_f32_ref};
#[cfg(feature = "parallel")]
use rayon::prelude::*;

const M: f32 = 16_777_216.0;
pub const MAX_SYMBOLS: usize = 1 << 18;
pub const BYTE_ESCAPE_SYMBOLS: usize = 256;

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum CdfError {
    Empty,
    Malformed,
    NonFiniteLogit,
    TooManySymbols,
    TotalTooLarge,
}

#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub struct Cdf {
    pub freq: Vec<u32>,
    pub cum: Vec<u64>,
    pub total: u64,
}

#[derive(Debug, Default)]
pub struct CdfScratch {
    exp: Vec<f32>,
    cdf: Cdf,
}

pub fn logits_to_cdf(logits: &[f32]) -> Result<Cdf, CdfError> {
    let mut scratch = CdfScratch::default();
    logits_to_cdf_with_scratch(logits, &mut scratch).cloned()
}

pub fn logits_to_cdf_with_scratch<'a>(
    logits: &[f32],
    scratch: &'a mut CdfScratch,
) -> Result<&'a Cdf, CdfError> {
    if logits.is_empty() {
        return Err(CdfError::Empty);
    }
    if logits.len() > MAX_SYMBOLS {
        return Err(CdfError::TooManySymbols);
    }
    let mut max = f32::NEG_INFINITY;
    for &x in logits {
        if !x.is_finite() {
            return Err(CdfError::NonFiniteLogit);
        }
        if x > max {
            max = x;
        }
    }

    fill_exp_scratch(logits, max, &mut scratch.exp);
    let z = sum_f32_ref(&scratch.exp);

    scratch.cdf.freq.clear();
    scratch.cdf.freq.reserve(logits.len());
    scratch.cdf.cum.clear();
    scratch.cdf.cum.reserve(logits.len());
    let mut total = 0u64;
    for &ei in &scratch.exp {
        let p = ei / z;
        let g = (p * M) as u32;
        let f = g + 1;
        scratch.cdf.cum.push(total);
        scratch.cdf.freq.push(f);
        total += f as u64;
    }

    if total >= (1u64 << 31) {
        return Err(CdfError::TotalTooLarge);
    }
    scratch.cdf.total = total;
    Ok(&scratch.cdf)
}

pub fn logits_to_cdf_with_byte_escapes<'a>(
    logits: &[f32],
    scratch: &'a mut CdfScratch,
) -> Result<&'a Cdf, CdfError> {
    if logits.len().saturating_add(BYTE_ESCAPE_SYMBOLS) > MAX_SYMBOLS {
        return Err(CdfError::TooManySymbols);
    }
    logits_to_cdf_with_scratch(logits, scratch)?;
    append_uniform_tail(&mut scratch.cdf, BYTE_ESCAPE_SYMBOLS, 1)?;
    Ok(&scratch.cdf)
}

pub fn uniform_cdf_with_byte_escapes(vocab_len: usize) -> Result<Cdf, CdfError> {
    uniform_cdf(
        vocab_len
            .checked_add(BYTE_ESCAPE_SYMBOLS)
            .ok_or(CdfError::TooManySymbols)?,
    )
}

fn append_uniform_tail(cdf: &mut Cdf, symbols: usize, freq: u32) -> Result<(), CdfError> {
    if cdf.freq.len().saturating_add(symbols) > MAX_SYMBOLS {
        return Err(CdfError::TooManySymbols);
    }
    if freq == 0 {
        return Err(CdfError::Malformed);
    }
    let add_total = (symbols as u64)
        .checked_mul(freq as u64)
        .ok_or(CdfError::TotalTooLarge)?;
    if cdf
        .total
        .checked_add(add_total)
        .ok_or(CdfError::TotalTooLarge)?
        >= (1u64 << 31)
    {
        return Err(CdfError::TotalTooLarge);
    }
    for _ in 0..symbols {
        cdf.cum.push(cdf.total);
        cdf.freq.push(freq);
        cdf.total += freq as u64;
    }
    Ok(())
}

fn fill_exp_scratch(logits: &[f32], max: f32, exp: &mut Vec<f32>) {
    exp.resize(logits.len(), 0.0);
    fill_exp_slice(logits, max, exp);
}

#[cfg(feature = "parallel")]
fn fill_exp_slice(logits: &[f32], max: f32, exp: &mut [f32]) {
    exp.par_iter_mut()
        .zip(logits.par_iter())
        .for_each(|(dst, &x)| {
            *dst = exp_f32((x - max).max(-88.0));
        });
}

#[cfg(not(feature = "parallel"))]
fn fill_exp_slice(logits: &[f32], max: f32, exp: &mut [f32]) {
    for (dst, &x) in exp.iter_mut().zip(logits) {
        *dst = exp_f32((x - max).max(-88.0));
    }
}

pub fn uniform_cdf(symbols: usize) -> Result<Cdf, CdfError> {
    if symbols == 0 {
        return Err(CdfError::Empty);
    }
    if symbols > MAX_SYMBOLS {
        return Err(CdfError::TooManySymbols);
    }
    let total = symbols as u64;
    if total >= (1u64 << 31) {
        return Err(CdfError::TotalTooLarge);
    }
    let mut freq = Vec::with_capacity(symbols);
    let mut cum = Vec::with_capacity(symbols);
    for i in 0..symbols {
        freq.push(1);
        cum.push(i as u64);
    }
    Ok(Cdf { freq, cum, total })
}

impl Cdf {
    pub fn validate(&self) -> Result<(), CdfError> {
        if self.freq.is_empty() || self.cum.is_empty() || self.freq.len() != self.cum.len() {
            return Err(CdfError::Malformed);
        }
        if self.freq.len() > MAX_SYMBOLS {
            return Err(CdfError::TooManySymbols);
        }
        if self.total == 0 || self.total >= (1u64 << 31) {
            return Err(CdfError::TotalTooLarge);
        }
        let mut total = 0u64;
        for (&cum, &freq) in self.cum.iter().zip(&self.freq) {
            if freq == 0 || cum != total {
                return Err(CdfError::Malformed);
            }
            total = total
                .checked_add(freq as u64)
                .ok_or(CdfError::TotalTooLarge)?;
        }
        if total != self.total {
            return Err(CdfError::Malformed);
        }
        Ok(())
    }

    pub fn symbol_for(&self, value: u64) -> Option<usize> {
        self.validate().ok()?;
        self.symbol_for_validated(value)
    }

    /// Look up a symbol in a CDF that the caller already knows is valid.
    ///
    /// This avoids the full O(vocabulary) validation scan in hot decode paths
    /// where the CDF was just built by this crate or validated by the caller.
    pub fn symbol_for_validated(&self, value: u64) -> Option<usize> {
        if self.cum.is_empty() || self.cum.len() != self.freq.len() || value >= self.total {
            return None;
        }
        let idx = self.cum.partition_point(|&c| c <= value);
        Some(idx.saturating_sub(1))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn cdf_assigns_minimum_frequency() {
        let cdf = logits_to_cdf(&[0.0, -1000.0, -1.0]).expect("cdf");
        assert_eq!(cdf.freq.len(), 3);
        assert!(cdf.freq.iter().all(|&f| f >= 1));
        assert_eq!(cdf.cum[0], 0);
        assert_eq!(cdf.total, cdf.freq.iter().map(|&f| f as u64).sum());
        for i in 0..cdf.freq.len() {
            assert_eq!(cdf.symbol_for(cdf.cum[i]), Some(i));
        }
    }

    #[test]
    fn logits_to_cdf_does_not_redistribute_to_fixed_total() {
        let cdf = logits_to_cdf(&[0.0, 0.0, 0.0, 0.0]).expect("cdf");
        let expected_freq = ((M / 4.0) as u32) + 1;
        assert_eq!(cdf.freq, [expected_freq; 4]);
        assert_eq!(
            cdf.cum,
            [
                0,
                expected_freq as u64,
                (expected_freq as u64) * 2,
                (expected_freq as u64) * 3,
            ]
        );
        assert_eq!(cdf.total, (M as u64) + 4);
    }

    #[test]
    fn logits_to_cdf_scratch_matches_owned_api_and_reuses_buffers() {
        let logits = [0.0, 1.5, -2.0, 0.25];
        let expected = logits_to_cdf(&logits).expect("owned cdf");
        let mut scratch = CdfScratch::default();
        let got = logits_to_cdf_with_scratch(&logits, &mut scratch).expect("scratch cdf");
        assert_eq!(got, &expected);

        let second = logits_to_cdf_with_scratch(&[4.0, -1.0], &mut scratch).expect("second cdf");
        assert_eq!(second.freq.len(), 2);
        assert_eq!(second.cum.len(), 2);
        assert_eq!(second.total, second.freq.iter().map(|&f| f as u64).sum());
    }

    #[test]
    fn byte_escape_cdf_appends_minimum_frequency_tail() {
        let mut scratch = CdfScratch::default();
        let base = logits_to_cdf(&[0.0, 1.0]).expect("base cdf");
        let cdf = logits_to_cdf_with_byte_escapes(&[0.0, 1.0], &mut scratch).expect("escape cdf");
        assert_eq!(cdf.freq.len(), 2 + BYTE_ESCAPE_SYMBOLS);
        assert_eq!(&cdf.freq[..2], base.freq.as_slice());
        assert_eq!(cdf.freq[2..], [1u32; BYTE_ESCAPE_SYMBOLS]);
        assert_eq!(cdf.cum[2], base.total);
        assert_eq!(cdf.total, base.total + BYTE_ESCAPE_SYMBOLS as u64);

        let uniform = uniform_cdf_with_byte_escapes(2).expect("uniform cdf");
        assert_eq!(uniform.freq.len(), 2 + BYTE_ESCAPE_SYMBOLS);
        assert!(uniform.freq.iter().all(|&freq| freq == 1));
    }

    #[test]
    fn byte_escape_cdf_enforces_symbol_limit() {
        let max_vocab = MAX_SYMBOLS - BYTE_ESCAPE_SYMBOLS;
        let mut scratch = CdfScratch::default();

        let uniform = uniform_cdf_with_byte_escapes(max_vocab).expect("max uniform cdf");
        assert_eq!(uniform.freq.len(), MAX_SYMBOLS);
        assert_eq!(uniform.total, MAX_SYMBOLS as u64);
        assert_eq!(
            uniform_cdf_with_byte_escapes(max_vocab + 1),
            Err(CdfError::TooManySymbols)
        );

        let logits = vec![0.0; max_vocab];
        let cdf = logits_to_cdf_with_byte_escapes(&logits, &mut scratch).expect("max logits cdf");
        assert_eq!(cdf.freq.len(), MAX_SYMBOLS);
        assert_eq!(cdf.cum[max_vocab], cdf.total - BYTE_ESCAPE_SYMBOLS as u64);
        assert!(cdf.freq[max_vocab..].iter().all(|&freq| freq == 1));

        let too_many_logits = vec![0.0; max_vocab + 1];
        assert_eq!(
            logits_to_cdf_with_byte_escapes(&too_many_logits, &mut scratch),
            Err(CdfError::TooManySymbols)
        );
    }

    #[test]
    fn logits_to_cdf_matches_scalar_reference_for_large_vocab() {
        let mut state = 0x1234_5678_9abc_def0u64;
        let mut logits = Vec::with_capacity(10_003);
        for _ in 0..10_003 {
            state = state
                .wrapping_mul(6_364_136_223_846_793_005)
                .wrapping_add(1_442_695_040_888_963_407);
            let unit = ((state >> 40) as f32) / ((1u32 << 24) as f32);
            logits.push(unit * 40.0 - 20.0);
        }

        let expected = logits_to_cdf_scalar_reference(&logits).expect("reference");
        let actual = logits_to_cdf(&logits).expect("actual");
        assert_eq!(actual, expected);
    }

    fn logits_to_cdf_scalar_reference(logits: &[f32]) -> Result<Cdf, CdfError> {
        if logits.is_empty() {
            return Err(CdfError::Empty);
        }
        if logits.len() > MAX_SYMBOLS {
            return Err(CdfError::TooManySymbols);
        }
        let mut max = f32::NEG_INFINITY;
        for &x in logits {
            if !x.is_finite() {
                return Err(CdfError::NonFiniteLogit);
            }
            if x > max {
                max = x;
            }
        }

        let mut exp = Vec::with_capacity(logits.len());
        for &x in logits {
            exp.push(det_num::exp_f32((x - max).max(-88.0)));
        }
        let z = det_num::sum_f32_ref(&exp);

        let mut freq = Vec::with_capacity(logits.len());
        let mut cum = Vec::with_capacity(logits.len());
        let mut total = 0u64;
        for &ei in &exp {
            let p = ei / z;
            let f = ((p * M) as u32) + 1;
            cum.push(total);
            freq.push(f);
            total += f as u64;
        }
        if total >= (1u64 << 31) {
            return Err(CdfError::TotalTooLarge);
        }
        Ok(Cdf { freq, cum, total })
    }

    #[test]
    fn uniform_cdf_has_one_count_per_symbol() {
        let cdf = uniform_cdf(4).expect("cdf");
        assert_eq!(cdf.freq, [1, 1, 1, 1]);
        assert_eq!(cdf.cum, [0, 1, 2, 3]);
        assert_eq!(cdf.total, 4);
        assert_eq!(cdf.symbol_for(2), Some(2));
    }

    #[test]
    fn cdf_rejects_symbol_counts_outside_design_limit() {
        assert_eq!(
            logits_to_cdf(&vec![0.0; MAX_SYMBOLS + 1]),
            Err(CdfError::TooManySymbols)
        );
        assert_eq!(uniform_cdf(MAX_SYMBOLS + 1), Err(CdfError::TooManySymbols));
        assert_eq!(
            uniform_cdf(MAX_SYMBOLS).expect("cdf").total,
            MAX_SYMBOLS as u64
        );

        let oversized = Cdf {
            freq: vec![1; MAX_SYMBOLS + 1],
            cum: (0..=MAX_SYMBOLS as u64).collect(),
            total: (MAX_SYMBOLS + 1) as u64,
        };
        assert_eq!(oversized.validate(), Err(CdfError::TooManySymbols));
    }

    #[test]
    fn cdf_validate_rejects_malformed_tables() {
        assert_eq!(
            Cdf {
                freq: vec![1],
                cum: vec![0],
                total: 1u64 << 31,
            }
            .validate(),
            Err(CdfError::TotalTooLarge)
        );

        for cdf in [
            Cdf {
                freq: Vec::new(),
                cum: Vec::new(),
                total: 0,
            },
            Cdf {
                freq: vec![1],
                cum: Vec::new(),
                total: 1,
            },
            Cdf {
                freq: vec![1, 0],
                cum: vec![0, 1],
                total: 1,
            },
            Cdf {
                freq: vec![1, 1],
                cum: vec![0, 2],
                total: 2,
            },
            Cdf {
                freq: vec![1, 1],
                cum: vec![0, 1],
                total: 3,
            },
        ] {
            assert!(cdf.validate().is_err());
        }
    }

    #[test]
    fn cdf_symbol_for_rejects_malformed_tables() {
        assert_eq!(
            Cdf {
                freq: vec![1],
                cum: vec![1],
                total: 2,
            }
            .symbol_for(1),
            None
        );
        assert_eq!(
            Cdf {
                freq: vec![1, 1],
                cum: vec![0, 1],
                total: 3,
            }
            .symbol_for(1),
            None
        );
        assert_eq!(
            Cdf {
                freq: vec![1],
                cum: vec![0],
                total: 1u64 << 31,
            }
            .symbol_for(0),
            None
        );
    }
}
