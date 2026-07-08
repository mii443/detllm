use det_num::{exp_f32, sum_f32_ref};

const M: f32 = 16_777_216.0;
pub const MAX_SYMBOLS: usize = 1 << 18;

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum CdfError {
    Empty,
    Malformed,
    NonFiniteLogit,
    TooManySymbols,
    TotalTooLarge,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Cdf {
    pub freq: Vec<u32>,
    pub cum: Vec<u64>,
    pub total: u64,
}

pub fn logits_to_cdf(logits: &[f32]) -> Result<Cdf, CdfError> {
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

    let mut e = Vec::with_capacity(logits.len());
    for &x in logits {
        e.push(exp_f32((x - max).max(-88.0)));
    }
    let z = sum_f32_ref(&e);

    let mut freq = Vec::with_capacity(logits.len());
    let mut cum = Vec::with_capacity(logits.len());
    let mut total = 0u64;
    for ei in e {
        let p = ei / z;
        let g = (p * M) as u32;
        let f = g + 1;
        cum.push(total);
        freq.push(f);
        total += f as u64;
    }

    if total >= (1u64 << 31) {
        return Err(CdfError::TotalTooLarge);
    }
    Ok(Cdf { freq, cum, total })
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

    pub(crate) fn symbol_for_validated(&self, value: u64) -> Option<usize> {
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
