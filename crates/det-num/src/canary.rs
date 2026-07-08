use crate::{dot_f32_ref, exp_f32, f16_to_f32, round_ties_even_i32, Sha256};

const EXPECTED: [u8; 32] = [
    0x2e, 0x42, 0xe5, 0x50, 0x7c, 0xc1, 0xbf, 0xd7, 0x11, 0x1a, 0x5a, 0x0f, 0x3c, 0x28, 0x05, 0x6d,
    0x66, 0x9e, 0x05, 0xea, 0x0f, 0x62, 0x7d, 0x1b, 0x92, 0xd9, 0x2e, 0xc7, 0xfe, 0xc1, 0x7a, 0x5d,
];

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct CanaryError {
    pub expected: [u8; 32],
    pub actual: [u8; 32],
}

pub fn run_canary() -> Result<(), CanaryError> {
    let actual = canary_hash();
    if actual == EXPECTED {
        Ok(())
    } else {
        Err(CanaryError {
            expected: EXPECTED,
            actual,
        })
    }
}

pub fn canary_hash() -> [u8; 32] {
    let x = [
        f32::from_bits(1),
        f32::from_bits(2),
        -0.0,
        1.0,
        -2.5,
        3.25,
        1024.0,
        -4096.0,
        0.125,
    ];
    let y = [2.0, -3.0, 4.0, 0.5, -8.0, 16.0, -0.25, 0.03125, -64.0];
    let values = [
        dot_f32_ref(&x, &y),
        exp_f32(-1.0),
        exp_f32(0.0),
        f16_to_f32(0x3555),
        round_ties_even_i32(2.5) as f32,
        f32::from_bits(1) + f32::from_bits(1),
    ];

    let mut h = Sha256::new();
    for v in values {
        h.update(&v.to_bits().to_le_bytes());
    }
    h.finalize()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn canary_matches_current_kernels() {
        assert_eq!(canary_hash(), EXPECTED);
        run_canary().expect("canary must match");
    }
}
