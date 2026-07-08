/// Deterministic pure-Rust exponential for f32 inputs.
///
/// This is vendored from rust-lang/libm 0.2.16 rather than calling platform
/// libm or Rust's standard transcendental methods.
pub fn exp_f32(x: f32) -> f32 {
    crate::vendored_libm::expf(x)
}

/// Deterministic pure-Rust exponential for f64 inputs.
pub fn exp_f64(x: f64) -> f64 {
    crate::vendored_libm::exp(x)
}

/// Deterministic pure-Rust natural logarithm for f64 inputs.
pub fn ln_f64(x: f64) -> f64 {
    crate::vendored_libm::log(x)
}

/// Deterministic pure-Rust sine for f64 inputs.
pub fn sin_f64(x: f64) -> f64 {
    crate::vendored_libm::sin(x)
}

/// Deterministic pure-Rust cosine for f64 inputs.
pub fn cos_f64(x: f64) -> f64 {
    crate::vendored_libm::cos(x)
}

#[inline]
pub fn silu_f32(x: f32) -> f32 {
    let t = exp_f32(-x);
    let d = 1.0 + t;
    x / d
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn exp_has_fixed_anchor_values() {
        assert_eq!(exp_f32(-88.0).to_bits(), 0x0041_edc4);
        assert_eq!(exp_f32(0.0).to_bits(), 1.0f32.to_bits());
        assert_eq!(exp_f32(1.0).to_bits(), 0x402d_f854);
        assert_eq!(exp_f32(-1.0).to_bits(), 0x3ebc_5ab2);
    }

    #[test]
    fn f64_transcendentals_have_fixed_anchor_values() {
        assert_eq!(exp_f64(1.0).to_bits(), 0x4005_bf0a_8b14_576a);
        assert_eq!(ln_f64(10_000.0).to_bits(), 0x4022_6bb1_bbb5_5516);
        assert_eq!(sin_f64(0.0).to_bits(), 0.0f64.to_bits());
        assert_eq!(cos_f64(0.0).to_bits(), 1.0f64.to_bits());
        assert_eq!(
            sin_f64(core::f64::consts::FRAC_PI_2).to_bits(),
            1.0f64.to_bits()
        );
        assert_eq!(
            cos_f64(core::f64::consts::FRAC_PI_2).to_bits(),
            0x3c91_a626_3314_5c07
        );
    }
}
