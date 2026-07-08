#![no_std]

pub mod canary;
pub mod f16;
pub mod hash;
pub mod math;
pub mod reduce;
pub mod round;
mod vendored_libm;

pub use canary::{run_canary, CanaryError};
pub use f16::{f16_is_finite, f16_to_f32};
pub use hash::Sha256;
pub use math::{cos_f64, exp_f32, exp_f64, ln_f64, silu_f32, sin_f64};
pub use reduce::{dot_f32_ref, sum_f32_ref, sum_squares_f32_ref};
pub use round::{round_ties_even_i32, round_ties_even_i64};
