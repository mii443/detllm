//! Vendored deterministic math routines.
//!
//! Source: rust-lang/libm crate version 0.2.16, selected files from
//! `src/math`. The crate metadata declares MIT licensing; individual fdlibm
//! source notices are preserved in the copied files.
//!
//! Local changes:
//! - architecture dispatch macros are disabled so there is only one pure-Rust
//!   implementation path;
//! - helper macros use safe indexing to satisfy this workspace's no-unsafe
//!   policy.

#![allow(clippy::all)]

macro_rules! force_eval {
    ($e:expr) => {{
        let value = $e;
        value
    }};
}

macro_rules! i {
    ($array:expr, $index:expr) => {
        *$array.get($index).expect("vendored libm index")
    };
    ($array:expr, $index:expr, =, $rhs:expr) => {
        *$array.get_mut($index).expect("vendored libm index") = $rhs;
    };
    ($array:expr, $index:expr, -=, $rhs:expr) => {
        *$array.get_mut($index).expect("vendored libm index") -= $rhs;
    };
    ($array:expr, $index:expr, +=, $rhs:expr) => {
        *$array.get_mut($index).expect("vendored libm index") += $rhs;
    };
    ($array:expr, $index:expr, &=, $rhs:expr) => {
        *$array.get_mut($index).expect("vendored libm index") &= $rhs;
    };
    ($array:expr, $index:expr, ==, $rhs:expr) => {
        *$array.get($index).expect("vendored libm index") == $rhs
    };
}

macro_rules! select_implementation {
    (name: $name:ident, use_arch_required: $feature:ident, args: $($args:expr),* $(,)?) => {};
}

macro_rules! div {
    ($a:expr, $b:expr) => {
        $a / $b
    };
}

mod cos;
mod exp;
mod expf;
mod k_cos;
mod k_sin;
mod log;
mod rem_pio2;
mod rem_pio2_large;
mod sin;

pub(crate) use cos::cos;
pub(crate) use exp::exp;
pub(crate) use expf::expf;
pub(crate) use k_cos::k_cos;
pub(crate) use k_sin::k_sin;
pub(crate) use log::log;
pub(crate) use rem_pio2::rem_pio2;
pub(crate) use rem_pio2_large::rem_pio2_large;
pub(crate) use sin::sin;

pub(crate) fn scalbnf(x: f32, n: i32) -> f32 {
    scalbn_f32(x, n)
}

pub(crate) fn scalbn(x: f64, n: i32) -> f64 {
    scalbn_f64(x, n)
}

pub(crate) fn floor(x: f64) -> f64 {
    if x.is_nan() || x.is_infinite() || x == 0.0 {
        return x;
    }
    let truncated = x as i64;
    let y = truncated as f64;
    if y > x {
        (truncated - 1) as f64
    } else {
        y
    }
}

fn scalbn_f32(mut x: f32, mut n: i32) -> f32 {
    let x1p127 = f32::from_bits(0x7f000000);
    let x1p_126 = f32::from_bits(0x00800000);
    if n > 127 {
        x *= x1p127;
        n -= 127;
        if n > 127 {
            x *= x1p127;
            n -= 127;
            if n > 127 {
                n = 127;
            }
        }
    } else if n < -126 {
        x *= x1p_126;
        n += 126;
        if n < -126 {
            x *= x1p_126;
            n += 126;
            if n < -126 {
                n = -126;
            }
        }
    }
    x * f32::from_bits(((0x7f + n) as u32) << 23)
}

fn scalbn_f64(mut x: f64, mut n: i32) -> f64 {
    let x1p1023 = f64::from_bits(0x7fe0000000000000);
    let x1p_1022 = f64::from_bits(0x0010000000000000);
    if n > 1023 {
        x *= x1p1023;
        n -= 1023;
        if n > 1023 {
            x *= x1p1023;
            n -= 1023;
            if n > 1023 {
                n = 1023;
            }
        }
    } else if n < -1022 {
        x *= x1p_1022;
        n += 1022;
        if n < -1022 {
            x *= x1p_1022;
            n += 1022;
            if n < -1022 {
                n = -1022;
            }
        }
    }
    x * f64::from_bits(((0x3ff + n) as u64) << 52)
}
