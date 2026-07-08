#![no_std]

extern crate alloc;

use det_num::{dot_f32_ref, f16_to_f32, round_ties_even_i32};

pub const BLOCK: usize = 32;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum QuantError {
    LengthMismatch,
    InvalidBlockLength,
    NonFiniteInput,
    NonFiniteScale,
    NonFiniteOutput,
}

#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Q8ABlock {
    pub d: f32,
    pub q: [i8; BLOCK],
}

#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Q8_0Block {
    pub d: f32,
    pub q: [i8; BLOCK],
}

#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Q4_0Block {
    pub d: f32,
    pub qs: [u8; 16],
}

pub fn quantize_q8a(input: &[f32]) -> Result<alloc::vec::Vec<Q8ABlock>, QuantError> {
    if input.is_empty() || input.len() % BLOCK != 0 {
        return Err(QuantError::InvalidBlockLength);
    }
    let mut out = alloc::vec::Vec::with_capacity(input.len() / BLOCK);
    for chunk in input.chunks_exact(BLOCK) {
        out.push(try_quantize_q8a_block(chunk)?);
    }
    Ok(out)
}

pub fn quantize_q8a_block(input: &[f32]) -> Q8ABlock {
    try_quantize_q8a_block(input).expect("Q8A input must be finite and exactly one block")
}

pub fn try_quantize_q8a_block(input: &[f32]) -> Result<Q8ABlock, QuantError> {
    if input.len() != BLOCK {
        return Err(QuantError::InvalidBlockLength);
    }
    let mut amax = 0.0f32;
    for &x in input {
        if !x.is_finite() {
            return Err(QuantError::NonFiniteInput);
        }
        let ax = if x < 0.0 { -x } else { x };
        if ax > amax {
            amax = ax;
        }
    }

    let mut q = [0i8; BLOCK];
    if amax == 0.0 {
        return Ok(Q8ABlock { d: 0.0, q });
    }

    let d = amax / 127.0;
    for (dst, &x) in q.iter_mut().zip(input) {
        let rounded = round_ties_even_i32(x / d).clamp(-127, 127);
        *dst = rounded as i8;
    }
    Ok(Q8ABlock { d, q })
}

pub fn q8_0_block_from_gguf(scale_f16: u16, q: [i8; BLOCK]) -> Result<Q8_0Block, QuantError> {
    let d = f16_to_f32(scale_f16);
    if !d.is_finite() {
        return Err(QuantError::NonFiniteScale);
    }
    Ok(Q8_0Block { d, q })
}

pub fn q4_0_block_from_gguf(scale_f16: u16, qs: [u8; 16]) -> Result<Q4_0Block, QuantError> {
    let d = f16_to_f32(scale_f16);
    if !d.is_finite() {
        return Err(QuantError::NonFiniteScale);
    }
    Ok(Q4_0Block { d, qs })
}

pub fn dot_q8_0_q8a(blocks_w: &[Q8_0Block], blocks_a: &[Q8ABlock]) -> Result<f32, QuantError> {
    if blocks_w.len() != blocks_a.len() {
        return Err(QuantError::LengthMismatch);
    }
    if blocks_w.is_empty() {
        return Err(QuantError::InvalidBlockLength);
    }
    let mut sum = 0.0f32;
    for (w, a) in blocks_w.iter().zip(blocks_a) {
        ensure_finite_scales(w.d, a.d)?;
        let block = dot_q8_0_q8a_block(*w, *a);
        if !block.is_finite() {
            return Err(QuantError::NonFiniteOutput);
        }
        sum += block;
        if !sum.is_finite() {
            return Err(QuantError::NonFiniteOutput);
        }
    }
    Ok(sum)
}

pub fn dot_q4_0_q8a(blocks_w: &[Q4_0Block], blocks_a: &[Q8ABlock]) -> Result<f32, QuantError> {
    if blocks_w.len() != blocks_a.len() {
        return Err(QuantError::LengthMismatch);
    }
    if blocks_w.is_empty() {
        return Err(QuantError::InvalidBlockLength);
    }
    let mut sum = 0.0f32;
    for (w, a) in blocks_w.iter().zip(blocks_a) {
        ensure_finite_scales(w.d, a.d)?;
        let block = dot_q4_0_q8a_block(*w, *a);
        if !block.is_finite() {
            return Err(QuantError::NonFiniteOutput);
        }
        sum += block;
        if !sum.is_finite() {
            return Err(QuantError::NonFiniteOutput);
        }
    }
    Ok(sum)
}

fn ensure_finite_scales(weight: f32, activation: f32) -> Result<(), QuantError> {
    if !weight.is_finite() || !activation.is_finite() {
        return Err(QuantError::NonFiniteScale);
    }
    Ok(())
}

#[inline]
pub fn dot_q8_0_q8a_block(w: Q8_0Block, a: Q8ABlock) -> f32 {
    #[cfg(all(feature = "simd", target_arch = "x86_64", target_feature = "avx2"))]
    {
        // SAFETY: this block is compiled only when AVX2 is enabled for the
        // target. The AVX2 path computes only the exact i32 dot sum; f32
        // scaling order is shared with the scalar path.
        unsafe { x86_64_avx2::dot_q8_0_q8a_block(w, a) }
    }
    #[cfg(all(feature = "simd", target_arch = "aarch64"))]
    {
        // SAFETY: NEON is part of the aarch64 baseline. The NEON path computes
        // only the exact i32 dot sum; f32 scaling order is shared.
        unsafe { aarch64_neon::dot_q8_0_q8a_block(w, a) }
    }
    #[cfg(all(feature = "simd", target_arch = "wasm32", target_feature = "simd128"))]
    {
        // SAFETY: this block is compiled only when non-relaxed wasm simd128 is
        // enabled. The SIMD path computes only the exact i32 dot sum.
        unsafe { wasm32_simd128::dot_q8_0_q8a_block(w, a) }
    }
    #[cfg(not(any(
        all(feature = "simd", target_arch = "x86_64", target_feature = "avx2"),
        all(feature = "simd", target_arch = "aarch64"),
        all(feature = "simd", target_arch = "wasm32", target_feature = "simd128")
    )))]
    {
        dot_q8_0_q8a_block_scalar(w, a)
    }
}

#[inline]
pub fn dot_q8_0_q8a_block_scalar(w: Q8_0Block, a: Q8ABlock) -> f32 {
    let mut isum = 0i32;
    for i in 0..BLOCK {
        isum += (w.q[i] as i32) * (a.q[i] as i32);
    }
    let scale = w.d * a.d;
    scale * (isum as f32)
}

#[inline]
pub fn dot_q4_0_q8a_block(w: Q4_0Block, a: Q8ABlock) -> f32 {
    #[cfg(all(feature = "simd", target_arch = "x86_64", target_feature = "avx2"))]
    {
        // SAFETY: this block is compiled only when AVX2 is enabled for the
        // target. The AVX2 path computes only the exact i32 dot sum; f32
        // scaling order is shared with the scalar path.
        unsafe { x86_64_avx2::dot_q4_0_q8a_block(w, a) }
    }
    #[cfg(all(feature = "simd", target_arch = "aarch64"))]
    {
        // SAFETY: NEON is part of the aarch64 baseline. The NEON path computes
        // only the exact i32 dot sum; f32 scaling order is shared.
        unsafe { aarch64_neon::dot_q4_0_q8a_block(w, a) }
    }
    #[cfg(all(feature = "simd", target_arch = "wasm32", target_feature = "simd128"))]
    {
        // SAFETY: this block is compiled only when non-relaxed wasm simd128 is
        // enabled. The SIMD path computes only the exact i32 dot sum.
        unsafe { wasm32_simd128::dot_q4_0_q8a_block(w, a) }
    }
    #[cfg(not(any(
        all(feature = "simd", target_arch = "x86_64", target_feature = "avx2"),
        all(feature = "simd", target_arch = "aarch64"),
        all(feature = "simd", target_arch = "wasm32", target_feature = "simd128")
    )))]
    {
        dot_q4_0_q8a_block_scalar(w, a)
    }
}

#[inline]
pub fn dot_q4_0_q8a_block_scalar(w: Q4_0Block, a: Q8ABlock) -> f32 {
    let mut isum = 0i32;
    for i in 0..BLOCK {
        let byte = w.qs[i / 2];
        let nibble = if i % 2 == 0 { byte & 0x0f } else { byte >> 4 };
        let q = (nibble as i32) - 8;
        isum += q * (a.q[i] as i32);
    }
    let scale = w.d * a.d;
    scale * (isum as f32)
}

#[cfg(all(feature = "simd", target_arch = "x86_64", target_feature = "avx2"))]
mod x86_64_avx2 {
    use super::{Q4_0Block, Q8ABlock, Q8_0Block, BLOCK};
    use core::arch::x86_64::{
        __m128i, __m256i, _mm256_add_epi32, _mm256_cvtepi8_epi16, _mm256_madd_epi16,
        _mm256_storeu_si256, _mm_loadu_si128,
    };

    #[target_feature(enable = "avx2")]
    pub unsafe fn dot_q8_0_q8a_block(w: Q8_0Block, a: Q8ABlock) -> f32 {
        let isum = dot_i8x32(&w.q, &a.q);
        let scale = w.d * a.d;
        scale * (isum as f32)
    }

    #[target_feature(enable = "avx2")]
    pub unsafe fn dot_q4_0_q8a_block(w: Q4_0Block, a: Q8ABlock) -> f32 {
        let mut q = [0i8; BLOCK];
        for (i, dst) in q.iter_mut().enumerate() {
            let byte = w.qs[i / 2];
            let nibble = if i % 2 == 0 { byte & 0x0f } else { byte >> 4 };
            *dst = (nibble as i8) - 8;
        }
        let isum = dot_i8x32(&q, &a.q);
        let scale = w.d * a.d;
        scale * (isum as f32)
    }

    #[target_feature(enable = "avx2")]
    unsafe fn dot_i8x32(x: &[i8; BLOCK], y: &[i8; BLOCK]) -> i32 {
        let x0 = _mm_loadu_si128(x.as_ptr().cast::<__m128i>());
        let x1 = _mm_loadu_si128(x.as_ptr().add(16).cast::<__m128i>());
        let y0 = _mm_loadu_si128(y.as_ptr().cast::<__m128i>());
        let y1 = _mm_loadu_si128(y.as_ptr().add(16).cast::<__m128i>());

        let x0 = _mm256_cvtepi8_epi16(x0);
        let x1 = _mm256_cvtepi8_epi16(x1);
        let y0 = _mm256_cvtepi8_epi16(y0);
        let y1 = _mm256_cvtepi8_epi16(y1);

        let prod0 = _mm256_madd_epi16(x0, y0);
        let prod1 = _mm256_madd_epi16(x1, y1);
        let sum = _mm256_add_epi32(prod0, prod1);
        horizontal_sum_i32x8(sum)
    }

    #[target_feature(enable = "avx2")]
    unsafe fn horizontal_sum_i32x8(v: __m256i) -> i32 {
        let mut lanes = [0i32; 8];
        _mm256_storeu_si256(lanes.as_mut_ptr().cast::<__m256i>(), v);
        lanes.iter().sum()
    }
}

#[cfg(all(feature = "simd", target_arch = "aarch64"))]
mod aarch64_neon {
    use super::{Q4_0Block, Q8ABlock, Q8_0Block, BLOCK};
    use core::arch::aarch64::{
        int32x4_t, vaddq_s32, vaddvq_s32, vdupq_n_s32, vld1_s8, vmull_s8, vpaddlq_s16,
    };

    pub unsafe fn dot_q8_0_q8a_block(w: Q8_0Block, a: Q8ABlock) -> f32 {
        let isum = dot_i8x32(&w.q, &a.q);
        let scale = w.d * a.d;
        scale * (isum as f32)
    }

    pub unsafe fn dot_q4_0_q8a_block(w: Q4_0Block, a: Q8ABlock) -> f32 {
        let mut q = [0i8; BLOCK];
        for (i, dst) in q.iter_mut().enumerate() {
            let byte = w.qs[i / 2];
            let nibble = if i % 2 == 0 { byte & 0x0f } else { byte >> 4 };
            *dst = (nibble as i8) - 8;
        }
        let isum = dot_i8x32(&q, &a.q);
        let scale = w.d * a.d;
        scale * (isum as f32)
    }

    unsafe fn dot_i8x32(x: &[i8; BLOCK], y: &[i8; BLOCK]) -> i32 {
        let mut acc = vdupq_n_s32(0);
        acc = add_i8x8_products(acc, x, y, 0);
        acc = add_i8x8_products(acc, x, y, 8);
        acc = add_i8x8_products(acc, x, y, 16);
        acc = add_i8x8_products(acc, x, y, 24);
        vaddvq_s32(acc)
    }

    unsafe fn add_i8x8_products(
        acc: int32x4_t,
        x: &[i8; BLOCK],
        y: &[i8; BLOCK],
        offset: usize,
    ) -> int32x4_t {
        let x8 = vld1_s8(x.as_ptr().add(offset));
        let y8 = vld1_s8(y.as_ptr().add(offset));
        let products = vmull_s8(x8, y8);
        vaddq_s32(acc, vpaddlq_s16(products))
    }
}

#[cfg(all(feature = "simd", target_arch = "wasm32", target_feature = "simd128"))]
mod wasm32_simd128 {
    use super::{Q4_0Block, Q8ABlock, Q8_0Block, BLOCK};
    use core::arch::wasm32::{
        i16x8_extmul_high_i8x16, i16x8_extmul_low_i8x16, i32x4_add, i32x4_extadd_pairwise_i16x8,
        i32x4_extract_lane, v128, v128_load,
    };

    pub unsafe fn dot_q8_0_q8a_block(w: Q8_0Block, a: Q8ABlock) -> f32 {
        let isum = dot_i8x32(&w.q, &a.q);
        let scale = w.d * a.d;
        scale * (isum as f32)
    }

    pub unsafe fn dot_q4_0_q8a_block(w: Q4_0Block, a: Q8ABlock) -> f32 {
        let mut q = [0i8; BLOCK];
        for (i, dst) in q.iter_mut().enumerate() {
            let byte = w.qs[i / 2];
            let nibble = if i % 2 == 0 { byte & 0x0f } else { byte >> 4 };
            *dst = (nibble as i8) - 8;
        }
        let isum = dot_i8x32(&q, &a.q);
        let scale = w.d * a.d;
        scale * (isum as f32)
    }

    unsafe fn dot_i8x32(x: &[i8; BLOCK], y: &[i8; BLOCK]) -> i32 {
        let x0 = v128_load(x.as_ptr().cast::<v128>());
        let x1 = v128_load(x.as_ptr().add(16).cast::<v128>());
        let y0 = v128_load(y.as_ptr().cast::<v128>());
        let y1 = v128_load(y.as_ptr().add(16).cast::<v128>());
        let sum0 = dot_i8x16(x0, y0);
        let sum1 = dot_i8x16(x1, y1);
        horizontal_sum_i32x4(i32x4_add(sum0, sum1))
    }

    fn dot_i8x16(x: v128, y: v128) -> v128 {
        let lo = i16x8_extmul_low_i8x16(x, y);
        let hi = i16x8_extmul_high_i8x16(x, y);
        i32x4_add(
            i32x4_extadd_pairwise_i16x8(lo),
            i32x4_extadd_pairwise_i16x8(hi),
        )
    }

    fn horizontal_sum_i32x4(v: v128) -> i32 {
        i32x4_extract_lane::<0>(v)
            + i32x4_extract_lane::<1>(v)
            + i32x4_extract_lane::<2>(v)
            + i32x4_extract_lane::<3>(v)
    }
}

pub fn gemv_f32_row_major(
    rows: usize,
    cols: usize,
    weights: &[f32],
    x: &[f32],
    y: &mut [f32],
) -> Result<(), QuantError> {
    let expected_weights = rows
        .checked_mul(cols)
        .ok_or(QuantError::InvalidBlockLength)?;
    if rows == 0
        || cols == 0
        || weights.len() != expected_weights
        || x.len() != cols
        || y.len() != rows
    {
        return Err(QuantError::LengthMismatch);
    }
    if weights.iter().chain(x).any(|v| !v.is_finite()) {
        return Err(QuantError::NonFiniteInput);
    }
    for r in 0..rows {
        let row = &weights[r * cols..(r + 1) * cols];
        y[r] = dot_f32_ref(row, x);
        if !y[r].is_finite() {
            return Err(QuantError::NonFiniteOutput);
        }
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn q8a_quantizes_ties_even_and_zero() {
        let zero = [0.0f32; BLOCK];
        let q = quantize_q8a_block(&zero);
        assert_eq!(q.d.to_bits(), 0.0f32.to_bits());
        assert_eq!(q.q, [0i8; BLOCK]);

        let mut input = [0.0f32; BLOCK];
        input[0] = 1.0;
        input[1] = -1.0;
        input[2] = 0.5 / 127.0;
        let q = quantize_q8a_block(&input);
        assert_eq!(q.q[0], 127);
        assert_eq!(q.q[1], -127);
        assert_eq!(q.q[2], 0);
    }

    #[test]
    fn q8a_rejects_nonfinite_and_bad_block_lengths() {
        assert_eq!(quantize_q8a(&[]), Err(QuantError::InvalidBlockLength));
        assert_eq!(
            try_quantize_q8a_block(&[0.0; BLOCK - 1]),
            Err(QuantError::InvalidBlockLength)
        );

        let mut input = [0.0f32; BLOCK];
        input[3] = f32::NAN;
        assert_eq!(
            try_quantize_q8a_block(&input),
            Err(QuantError::NonFiniteInput)
        );
        assert_eq!(quantize_q8a(&input), Err(QuantError::NonFiniteInput));

        input[3] = f32::INFINITY;
        assert_eq!(
            try_quantize_q8a_block(&input),
            Err(QuantError::NonFiniteInput)
        );
    }

    #[test]
    fn quantized_dot_rejects_empty_block_lists() {
        assert_eq!(dot_q8_0_q8a(&[], &[]), Err(QuantError::InvalidBlockLength));
        assert_eq!(dot_q4_0_q8a(&[], &[]), Err(QuantError::InvalidBlockLength));
    }

    #[test]
    fn quantized_dot_rejects_nonfinite_public_block_scales() {
        let a = Q8ABlock {
            d: 0.25,
            q: [2; BLOCK],
        };
        let q8 = Q8_0Block {
            d: 0.5,
            q: [-3; BLOCK],
        };
        let q4 = Q4_0Block {
            d: 0.5,
            qs: [0x88; 16],
        };

        let mut bad_a = a;
        bad_a.d = f32::NAN;
        assert_eq!(
            dot_q8_0_q8a(&[q8], &[bad_a]),
            Err(QuantError::NonFiniteScale)
        );
        assert_eq!(
            dot_q4_0_q8a(&[q4], &[bad_a]),
            Err(QuantError::NonFiniteScale)
        );

        let mut bad_q8 = q8;
        bad_q8.d = f32::INFINITY;
        assert_eq!(
            dot_q8_0_q8a(&[bad_q8], &[a]),
            Err(QuantError::NonFiniteScale)
        );

        let mut bad_q4 = q4;
        bad_q4.d = f32::NEG_INFINITY;
        assert_eq!(
            dot_q4_0_q8a(&[bad_q4], &[a]),
            Err(QuantError::NonFiniteScale)
        );
    }

    #[test]
    fn quantized_dot_rejects_nonfinite_outputs_from_finite_scales() {
        let huge_a = Q8ABlock {
            d: f32::MAX,
            q: [127; BLOCK],
        };
        let q8 = Q8_0Block {
            d: 2.0,
            q: [127; BLOCK],
        };
        let q4 = Q4_0Block {
            d: 2.0,
            qs: [0xff; 16],
        };

        assert_eq!(
            dot_q8_0_q8a(&[q8], &[huge_a]),
            Err(QuantError::NonFiniteOutput)
        );
        assert_eq!(
            dot_q4_0_q8a(&[q4], &[huge_a]),
            Err(QuantError::NonFiniteOutput)
        );

        let large_a = Q8ABlock {
            d: f32::MAX / 64.0,
            q: [1; BLOCK],
        };
        let large_q8 = Q8_0Block {
            d: 1.0,
            q: [1; BLOCK],
        };
        assert!(dot_q8_0_q8a(&[large_q8], &[large_a])
            .expect("single block stays finite")
            .is_finite());
        assert_eq!(
            dot_q8_0_q8a(
                &[large_q8, large_q8, large_q8],
                &[large_a, large_a, large_a]
            ),
            Err(QuantError::NonFiniteOutput)
        );
    }

    #[test]
    fn q8_dot_uses_block_sequential_f32_add() {
        let a = Q8ABlock {
            d: 0.25,
            q: [2; BLOCK],
        };
        let w = Q8_0Block {
            d: 0.5,
            q: [-3; BLOCK],
        };
        let one = dot_q8_0_q8a_block(w, a);
        assert_eq!(one.to_bits(), (-24.0f32).to_bits());
        assert_eq!(dot_q8_0_q8a(&[w, w], &[a, a]).expect("same len"), -48.0);
    }

    #[test]
    fn f32_gemv_row_major_reports_shape_and_nonfinite_errors() {
        let weights = [1.0, 2.0, 3.0, 4.0];
        let x = [5.0, 6.0];
        let mut y = [0.0, 0.0];
        gemv_f32_row_major(2, 2, &weights, &x, &mut y).expect("gemv");
        assert_eq!(y.map(f32::to_bits), [17.0f32.to_bits(), 39.0f32.to_bits()]);

        assert_eq!(
            gemv_f32_row_major(0, 2, &weights, &x, &mut y),
            Err(QuantError::LengthMismatch)
        );
        assert_eq!(
            gemv_f32_row_major(2, 2, &weights[..3], &x, &mut y),
            Err(QuantError::LengthMismatch)
        );

        let bad_x = [f32::NAN, 1.0];
        assert_eq!(
            gemv_f32_row_major(2, 2, &weights, &bad_x, &mut y),
            Err(QuantError::NonFiniteInput)
        );

        let huge = [f32::MAX, f32::MAX];
        let mut y1 = [0.0];
        assert_eq!(
            gemv_f32_row_major(1, 2, &huge, &[2.0, 2.0], &mut y1),
            Err(QuantError::NonFiniteOutput)
        );
    }

    #[cfg(any(
        all(feature = "simd", target_arch = "x86_64", target_feature = "avx2"),
        all(feature = "simd", target_arch = "aarch64"),
        all(feature = "simd", target_arch = "wasm32", target_feature = "simd128")
    ))]
    #[test]
    fn simd_blocks_match_scalar_bits() {
        for seed in 0..1024u32 {
            let q8 = q8_block(seed);
            let q4 = q4_block(seed ^ 0xa5a5_5a5a);
            let a = q8a_block(seed.wrapping_mul(1_664_525).wrapping_add(1_013_904_223));

            let q8_simd = dot_q8_0_q8a_block(q8, a);
            let q8_scalar = dot_q8_0_q8a_block_scalar(q8, a);
            assert_eq!(q8_simd.to_bits(), q8_scalar.to_bits());

            let q4_simd = dot_q4_0_q8a_block(q4, a);
            let q4_scalar = dot_q4_0_q8a_block_scalar(q4, a);
            assert_eq!(q4_simd.to_bits(), q4_scalar.to_bits());
        }
    }

    #[cfg(any(
        all(feature = "simd", target_arch = "x86_64", target_feature = "avx2"),
        all(feature = "simd", target_arch = "aarch64"),
        all(feature = "simd", target_arch = "wasm32", target_feature = "simd128")
    ))]
    fn q8_block(seed: u32) -> Q8_0Block {
        let mut q = [0i8; BLOCK];
        let mut x = seed;
        for dst in &mut q {
            x = x.wrapping_mul(1_103_515_245).wrapping_add(12_345);
            *dst = (((x >> 16) % 255) as i16 - 127) as i8;
        }
        Q8_0Block {
            d: f32::from_bits(0x3c00_0000 + (seed & 0xff)),
            q,
        }
    }

    #[cfg(any(
        all(feature = "simd", target_arch = "x86_64", target_feature = "avx2"),
        all(feature = "simd", target_arch = "aarch64"),
        all(feature = "simd", target_arch = "wasm32", target_feature = "simd128")
    ))]
    fn q8a_block(seed: u32) -> Q8ABlock {
        let mut q = [0i8; BLOCK];
        let mut x = seed;
        for dst in &mut q {
            x = x.wrapping_mul(22_695_477).wrapping_add(1);
            *dst = (((x >> 17) % 255) as i16 - 127) as i8;
        }
        Q8ABlock {
            d: f32::from_bits(0x3d00_0000 + (seed & 0xff)),
            q,
        }
    }

    #[cfg(any(
        all(feature = "simd", target_arch = "x86_64", target_feature = "avx2"),
        all(feature = "simd", target_arch = "aarch64"),
        all(feature = "simd", target_arch = "wasm32", target_feature = "simd128")
    ))]
    fn q4_block(seed: u32) -> Q4_0Block {
        let mut qs = [0u8; 16];
        let mut x = seed;
        for byte in &mut qs {
            x = x.wrapping_mul(747_796_405).wrapping_add(2_891_336_453);
            let lo = ((x >> 8) & 0x0f) as u8;
            let hi = ((x >> 20) & 0x0f) as u8;
            *byte = lo | (hi << 4);
        }
        Q4_0Block {
            d: f32::from_bits(0x3c80_0000 + (seed & 0xff)),
            qs,
        }
    }
}
