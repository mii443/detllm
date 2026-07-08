#[inline]
pub fn f16_is_finite(bits: u16) -> bool {
    (bits & 0x7c00) != 0x7c00
}

pub fn f16_to_f32(bits: u16) -> f32 {
    let sign = ((bits as u32) & 0x8000) << 16;
    let exp = (bits >> 10) & 0x1f;
    let frac = (bits & 0x03ff) as u32;

    let out = match exp {
        0 => {
            if frac == 0 {
                sign
            } else {
                let mut f = frac;
                let mut e = -14i32;
                while (f & 0x0400) == 0 {
                    f <<= 1;
                    e -= 1;
                }
                f &= 0x03ff;
                let exp32 = ((e + 127) as u32) << 23;
                sign | exp32 | (f << 13)
            }
        }
        0x1f => sign | 0x7f80_0000 | (frac << 13),
        _ => {
            let exp32 = (((exp as i32) - 15 + 127) as u32) << 23;
            sign | exp32 | (frac << 13)
        }
    };
    f32::from_bits(out)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn converts_common_values() {
        assert_eq!(f16_to_f32(0x0000).to_bits(), 0.0f32.to_bits());
        assert_eq!(f16_to_f32(0x8000).to_bits(), (-0.0f32).to_bits());
        assert_eq!(f16_to_f32(0x3c00).to_bits(), 1.0f32.to_bits());
        assert_eq!(f16_to_f32(0xc000).to_bits(), (-2.0f32).to_bits());
        assert_eq!(f16_to_f32(0x7bff), 65504.0);
        assert!(!f16_is_finite(0x7c00));
        assert!(!f16_is_finite(0x7e00));
    }
}
