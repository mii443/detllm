#[inline]
pub fn round_ties_even_i32(x: f32) -> i32 {
    if !x.is_finite() {
        return x as i32;
    }
    if !(-8_388_608.0..8_388_608.0).contains(&x) {
        return x as i32;
    }

    let t = x as i32;
    let frac = x - (t as f32);
    if frac > 0.5 || (frac == 0.5 && (t & 1) != 0) {
        t + 1
    } else if frac < -0.5 || (frac == -0.5 && (t & 1) != 0) {
        t - 1
    } else {
        t
    }
}

#[inline]
pub fn round_ties_even_i64(x: f64) -> i64 {
    if !x.is_finite() {
        return x as i64;
    }
    if !(-4_503_599_627_370_496.0..4_503_599_627_370_496.0).contains(&x) {
        return x as i64;
    }

    let t = x as i64;
    let frac = x - (t as f64);
    if frac > 0.5 || (frac == 0.5 && (t & 1) != 0) {
        t + 1
    } else if frac < -0.5 || (frac == -0.5 && (t & 1) != 0) {
        t - 1
    } else {
        t
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn rounds_half_to_even() {
        assert_eq!(round_ties_even_i32(0.5), 0);
        assert_eq!(round_ties_even_i32(1.5), 2);
        assert_eq!(round_ties_even_i32(2.5), 2);
        assert_eq!(round_ties_even_i32(-0.5), 0);
        assert_eq!(round_ties_even_i32(-1.5), -2);
        assert_eq!(round_ties_even_i32(-2.5), -2);
        assert_eq!(round_ties_even_i64(2.5), 2);
        assert_eq!(round_ties_even_i64(3.5), 4);
        assert_eq!(round_ties_even_i64(-2.5), -2);
        assert_eq!(round_ties_even_i64(-3.5), -4);
    }
}
