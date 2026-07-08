#[inline]
pub fn dot_f32_ref(x: &[f32], y: &[f32]) -> f32 {
    assert_eq!(x.len(), y.len());
    let mut acc = [0.0f32; 8];
    let mut i = 0usize;
    while i < x.len() {
        acc[i & 7] += x[i] * y[i];
        i += 1;
    }
    finish_8lane(acc)
}

#[inline]
pub fn sum_f32_ref(x: &[f32]) -> f32 {
    let mut acc = [0.0f32; 8];
    let mut i = 0usize;
    while i < x.len() {
        acc[i & 7] += x[i];
        i += 1;
    }
    finish_8lane(acc)
}

#[inline]
pub fn sum_squares_f32_ref(x: &[f32]) -> f32 {
    let mut acc = [0.0f32; 8];
    let mut i = 0usize;
    while i < x.len() {
        acc[i & 7] += x[i] * x[i];
        i += 1;
    }
    finish_8lane(acc)
}

#[inline]
pub fn finish_8lane(acc: [f32; 8]) -> f32 {
    ((acc[0] + acc[1]) + (acc[2] + acc[3])) + ((acc[4] + acc[5]) + (acc[6] + acc[7]))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn dot_uses_normative_lane_order() {
        let x: [f32; 17] = [
            1.0, 2.0, -3.0, 4.0, 5.5, -6.25, 7.0, 8.0, -9.0, 10.0, 11.0, -12.0, 13.0, 14.0, -15.0,
            16.0, 17.0,
        ];
        let y: [f32; 17] = [
            0.5, -1.0, 2.0, 3.0, -4.0, 5.0, 6.0, -7.0, 8.0, 9.0, -10.0, 11.0, 12.0, -13.0, 14.0,
            15.0, -16.0,
        ];
        let mut acc = [0.0f32; 8];
        for i in 0..x.len() {
            acc[i % 8] += x[i] * y[i];
        }
        let expected =
            ((acc[0] + acc[1]) + (acc[2] + acc[3])) + ((acc[4] + acc[5]) + (acc[6] + acc[7]));
        assert_eq!(dot_f32_ref(&x, &y).to_bits(), expected.to_bits());
    }
}
