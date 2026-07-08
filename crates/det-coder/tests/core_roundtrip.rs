use det_coder::range::{RangeDecoder, RangeEncoder};

#[test]
fn range_coder_round_trips_deterministic_pattern() {
    let freq = [5u64, 1, 9, 2, 7];
    let mut cum = [0u64; 5];
    for i in 1..freq.len() {
        cum[i] = cum[i - 1] + freq[i - 1];
    }
    let total: u64 = freq.iter().sum();
    let symbols: Vec<usize> = (0..2048).map(|i| (i * 17 + i / 3) % freq.len()).collect();

    let mut enc = RangeEncoder::new();
    for &s in &symbols {
        enc.encode(cum[s], freq[s], total).expect("encode");
    }
    let bytes = enc.finish();

    let mut dec = RangeDecoder::new(&bytes).expect("decoder");
    let mut decoded = Vec::with_capacity(symbols.len());
    for _ in 0..symbols.len() {
        let u = dec.decode_freq(total).expect("decode freq");
        let s = cum.partition_point(|&c| c <= u).saturating_sub(1);
        dec.advance(cum[s], freq[s], total).expect("advance");
        decoded.push(s);
    }
    assert_eq!(decoded, symbols);
}
