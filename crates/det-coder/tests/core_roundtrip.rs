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

#[test]
fn range_coder_round_trips_large_lcg_stream() {
    const SYMBOLS: usize = 1_000_000;
    const ALPHABET: usize = 257;

    let mut seed = 0x9e37_79b9_7f4a_7c15u64;
    let mut freq = [0u64; ALPHABET];
    for f in &mut freq {
        seed = seed.wrapping_mul(6364136223846793005).wrapping_add(1);
        *f = ((seed >> 56) % 31) + 1;
    }

    let mut cum = [0u64; ALPHABET];
    for i in 1..ALPHABET {
        cum[i] = cum[i - 1] + freq[i - 1];
    }
    let total: u64 = freq.iter().sum();

    let mut symbols = Vec::with_capacity(SYMBOLS);
    let mut enc = RangeEncoder::new();
    for _ in 0..SYMBOLS {
        seed = seed
            .wrapping_mul(2862933555777941757)
            .wrapping_add(3037000493);
        let symbol = (seed as usize) % ALPHABET;
        symbols.push(symbol);
        enc.encode(cum[symbol], freq[symbol], total)
            .expect("encode");
    }
    let bytes = enc.finish();

    let mut dec = RangeDecoder::new(&bytes).expect("decoder");
    for expected in symbols {
        let value = dec.decode_freq(total).expect("decode freq");
        let symbol = cum.partition_point(|&c| c <= value).saturating_sub(1);
        dec.advance(cum[symbol], freq[symbol], total)
            .expect("advance");
        assert_eq!(symbol, expected);
    }
}
