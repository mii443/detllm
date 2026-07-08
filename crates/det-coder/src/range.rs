const TOP: u64 = 1 << 56;
const BOTTOM: u64 = 1 << 48;

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum RangeError {
    InvalidFrequency,
    UnexpectedEof,
}

#[derive(Debug, Clone)]
pub struct RangeEncoder {
    low: u64,
    range: u64,
    out: Vec<u8>,
}

impl Default for RangeEncoder {
    fn default() -> Self {
        Self::new()
    }
}

impl RangeEncoder {
    pub fn new() -> Self {
        Self {
            low: 0,
            range: u64::MAX,
            out: Vec::new(),
        }
    }

    pub fn encode(&mut self, cum: u64, freq: u64, total: u64) -> Result<(), RangeError> {
        validate(cum, freq, total)?;
        self.range /= total;
        self.low = self.low.wrapping_add(self.range.wrapping_mul(cum));
        self.range *= freq;
        self.renorm();
        Ok(())
    }

    pub fn finish(mut self) -> Vec<u8> {
        for _ in 0..8 {
            self.out.push((self.low >> 56) as u8);
            self.low <<= 8;
        }
        self.out
    }

    fn renorm(&mut self) {
        while normalize_step(self.low, &mut self.range) {
            self.out.push((self.low >> 56) as u8);
            self.low <<= 8;
            self.range <<= 8;
        }
    }
}

#[derive(Debug, Clone)]
pub struct RangeDecoder<'a> {
    low: u64,
    code: u64,
    range: u64,
    pending: Option<(u64, u64)>,
    input: &'a [u8],
    pos: usize,
}

impl<'a> RangeDecoder<'a> {
    pub fn new(input: &'a [u8]) -> Result<Self, RangeError> {
        let mut dec = Self {
            low: 0,
            code: 0,
            range: u64::MAX,
            pending: None,
            input,
            pos: 0,
        };
        for _ in 0..8 {
            dec.code = (dec.code << 8) | dec.read_byte()? as u64;
        }
        Ok(dec)
    }

    pub fn decode_freq(&mut self, total: u64) -> Result<u64, RangeError> {
        if self.pending.is_some() {
            return Err(RangeError::InvalidFrequency);
        }
        if total == 0 || total >= (1u64 << 31) {
            return Err(RangeError::InvalidFrequency);
        }
        let range = self.range / total;
        if range == 0 {
            return Err(RangeError::InvalidFrequency);
        }
        let value = self.code.wrapping_sub(self.low) / range;
        if value >= total {
            return Err(RangeError::InvalidFrequency);
        }
        self.pending = Some((range, total));
        Ok(value)
    }

    pub fn advance(&mut self, cum: u64, freq: u64, total: u64) -> Result<(), RangeError> {
        validate(cum, freq, total)?;
        let (range, pending_total) = self.pending.ok_or(RangeError::InvalidFrequency)?;
        if pending_total != total {
            return Err(RangeError::InvalidFrequency);
        }
        self.pending = None;
        self.range = range;
        self.low = self.low.wrapping_add(self.range.wrapping_mul(cum));
        self.range *= freq;
        self.renorm()
    }

    fn renorm(&mut self) -> Result<(), RangeError> {
        while normalize_step(self.low, &mut self.range) {
            self.code = (self.code << 8) | self.read_byte()? as u64;
            self.low <<= 8;
            self.range <<= 8;
        }
        Ok(())
    }

    fn read_byte(&mut self) -> Result<u8, RangeError> {
        if self.pos >= self.input.len() {
            return Err(RangeError::UnexpectedEof);
        }
        let b = self.input[self.pos];
        self.pos += 1;
        Ok(b)
    }
}

fn validate(cum: u64, freq: u64, total: u64) -> Result<(), RangeError> {
    let end = cum.checked_add(freq).ok_or(RangeError::InvalidFrequency)?;
    if freq == 0 || total == 0 || total >= (1u64 << 31) || end > total {
        return Err(RangeError::InvalidFrequency);
    }
    Ok(())
}

fn normalize_step(low: u64, range: &mut u64) -> bool {
    if (low ^ low.wrapping_add(*range)) < TOP {
        true
    } else if *range < BOTTOM {
        *range = low.wrapping_neg() & (BOTTOM - 1);
        true
    } else {
        false
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn range_round_trips_small_stream() {
        let freq = [3u64, 2, 7, 1];
        let cum = [0u64, 3, 5, 12];
        let total = 13;
        let symbols = [2usize, 0, 2, 3, 1, 2, 0, 0, 2, 1, 3, 2];

        let mut enc = RangeEncoder::new();
        for &s in &symbols {
            enc.encode(cum[s], freq[s], total).expect("encode");
        }
        let bytes = enc.finish();

        let mut dec = RangeDecoder::new(&bytes).expect("decoder");
        let mut out = Vec::new();
        for _ in 0..symbols.len() {
            let u = dec.decode_freq(total).expect("freq");
            let s = cum.partition_point(|&c| c <= u).saturating_sub(1);
            dec.advance(cum[s], freq[s], total).expect("advance");
            out.push(s);
        }
        assert_eq!(out, symbols);
    }

    #[test]
    fn decoder_rejects_truncated_payload() {
        assert!(matches!(
            RangeDecoder::new(&[]),
            Err(RangeError::UnexpectedEof)
        ));
        assert!(matches!(
            RangeDecoder::new(&[0, 1, 2, 3, 4, 5, 6]),
            Err(RangeError::UnexpectedEof)
        ));

        let freq = [1u64, 1];
        let cum = [0u64, 1];
        let total = 2;
        let symbols: Vec<usize> = (0..10_000).map(|i| (i * 17 + i / 5) % 2).collect();
        let mut enc = RangeEncoder::new();
        for &symbol in &symbols {
            enc.encode(cum[symbol], freq[symbol], total)
                .expect("encode");
        }
        let mut bytes = enc.finish();
        assert!(bytes.len() > 8);
        bytes.truncate(bytes.len() - 1);

        let mut dec = RangeDecoder::new(&bytes).expect("initial bytes present");
        let mut saw_eof = false;
        for _ in 0..symbols.len() {
            let value = dec.decode_freq(total).expect("decode freq");
            let token = cum.partition_point(|&c| c <= value).saturating_sub(1);
            if dec.advance(cum[token], freq[token], total) == Err(RangeError::UnexpectedEof) {
                saw_eof = true;
                break;
            }
        }
        assert!(saw_eof);
    }

    #[test]
    fn rejects_invalid_frequency_ranges_without_overflow() {
        let mut enc = RangeEncoder::new();
        assert_eq!(
            enc.encode(u64::MAX, 1, 8),
            Err(RangeError::InvalidFrequency)
        );
        assert_eq!(enc.encode(0, 0, 8), Err(RangeError::InvalidFrequency));
        assert_eq!(enc.encode(8, 1, 8), Err(RangeError::InvalidFrequency));
        assert_eq!(
            enc.encode(0, 1, 1u64 << 31),
            Err(RangeError::InvalidFrequency)
        );

        let bytes = RangeEncoder::new().finish();
        let mut dec = RangeDecoder::new(&bytes).expect("decoder");
        assert_eq!(
            dec.decode_freq(1u64 << 31),
            Err(RangeError::InvalidFrequency)
        );
        let mut dec = RangeDecoder::new(&[0xff; 8]).expect("decoder");
        assert_eq!(dec.decode_freq(1), Err(RangeError::InvalidFrequency));
        assert_eq!(
            dec.advance(u64::MAX, 1, 8),
            Err(RangeError::InvalidFrequency)
        );
    }

    #[test]
    fn decoder_enforces_decode_advance_pairs() {
        let freq = [1u64, 1];
        let cum = [0u64, 1];
        let total = 2;
        let mut enc = RangeEncoder::new();
        enc.encode(cum[1], freq[1], total).expect("encode");
        enc.encode(cum[0], freq[0], total).expect("encode");
        let bytes = enc.finish();

        let mut dec = RangeDecoder::new(&bytes).expect("decoder");
        assert_eq!(
            dec.advance(cum[0], freq[0], total),
            Err(RangeError::InvalidFrequency)
        );

        let first = dec.decode_freq(total).expect("decode first");
        assert_eq!(dec.decode_freq(total), Err(RangeError::InvalidFrequency));
        assert_eq!(
            dec.advance(cum[first as usize], freq[first as usize], 3),
            Err(RangeError::InvalidFrequency)
        );
        dec.advance(cum[first as usize], freq[first as usize], total)
            .expect("advance first");

        let second = dec.decode_freq(total).expect("decode second");
        dec.advance(cum[second as usize], freq[second as usize], total)
            .expect("advance second");
        assert_eq!([first as usize, second as usize], [1, 0]);
    }

    #[test]
    fn range_round_trips_million_symbol_deterministic_random_stream() {
        const SYMBOLS: usize = 257;
        const LEN: usize = 1_000_000;

        let mut rng = Lcg::new(0xdecaf_bad5eed);
        let mut freq = [0u64; SYMBOLS];
        for f in &mut freq {
            *f = 1 + (rng.next() % 997);
        }
        let mut cum = [0u64; SYMBOLS];
        for i in 1..SYMBOLS {
            cum[i] = cum[i - 1] + freq[i - 1];
        }
        let total = cum[SYMBOLS - 1] + freq[SYMBOLS - 1];
        assert!(total < (1u64 << 31));

        let symbols: Vec<usize> = (0..LEN).map(|_| (rng.next() as usize) % SYMBOLS).collect();
        let mut enc = RangeEncoder::new();
        for &s in &symbols {
            enc.encode(cum[s], freq[s], total).expect("encode");
        }
        let bytes = enc.finish();

        let mut dec = RangeDecoder::new(&bytes).expect("decoder");
        for &expected in &symbols {
            let u = dec.decode_freq(total).expect("decode freq");
            let actual = cum.partition_point(|&c| c <= u).saturating_sub(1);
            assert_eq!(actual, expected);
            dec.advance(cum[actual], freq[actual], total)
                .expect("advance");
        }
    }

    struct Lcg(u64);

    impl Lcg {
        fn new(seed: u64) -> Self {
            Self(seed)
        }

        fn next(&mut self) -> u64 {
            self.0 = self
                .0
                .wrapping_mul(6_364_136_223_846_793_005)
                .wrapping_add(1_442_695_040_888_963_407);
            self.0 >> 32
        }
    }
}
