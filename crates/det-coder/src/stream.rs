use crate::{Cdf, RangeDecoder, RangeEncoder, RangeError};

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum StreamError {
    LengthMismatch,
    SymbolOutOfRange,
    Cdf,
    NonCanonicalPayload,
    Range(RangeError),
}

pub fn encode_token_stream(cdfs: &[Cdf], tokens: &[usize]) -> Result<Vec<u8>, StreamError> {
    if cdfs.len() != tokens.len() {
        return Err(StreamError::LengthMismatch);
    }
    let mut enc = RangeEncoder::new();
    for (cdf, &token) in cdfs.iter().zip(tokens) {
        cdf.validate().map_err(|_| StreamError::Cdf)?;
        let (&cum, &freq) = cdf
            .cum
            .get(token)
            .zip(cdf.freq.get(token))
            .ok_or(StreamError::SymbolOutOfRange)?;
        enc.encode(cum, freq as u64, cdf.total)
            .map_err(StreamError::Range)?;
    }
    Ok(enc.finish())
}

pub fn decode_token_stream(cdfs: &[Cdf], bytes: &[u8]) -> Result<Vec<usize>, StreamError> {
    let mut dec = RangeDecoder::new(bytes).map_err(StreamError::Range)?;
    let mut out = Vec::with_capacity(cdfs.len());
    for cdf in cdfs {
        cdf.validate().map_err(|_| StreamError::Cdf)?;
        let value = dec.decode_freq(cdf.total).map_err(StreamError::Range)?;
        let token = cdf.symbol_for_validated(value).ok_or(StreamError::Cdf)?;
        let cum = cdf.cum[token];
        let freq = cdf.freq[token] as u64;
        dec.advance(cum, freq, cdf.total)
            .map_err(StreamError::Range)?;
        out.push(token);
    }
    let canonical = encode_token_stream(cdfs, &out)?;
    if canonical != bytes {
        return Err(StreamError::NonCanonicalPayload);
    }
    Ok(out)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::logits_to_cdf;

    #[test]
    fn token_stream_round_trips_with_position_cdfs() {
        let cdfs = vec![
            logits_to_cdf(&[0.0, 1.0, -1.0]).expect("cdf0"),
            logits_to_cdf(&[-2.0, 0.5, 3.0]).expect("cdf1"),
            logits_to_cdf(&[4.0, 0.0, -3.0]).expect("cdf2"),
            logits_to_cdf(&[0.25, 0.5, 0.75]).expect("cdf3"),
        ];
        let tokens = vec![1usize, 2, 0, 1];
        let bytes = encode_token_stream(&cdfs, &tokens).expect("encode");
        let decoded = decode_token_stream(&cdfs, &bytes).expect("decode");
        assert_eq!(decoded, tokens);
    }

    #[test]
    fn token_stream_rejects_bad_symbol() {
        let cdf = logits_to_cdf(&[0.0, 1.0]).expect("cdf");
        assert_eq!(
            encode_token_stream(&[cdf], &[2]),
            Err(StreamError::SymbolOutOfRange)
        );
    }

    #[test]
    fn token_stream_rejects_malformed_cdf_without_panicking() {
        let bad = Cdf {
            freq: Vec::new(),
            cum: Vec::new(),
            total: 1,
        };
        assert_eq!(
            encode_token_stream(core::slice::from_ref(&bad), &[0]),
            Err(StreamError::Cdf)
        );
        let bytes = RangeEncoder::new().finish();
        assert_eq!(
            decode_token_stream(core::slice::from_ref(&bad), &bytes),
            Err(StreamError::Cdf)
        );
    }

    #[test]
    fn token_stream_rejects_corrupted_payload_frequency() {
        let cdf = Cdf {
            freq: vec![1],
            cum: vec![0],
            total: 1,
        };
        assert_eq!(
            decode_token_stream(&[cdf], &[0xff; 8]),
            Err(StreamError::Range(RangeError::InvalidFrequency))
        );
    }

    #[test]
    fn token_stream_rejects_noncanonical_trailing_payload() {
        let cdfs = vec![logits_to_cdf(&[0.0, 1.0]).expect("cdf")];
        let mut bytes = encode_token_stream(&cdfs, &[1]).expect("encode");
        bytes.push(0);
        assert_eq!(
            decode_token_stream(&cdfs, &bytes),
            Err(StreamError::NonCanonicalPayload)
        );
    }
}
