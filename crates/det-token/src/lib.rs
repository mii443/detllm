use std::collections::BTreeMap;

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum TokenError {
    MissingByteFallback(u8),
    InvalidToken(u32),
    MissingVocabulary,
    MissingTokenBytes(Vec<u8>),
    MissingMergeToken(Vec<u8>),
    IncompleteByteFallback,
    MetadataType,
    TokenTypeLength,
    InvalidTokenType(i64),
    ScoresLength,
    NonFiniteScore,
    TokenIndexOverflow,
    DuplicateByteFallback(u8),
    DuplicateTokenId(u32),
    DuplicateTokenBytes(Vec<u8>),
    InvalidMerge(String),
    DuplicateMerge(String),
    UnsupportedTokenizerModel(String),
}

#[derive(Debug, Clone)]
pub struct ByteFallbackTokenizer {
    byte_to_token: [Option<u32>; 256],
    token_to_byte: BTreeMap<u32, u8>,
}

#[derive(Debug, Clone)]
pub enum Tokenizer {
    ByteFallback(ByteFallbackTokenizer),
    ByteBpe(ByteBpeTokenizer),
    SentencePiece(SentencePieceTokenizer),
}

#[derive(Debug, Clone)]
pub struct ByteBpeTokenizer {
    byte_to_token: [Option<u32>; 256],
    token_to_bytes: BTreeMap<u32, Vec<u8>>,
    merge_ranks: BTreeMap<(u32, u32), (usize, u32)>,
}

#[derive(Debug, Clone)]
pub struct SentencePieceTokenizer {
    byte_fallback: ByteFallbackTokenizer,
    token_to_bytes: BTreeMap<u32, Vec<u8>>,
    pieces: BTreeMap<Vec<u8>, (u32, f32)>,
    max_piece_len: usize,
}

impl Tokenizer {
    pub fn from_gguf(gguf: &det_gguf::Gguf) -> Result<Self, TokenError> {
        let tokens = gguf_tokens(gguf)?;
        let emit_mask = optional_gguf_emit_mask(gguf, tokens.len())?;
        let merges = optional_gguf_merges(gguf)?;
        let model = optional_gguf_model(gguf)?;
        let empty_merges: &[String] = &[];
        match model {
            Some("gpt2") => {
                let merges = merges.unwrap_or(empty_merges);
                ByteBpeTokenizer::from_tokens_merges_and_emit_mask(
                    tokens,
                    merges,
                    emit_mask.as_deref(),
                )
                .map(Self::ByteBpe)
            }
            Some("llama" | "spm" | "sentencepiece") => {
                SentencePieceTokenizer::from_tokens_scores_and_emit_mask(
                    tokens,
                    optional_gguf_scores(gguf, tokens.len())?,
                    emit_mask.as_deref(),
                )
                .map(Self::SentencePiece)
            }
            Some(other) => Err(TokenError::UnsupportedTokenizerModel(other.to_owned())),
            None => match merges {
                Some(merges) if !merges.is_empty() => {
                    ByteBpeTokenizer::from_tokens_merges_and_emit_mask(
                        tokens,
                        merges,
                        emit_mask.as_deref(),
                    )
                    .map(Self::ByteBpe)
                }
                _ if has_gguf_scores(gguf)? => {
                    SentencePieceTokenizer::from_tokens_scores_and_emit_mask(
                        tokens,
                        optional_gguf_scores(gguf, tokens.len())?,
                        emit_mask.as_deref(),
                    )
                    .map(Self::SentencePiece)
                }
                _ => ByteFallbackTokenizer::from_tokens(tokens).map(Self::ByteFallback),
            },
        }
    }

    pub fn tokenize_bytes(&self, input: &[u8]) -> Result<Vec<u32>, TokenError> {
        match self {
            Self::ByteFallback(t) => t.tokenize_bytes(input),
            Self::ByteBpe(t) => t.tokenize_bytes(input),
            Self::SentencePiece(t) => t.tokenize_bytes(input),
        }
    }

    pub fn detokenize_bytes(&self, tokens: &[u32]) -> Result<Vec<u8>, TokenError> {
        match self {
            Self::ByteFallback(t) => t.detokenize_bytes(tokens),
            Self::ByteBpe(t) => t.detokenize_bytes(tokens),
            Self::SentencePiece(t) => t.detokenize_bytes(tokens),
        }
    }
}

impl ByteFallbackTokenizer {
    pub fn new(pairs: &[(u8, u32)]) -> Result<Self, TokenError> {
        let mut byte_to_token = [None; 256];
        let mut token_to_byte = BTreeMap::new();
        for &(b, t) in pairs {
            if byte_to_token[b as usize].is_some() {
                return Err(TokenError::DuplicateByteFallback(b));
            }
            if token_to_byte.contains_key(&t) {
                return Err(TokenError::DuplicateTokenId(t));
            }
            byte_to_token[b as usize] = Some(t);
            token_to_byte.insert(t, b);
        }
        Ok(Self {
            byte_to_token,
            token_to_byte,
        })
    }

    pub fn from_gguf(gguf: &det_gguf::Gguf) -> Result<Self, TokenError> {
        Self::from_tokens(gguf_tokens(gguf)?)
    }

    pub fn from_tokens(tokens: &[String]) -> Result<Self, TokenError> {
        let mut pairs = Vec::new();
        for (idx, token) in tokens.iter().enumerate() {
            if let Some(byte) = canonical_byte_fallback_token(token) {
                let id = u32::try_from(idx).map_err(|_| TokenError::TokenIndexOverflow)?;
                if pairs.iter().any(|&(existing, _)| existing == byte) {
                    return Err(TokenError::DuplicateByteFallback(byte));
                }
                pairs.push((byte, id));
            }
        }
        let out = Self::new(&pairs)?;
        if out.byte_to_token.iter().all(Option::is_some) {
            Ok(out)
        } else {
            Err(TokenError::IncompleteByteFallback)
        }
    }

    pub fn tokenize_bytes(&self, input: &[u8]) -> Result<Vec<u32>, TokenError> {
        let mut out = Vec::with_capacity(input.len());
        for &b in input {
            out.push(self.byte_to_token[b as usize].ok_or(TokenError::MissingByteFallback(b))?);
        }
        Ok(out)
    }

    pub fn detokenize_bytes(&self, tokens: &[u32]) -> Result<Vec<u8>, TokenError> {
        let mut out = Vec::with_capacity(tokens.len());
        for &t in tokens {
            out.push(
                *self
                    .token_to_byte
                    .get(&t)
                    .ok_or(TokenError::InvalidToken(t))?,
            );
        }
        Ok(out)
    }
}

impl ByteBpeTokenizer {
    pub fn from_gguf(gguf: &det_gguf::Gguf) -> Result<Self, TokenError> {
        let tokens = gguf_tokens(gguf)?;
        let merges = optional_gguf_merges(gguf)?.ok_or(TokenError::MissingVocabulary)?;
        let emit_mask = optional_gguf_emit_mask(gguf, tokens.len())?;
        Self::from_tokens_merges_and_emit_mask(tokens, merges, emit_mask.as_deref())
    }

    pub fn from_tokens_and_merges(
        tokens: &[String],
        merges: &[String],
    ) -> Result<Self, TokenError> {
        Self::from_tokens_merges_and_emit_mask(tokens, merges, None)
    }

    fn from_tokens_merges_and_emit_mask(
        tokens: &[String],
        merges: &[String],
        emit_mask: Option<&[bool]>,
    ) -> Result<Self, TokenError> {
        let mut byte_to_token = [None; 256];
        let mut token_to_bytes = BTreeMap::new();
        let mut bytes_to_token = BTreeMap::new();

        for (idx, token) in tokens.iter().enumerate() {
            let id = u32::try_from(idx).map_err(|_| TokenError::TokenIndexOverflow)?;
            let bytes = token_piece_bytes(token);
            if token_is_emittable(emit_mask, idx) {
                if let [byte] = bytes.as_slice() {
                    if byte_to_token[*byte as usize].is_some() {
                        return Err(TokenError::DuplicateByteFallback(*byte));
                    }
                    byte_to_token[*byte as usize] = Some(id);
                }
                if bytes_to_token.insert(bytes.clone(), id).is_some() {
                    return Err(TokenError::DuplicateTokenBytes(bytes));
                }
            }
            token_to_bytes.insert(id, bytes);
        }

        if byte_to_token.iter().any(Option::is_none) {
            return Err(TokenError::IncompleteByteFallback);
        }

        let mut merge_ranks = BTreeMap::new();
        for (rank, merge) in merges.iter().enumerate() {
            let (left, right) = parse_merge(merge)?;
            let left_bytes = token_piece_bytes(left);
            let right_bytes = token_piece_bytes(right);
            let Some(&left_id) = bytes_to_token.get(&left_bytes) else {
                if emit_mask.is_some() {
                    continue;
                }
                return Err(TokenError::MissingTokenBytes(left_bytes.clone()));
            };
            let Some(&right_id) = bytes_to_token.get(&right_bytes) else {
                if emit_mask.is_some() {
                    continue;
                }
                return Err(TokenError::MissingTokenBytes(right_bytes.clone()));
            };
            let mut merged = left_bytes;
            merged.extend_from_slice(&right_bytes);
            let Some(&merged_id) = bytes_to_token.get(&merged) else {
                if emit_mask.is_some() {
                    continue;
                }
                return Err(TokenError::MissingMergeToken(merged.clone()));
            };
            if merge_ranks
                .insert((left_id, right_id), (rank, merged_id))
                .is_some()
            {
                return Err(TokenError::DuplicateMerge(merge.clone()));
            }
        }

        Ok(Self {
            byte_to_token,
            token_to_bytes,
            merge_ranks,
        })
    }

    pub fn tokenize_bytes(&self, input: &[u8]) -> Result<Vec<u32>, TokenError> {
        let mut ids = Vec::with_capacity(input.len());
        for &byte in input {
            ids.push(
                self.byte_to_token[byte as usize].ok_or(TokenError::MissingByteFallback(byte))?,
            );
        }

        loop {
            let mut best: Option<(usize, usize, u32)> = None;
            for i in 0..ids.len().saturating_sub(1) {
                if let Some(&(rank, merged_id)) = self.merge_ranks.get(&(ids[i], ids[i + 1])) {
                    match best {
                        Some((best_rank, _, _)) if best_rank <= rank => {}
                        _ => best = Some((rank, i, merged_id)),
                    }
                }
            }
            let Some((_, index, merged_id)) = best else {
                break;
            };
            ids[index] = merged_id;
            ids.remove(index + 1);
        }

        Ok(ids)
    }

    pub fn detokenize_bytes(&self, tokens: &[u32]) -> Result<Vec<u8>, TokenError> {
        let mut out = Vec::new();
        for &token in tokens {
            let bytes = self
                .token_to_bytes
                .get(&token)
                .ok_or(TokenError::InvalidToken(token))?;
            out.extend_from_slice(bytes);
        }
        Ok(out)
    }
}

impl SentencePieceTokenizer {
    pub fn from_gguf(gguf: &det_gguf::Gguf) -> Result<Self, TokenError> {
        let tokens = gguf_tokens(gguf)?;
        let scores = optional_gguf_scores(gguf, tokens.len())?;
        let emit_mask = optional_gguf_emit_mask(gguf, tokens.len())?;
        Self::from_tokens_scores_and_emit_mask(tokens, scores, emit_mask.as_deref())
    }

    pub fn from_tokens_and_scores(
        tokens: &[String],
        scores: Option<&[f32]>,
    ) -> Result<Self, TokenError> {
        Self::from_tokens_scores_and_emit_mask(tokens, scores, None)
    }

    fn from_tokens_scores_and_emit_mask(
        tokens: &[String],
        scores: Option<&[f32]>,
        emit_mask: Option<&[bool]>,
    ) -> Result<Self, TokenError> {
        if let Some(scores) = scores {
            if scores.len() != tokens.len() {
                return Err(TokenError::ScoresLength);
            }
            if scores.iter().any(|score| !score.is_finite()) {
                return Err(TokenError::NonFiniteScore);
            }
        }
        let byte_fallback = ByteFallbackTokenizer::from_tokens(tokens)?;
        let mut token_to_bytes = BTreeMap::new();
        let mut pieces: BTreeMap<Vec<u8>, (u32, f32)> = BTreeMap::new();
        let mut max_piece_len = 1usize;

        for (idx, token) in tokens.iter().enumerate() {
            let id = u32::try_from(idx).map_err(|_| TokenError::TokenIndexOverflow)?;
            let bytes = token_piece_bytes(token);
            let score = scores.map(|s| s[idx]).unwrap_or(0.0);
            if bytes.len() > 1 && token_is_emittable(emit_mask, idx) {
                max_piece_len = max_piece_len.max(bytes.len());
                match pieces.get(&bytes).copied() {
                    Some((best_id, best_score)) => {
                        if score > best_score
                            || (score.to_bits() == best_score.to_bits() && id < best_id)
                        {
                            pieces.insert(bytes.clone(), (id, score));
                        }
                    }
                    None => {
                        pieces.insert(bytes.clone(), (id, score));
                    }
                }
            }
            token_to_bytes.insert(id, bytes);
        }

        Ok(Self {
            byte_fallback,
            token_to_bytes,
            pieces,
            max_piece_len,
        })
    }

    pub fn tokenize_bytes(&self, input: &[u8]) -> Result<Vec<u32>, TokenError> {
        let mut out = Vec::new();
        let mut pos = 0usize;
        while pos < input.len() {
            let max_end = input.len().min(pos + self.max_piece_len);
            let mut best: Option<(usize, u32, f32)> = None;
            for end in pos + 1..=max_end {
                let piece = &input[pos..end];
                if let Some(&(id, score)) = self.pieces.get(piece) {
                    match best {
                        Some((best_len, best_id, best_score)) => {
                            let len = end - pos;
                            if len > best_len
                                || (len == best_len
                                    && (score > best_score
                                        || (score.to_bits() == best_score.to_bits()
                                            && id < best_id)))
                            {
                                best = Some((len, id, score));
                            }
                        }
                        None => best = Some((end - pos, id, score)),
                    }
                }
            }
            if let Some((len, id, _)) = best {
                out.push(id);
                pos += len;
            } else {
                let token = self.byte_fallback.byte_to_token[input[pos] as usize]
                    .ok_or(TokenError::MissingByteFallback(input[pos]))?;
                out.push(token);
                pos += 1;
            }
        }
        Ok(out)
    }

    pub fn detokenize_bytes(&self, tokens: &[u32]) -> Result<Vec<u8>, TokenError> {
        let mut out = Vec::new();
        for &token in tokens {
            let bytes = self
                .token_to_bytes
                .get(&token)
                .ok_or(TokenError::InvalidToken(token))?;
            out.extend_from_slice(bytes);
        }
        Ok(out)
    }
}

fn gguf_tokens(gguf: &det_gguf::Gguf) -> Result<&[String], TokenError> {
    match gguf.metadata_value("tokenizer.ggml.tokens") {
        Ok(det_gguf::MetadataValue::ArrayString(tokens)) => Ok(tokens),
        Ok(_) => Err(TokenError::MetadataType),
        Err(det_gguf::GgufError::MetadataNotFound) => Err(TokenError::MissingVocabulary),
        Err(_) => Err(TokenError::MetadataType),
    }
}

fn optional_gguf_merges(gguf: &det_gguf::Gguf) -> Result<Option<&[String]>, TokenError> {
    match gguf.metadata_value("tokenizer.ggml.merges") {
        Ok(det_gguf::MetadataValue::ArrayString(merges)) => Ok(Some(merges)),
        Ok(_) => Err(TokenError::MetadataType),
        Err(det_gguf::GgufError::MetadataNotFound) => Ok(None),
        Err(_) => Err(TokenError::MetadataType),
    }
}

fn optional_gguf_model(gguf: &det_gguf::Gguf) -> Result<Option<&str>, TokenError> {
    match gguf.metadata_value("tokenizer.ggml.model") {
        Ok(det_gguf::MetadataValue::String(model)) => Ok(Some(model)),
        Ok(_) => Err(TokenError::MetadataType),
        Err(det_gguf::GgufError::MetadataNotFound) => Ok(None),
        Err(_) => Err(TokenError::MetadataType),
    }
}

fn optional_gguf_scores(
    gguf: &det_gguf::Gguf,
    vocab_len: usize,
) -> Result<Option<&[f32]>, TokenError> {
    match gguf.metadata_value("tokenizer.ggml.scores") {
        Ok(det_gguf::MetadataValue::ArrayF32(scores)) => {
            if scores.len() != vocab_len {
                return Err(TokenError::ScoresLength);
            }
            if scores.iter().any(|score| !score.is_finite()) {
                return Err(TokenError::NonFiniteScore);
            }
            Ok(Some(scores))
        }
        Ok(_) => Err(TokenError::MetadataType),
        Err(det_gguf::GgufError::MetadataNotFound) => Ok(None),
        Err(_) => Err(TokenError::MetadataType),
    }
}

fn optional_gguf_emit_mask(
    gguf: &det_gguf::Gguf,
    vocab_len: usize,
) -> Result<Option<Vec<bool>>, TokenError> {
    match gguf.metadata_value("tokenizer.ggml.token_type") {
        Ok(det_gguf::MetadataValue::ArrayI32(types)) => {
            if types.len() != vocab_len {
                return Err(TokenError::TokenTypeLength);
            }
            types
                .iter()
                .map(|&token_type| token_type_is_emittable(i64::from(token_type)))
                .collect::<Result<Vec<_>, _>>()
                .map(Some)
        }
        Ok(det_gguf::MetadataValue::ArrayU32(types)) => {
            if types.len() != vocab_len {
                return Err(TokenError::TokenTypeLength);
            }
            types
                .iter()
                .map(|&token_type| token_type_is_emittable(i64::from(token_type)))
                .collect::<Result<Vec<_>, _>>()
                .map(Some)
        }
        Ok(_) => Err(TokenError::MetadataType),
        Err(det_gguf::GgufError::MetadataNotFound) => Ok(None),
        Err(_) => Err(TokenError::MetadataType),
    }
}

fn token_is_emittable(emit_mask: Option<&[bool]>, idx: usize) -> bool {
    emit_mask
        .and_then(|mask| mask.get(idx))
        .copied()
        .unwrap_or(true)
}

fn token_type_is_emittable(token_type: i64) -> Result<bool, TokenError> {
    match token_type {
        1 | 4 | 6 => Ok(true),
        2 | 3 | 5 => Ok(false),
        other => Err(TokenError::InvalidTokenType(other)),
    }
}

fn parse_merge(merge: &str) -> Result<(&str, &str), TokenError> {
    let mut parts = merge.split(' ');
    let left = parts
        .next()
        .filter(|part| !part.is_empty())
        .ok_or_else(|| TokenError::InvalidMerge(merge.to_owned()))?;
    let right = parts
        .next()
        .filter(|part| !part.is_empty())
        .ok_or_else(|| TokenError::InvalidMerge(merge.to_owned()))?;
    if parts.next().is_some() {
        return Err(TokenError::InvalidMerge(merge.to_owned()));
    }
    Ok((left, right))
}

fn has_gguf_scores(gguf: &det_gguf::Gguf) -> Result<bool, TokenError> {
    match gguf.metadata_value("tokenizer.ggml.scores") {
        Ok(det_gguf::MetadataValue::ArrayF32(_)) => Ok(true),
        Ok(_) => Err(TokenError::MetadataType),
        Err(det_gguf::GgufError::MetadataNotFound) => Ok(false),
        Err(_) => Err(TokenError::MetadataType),
    }
}

fn token_piece_bytes(token: &str) -> Vec<u8> {
    if let Some(byte) = canonical_byte_fallback_token(token) {
        return vec![byte];
    }
    if let Some(bytes) = gpt2_byte_unicode_decode(token) {
        return bytes;
    }
    if let Some(rest) = token.strip_prefix('▁') {
        let mut bytes = Vec::with_capacity(1 + rest.len());
        bytes.push(b' ');
        bytes.extend_from_slice(rest.as_bytes());
        return bytes;
    }
    token.as_bytes().to_vec()
}

fn gpt2_byte_unicode_decode(token: &str) -> Option<Vec<u8>> {
    let mut out = Vec::with_capacity(token.len());
    for ch in token.chars() {
        out.push(gpt2_char_to_byte(ch)?);
    }
    Some(out)
}

fn gpt2_char_to_byte(ch: char) -> Option<u8> {
    let code = ch as u32;
    if (33..=126).contains(&code) || (161..=172).contains(&code) || (174..=255).contains(&code) {
        return Some(code as u8);
    }

    let mut n = 0u32;
    for byte in 0u32..=255 {
        if (33..=126).contains(&byte) || (161..=172).contains(&byte) || (174..=255).contains(&byte)
        {
            continue;
        }
        if code == 256 + n {
            return Some(byte as u8);
        }
        n += 1;
    }
    None
}

pub fn canonical_byte_fallback_token(token: &str) -> Option<u8> {
    let hex = token.strip_prefix("<0x")?.strip_suffix('>')?;
    if hex.len() != 2 {
        return None;
    }
    u8::from_str_radix(hex, 16).ok()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn byte_fallback_round_trips() {
        let pairs = [(0u8, 10u32), (65, 11), (255, 12)];
        let tok = ByteFallbackTokenizer::new(&pairs).expect("tokenizer");
        let tokens = tok.tokenize_bytes(&[65, 0, 255]).expect("tokenize");
        assert_eq!(tokens, [11, 10, 12]);
        assert_eq!(tok.detokenize_bytes(&tokens).expect("detok"), [65, 0, 255]);
        assert_eq!(canonical_byte_fallback_token("<0xAF>"), Some(0xaf));
    }

    #[test]
    fn byte_fallback_constructor_rejects_ambiguous_pairs() {
        assert!(matches!(
            ByteFallbackTokenizer::new(&[(0, 10), (0, 11)]),
            Err(TokenError::DuplicateByteFallback(0))
        ));
        assert!(matches!(
            ByteFallbackTokenizer::new(&[(0, 10), (1, 10)]),
            Err(TokenError::DuplicateTokenId(10))
        ));
    }

    #[test]
    fn byte_fallback_round_trips_all_byte_values() {
        let tokens = canonical_byte_tokens();
        let tok = ByteFallbackTokenizer::from_tokens(&tokens).expect("tokenizer");
        assert_tokenizer_round_trips_all_bytes(&Tokenizer::ByteFallback(tok));
    }

    #[test]
    fn builds_from_gguf_tokens_metadata() {
        let tokens: Vec<String> = (0..=255).map(|b| format!("<0x{b:02X}>")).collect();
        let mut metadata = BTreeMap::new();
        metadata.insert(
            "tokenizer.ggml.tokens".to_owned(),
            det_gguf::MetadataValue::ArrayString(tokens),
        );
        let gguf = det_gguf::Gguf::from_parts(3, metadata, Vec::new(), 0, 0);
        let tok = ByteFallbackTokenizer::from_gguf(&gguf).expect("tokenizer");
        assert_eq!(
            tok.tokenize_bytes(&[0, 65, 255]).expect("tokenize"),
            [0, 65, 255]
        );
    }

    #[test]
    fn rejects_incomplete_byte_fallback() {
        let tokens = vec!["<0x00>".to_owned(), "<0x01>".to_owned()];
        assert!(matches!(
            ByteFallbackTokenizer::from_tokens(&tokens),
            Err(TokenError::IncompleteByteFallback)
        ));
    }

    #[test]
    fn rejects_duplicate_byte_fallback_tokens() {
        let mut tokens = canonical_byte_tokens();
        tokens.push("<0x00>".to_owned());
        assert!(matches!(
            ByteFallbackTokenizer::from_tokens(&tokens),
            Err(TokenError::DuplicateByteFallback(0))
        ));
    }

    #[test]
    fn byte_bpe_merges_by_rank_then_leftmost_position() {
        let mut tokens: Vec<String> = (0..=255).map(|b| format!("<0x{b:02X}>")).collect();
        let ab = tokens.len() as u32;
        tokens.push("ab".to_owned());
        let _ba = tokens.len() as u32;
        tokens.push("ba".to_owned());
        let aba = tokens.len() as u32;
        tokens.push("aba".to_owned());

        let merges = vec![
            "<0x61> <0x62>".to_owned(),
            "<0x62> <0x61>".to_owned(),
            "ab <0x61>".to_owned(),
        ];
        let tok = ByteBpeTokenizer::from_tokens_and_merges(&tokens, &merges).expect("bpe");

        assert_eq!(tok.tokenize_bytes(b"ab").expect("ab"), [ab]);
        assert_eq!(tok.tokenize_bytes(b"aba").expect("aba"), [aba]);
        assert_eq!(
            tok.tokenize_bytes(b"baba").expect("baba"),
            [b'b' as u32, aba]
        );
        assert_eq!(tok.detokenize_bytes(&[aba]).expect("detok"), b"aba");
    }

    #[test]
    fn byte_bpe_rejects_duplicate_or_malformed_merges() {
        let mut tokens: Vec<String> = (0..=255).map(|b| format!("<0x{b:02X}>")).collect();
        tokens.push("ab".to_owned());

        let duplicate = vec!["<0x61> <0x62>".to_owned(), "<0x61> <0x62>".to_owned()];
        assert!(matches!(
            ByteBpeTokenizer::from_tokens_and_merges(&tokens, &duplicate),
            Err(TokenError::DuplicateMerge(_))
        ));

        for merge in ["<0x61>", "<0x61>  <0x62>", "<0x61> <0x62> extra"] {
            assert!(matches!(
                ByteBpeTokenizer::from_tokens_and_merges(&tokens, &[merge.to_owned()]),
                Err(TokenError::InvalidMerge(_))
            ));
        }
    }

    #[test]
    fn byte_bpe_rejects_duplicate_single_byte_tokens() {
        let mut tokens = canonical_byte_tokens();
        tokens.push("A".to_owned());
        assert!(matches!(
            ByteBpeTokenizer::from_tokens_and_merges(&tokens, &[]),
            Err(TokenError::DuplicateByteFallback(b'A'))
        ));
    }

    #[test]
    fn byte_bpe_rejects_duplicate_multi_byte_tokens() {
        let mut tokens = canonical_byte_tokens();
        tokens.push("ab".to_owned());
        tokens.push("ab".to_owned());
        assert!(matches!(
            ByteBpeTokenizer::from_tokens_and_merges(&tokens, &[]),
            Err(TokenError::DuplicateTokenBytes(bytes)) if bytes == b"ab"
        ));
    }

    #[test]
    fn tokenizer_from_gguf_prefers_bpe_when_merges_exist() {
        let mut tokens: Vec<String> = (0..=255).map(|b| format!("<0x{b:02X}>")).collect();
        let ab = tokens.len() as u32;
        tokens.push("ab".to_owned());
        let mut metadata = BTreeMap::new();
        metadata.insert(
            "tokenizer.ggml.tokens".to_owned(),
            det_gguf::MetadataValue::ArrayString(tokens),
        );
        metadata.insert(
            "tokenizer.ggml.merges".to_owned(),
            det_gguf::MetadataValue::ArrayString(vec!["<0x61> <0x62>".to_owned()]),
        );
        let gguf = det_gguf::Gguf::from_parts(3, metadata, Vec::new(), 0, 0);
        let tok = Tokenizer::from_gguf(&gguf).expect("tokenizer");
        assert_eq!(tok.tokenize_bytes(b"ab").expect("tokens"), [ab]);
    }

    #[test]
    fn tokenizer_from_gguf_uses_gpt2_model_without_merges() {
        let tokens: Vec<String> = (0..=255)
            .map(|b| String::from_utf8(gpt2_byte_unicode_token_bytes(b)).expect("utf8"))
            .collect();
        let mut metadata = BTreeMap::new();
        metadata.insert(
            "tokenizer.ggml.tokens".to_owned(),
            det_gguf::MetadataValue::ArrayString(tokens),
        );
        metadata.insert(
            "tokenizer.ggml.model".to_owned(),
            det_gguf::MetadataValue::String("gpt2".to_owned()),
        );
        let gguf = det_gguf::Gguf::from_parts(3, metadata, Vec::new(), 0, 0);
        let tok = Tokenizer::from_gguf(&gguf).expect("tokenizer");

        assert!(matches!(tok, Tokenizer::ByteBpe(_)));
        let input = b"\x00A \xff\n";
        let ids = tok.tokenize_bytes(input).expect("tokenize");
        assert_eq!(tok.detokenize_bytes(&ids).expect("detok"), input);
    }

    #[test]
    fn tokenizer_from_gguf_rejects_unknown_model_metadata() {
        let tokens = canonical_byte_tokens();
        let mut metadata = BTreeMap::new();
        metadata.insert(
            "tokenizer.ggml.tokens".to_owned(),
            det_gguf::MetadataValue::ArrayString(tokens),
        );
        metadata.insert(
            "tokenizer.ggml.model".to_owned(),
            det_gguf::MetadataValue::String("unsupported".to_owned()),
        );
        let gguf = det_gguf::Gguf::from_parts(3, metadata, Vec::new(), 0, 0);

        assert!(matches!(
            Tokenizer::from_gguf(&gguf),
            Err(TokenError::UnsupportedTokenizerModel(model)) if model == "unsupported"
        ));
    }

    #[test]
    fn byte_bpe_does_not_emit_control_tokens_from_token_type() {
        let mut tokens = canonical_byte_tokens();
        let control_ab = tokens.len() as u32;
        tokens.push("ab".to_owned());
        let mut token_types = vec![6; tokens.len()];
        token_types[control_ab as usize] = 3;

        let mut metadata = BTreeMap::new();
        metadata.insert(
            "tokenizer.ggml.tokens".to_owned(),
            det_gguf::MetadataValue::ArrayString(tokens),
        );
        metadata.insert(
            "tokenizer.ggml.merges".to_owned(),
            det_gguf::MetadataValue::ArrayString(vec!["<0x61> <0x62>".to_owned()]),
        );
        metadata.insert(
            "tokenizer.ggml.token_type".to_owned(),
            det_gguf::MetadataValue::ArrayI32(token_types),
        );
        let gguf = det_gguf::Gguf::from_parts(3, metadata, Vec::new(), 0, 0);
        let tok = Tokenizer::from_gguf(&gguf).expect("tokenizer");

        assert_eq!(tok.tokenize_bytes(b"ab").expect("tokens"), [97, 98]);
        assert_eq!(tok.detokenize_bytes(&[control_ab]).expect("detok"), b"ab");
    }

    #[test]
    fn byte_bpe_decodes_gpt2_byte_unicode_tokens() {
        let mut tokens: Vec<String> = (0..=255)
            .map(|b| String::from_utf8(gpt2_byte_unicode_token_bytes(b)).expect("utf8"))
            .collect();
        tokens.push("he".to_owned());
        tokens.push("hel".to_owned());
        tokens.push("hell".to_owned());
        tokens.push("hello".to_owned());
        let hello_space = tokens.len() as u32;
        tokens.push("helloĠ".to_owned());
        let merges = vec![
            "h e".to_owned(),
            "he l".to_owned(),
            "hel l".to_owned(),
            "hell o".to_owned(),
            "hello Ġ".to_owned(),
        ];
        let tok = ByteBpeTokenizer::from_tokens_and_merges(&tokens, &merges).expect("bpe");
        assert_eq!(
            tok.tokenize_bytes(b"hello ").expect("tokens"),
            [hello_space]
        );
        assert_eq!(
            tok.detokenize_bytes(&[hello_space]).expect("detok"),
            b"hello "
        );
        assert_eq!(tok.tokenize_bytes(b"\n").expect("newline"), [10]);
    }

    #[test]
    fn byte_bpe_round_trips_all_byte_values() {
        let mut tokens: Vec<String> = (0..=255)
            .map(|b| String::from_utf8(gpt2_byte_unicode_token_bytes(b)).expect("utf8"))
            .collect();
        tokens.push("he".to_owned());
        tokens.push("hel".to_owned());
        tokens.push("hell".to_owned());
        tokens.push("hello".to_owned());
        tokens.push("helloĠ".to_owned());
        let merges = vec![
            "h e".to_owned(),
            "he l".to_owned(),
            "hel l".to_owned(),
            "hell o".to_owned(),
            "hello Ġ".to_owned(),
        ];
        let tok = ByteBpeTokenizer::from_tokens_and_merges(&tokens, &merges).expect("bpe");
        assert_tokenizer_round_trips_all_bytes(&Tokenizer::ByteBpe(tok.clone()));

        let input = b"\x00hello \xff\nhello ";
        let ids = tok.tokenize_bytes(input).expect("tokenize");
        assert_eq!(tok.detokenize_bytes(&ids).expect("detok"), input);
    }

    #[test]
    fn sentencepiece_uses_longest_piece_and_space_marker() {
        let mut tokens = canonical_byte_tokens();
        let hello = tokens.len() as u32;
        tokens.push("▁hello".to_owned());
        let mut scores = vec![0.0; tokens.len()];
        scores[hello as usize] = 10.0;

        let tok =
            SentencePieceTokenizer::from_tokens_and_scores(&tokens, Some(&scores)).expect("spm");
        assert_eq!(
            tok.tokenize_bytes(b" hello!").expect("tokens"),
            [hello, b'!' as u32]
        );
        assert_eq!(tok.detokenize_bytes(&[hello]).expect("detok"), b" hello");
    }

    #[test]
    fn sentencepiece_round_trips_all_byte_values() {
        let mut tokens = canonical_byte_tokens();
        let hello = tokens.len() as u32;
        tokens.push("▁hello".to_owned());
        let mut scores = vec![0.0; tokens.len()];
        scores[hello as usize] = 10.0;
        let tok =
            SentencePieceTokenizer::from_tokens_and_scores(&tokens, Some(&scores)).expect("spm");
        assert_tokenizer_round_trips_all_bytes(&Tokenizer::SentencePiece(tok.clone()));

        let input = b"\x00 hello\xff\n hello";
        let ids = tok.tokenize_bytes(input).expect("tokenize");
        assert_eq!(tok.detokenize_bytes(&ids).expect("detok"), input);
    }

    #[test]
    fn sentencepiece_breaks_duplicate_piece_ties_deterministically() {
        let mut tokens = canonical_byte_tokens();
        let low_score = tokens.len() as u32;
        tokens.push("▁x".to_owned());
        let high_score = tokens.len() as u32;
        tokens.push("▁x".to_owned());
        let same_score_low_id = tokens.len() as u32;
        tokens.push("▁y".to_owned());
        let _same_score_high_id = tokens.len() as u32;
        tokens.push("▁y".to_owned());

        let mut scores = vec![0.0; tokens.len()];
        scores[low_score as usize] = 1.0;
        scores[high_score as usize] = 2.0;
        scores[same_score_low_id as usize] = 3.0;
        scores[(same_score_low_id + 1) as usize] = 3.0;

        let tok =
            SentencePieceTokenizer::from_tokens_and_scores(&tokens, Some(&scores)).expect("spm");
        assert_eq!(tok.tokenize_bytes(b" x").expect("x"), [high_score]);
        assert_eq!(tok.tokenize_bytes(b" y").expect("y"), [same_score_low_id]);
    }

    #[test]
    fn sentencepiece_constructor_rejects_malformed_scores() {
        let tokens = canonical_byte_tokens();

        assert!(matches!(
            SentencePieceTokenizer::from_tokens_and_scores(&tokens, Some(&vec![0.0; 255])),
            Err(TokenError::ScoresLength)
        ));

        let mut scores = vec![0.0; tokens.len()];
        scores[17] = f32::INFINITY;

        assert!(matches!(
            SentencePieceTokenizer::from_tokens_and_scores(&tokens, Some(&scores)),
            Err(TokenError::NonFiniteScore)
        ));
    }

    #[test]
    fn tokenizer_from_gguf_selects_sentencepiece_for_llama_model() {
        let mut tokens = canonical_byte_tokens();
        let hello = tokens.len() as u32;
        tokens.push("▁hello".to_owned());
        let mut metadata = BTreeMap::new();
        metadata.insert(
            "tokenizer.ggml.tokens".to_owned(),
            det_gguf::MetadataValue::ArrayString(tokens),
        );
        metadata.insert(
            "tokenizer.ggml.model".to_owned(),
            det_gguf::MetadataValue::String("llama".to_owned()),
        );
        let gguf = det_gguf::Gguf::from_parts(3, metadata, Vec::new(), 0, 0);

        let tok = Tokenizer::from_gguf(&gguf).expect("tokenizer");
        assert!(matches!(tok, Tokenizer::SentencePiece(_)));
        assert_eq!(tok.tokenize_bytes(b" hello").expect("tokens"), [hello]);
    }

    #[test]
    fn tokenizer_from_gguf_selects_sentencepiece_for_scores() {
        let mut tokens = canonical_byte_tokens();
        let hello = tokens.len() as u32;
        tokens.push("▁hello".to_owned());
        let mut metadata = BTreeMap::new();
        metadata.insert(
            "tokenizer.ggml.tokens".to_owned(),
            det_gguf::MetadataValue::ArrayString(tokens.clone()),
        );
        metadata.insert(
            "tokenizer.ggml.scores".to_owned(),
            det_gguf::MetadataValue::ArrayF32(vec![0.0; tokens.len()]),
        );
        let gguf = det_gguf::Gguf::from_parts(3, metadata, Vec::new(), 0, 0);

        let tok = Tokenizer::from_gguf(&gguf).expect("tokenizer");
        assert!(matches!(tok, Tokenizer::SentencePiece(_)));
        assert_eq!(tok.tokenize_bytes(b" hello").expect("tokens"), [hello]);
    }

    #[test]
    fn sentencepiece_does_not_emit_control_tokens_from_token_type() {
        let mut tokens = canonical_byte_tokens();
        let control_hello = tokens.len() as u32;
        tokens.push("▁hello".to_owned());
        let mut token_types = vec![6; tokens.len()];
        token_types[control_hello as usize] = 3;

        let mut metadata = BTreeMap::new();
        metadata.insert(
            "tokenizer.ggml.tokens".to_owned(),
            det_gguf::MetadataValue::ArrayString(tokens),
        );
        metadata.insert(
            "tokenizer.ggml.model".to_owned(),
            det_gguf::MetadataValue::String("llama".to_owned()),
        );
        metadata.insert(
            "tokenizer.ggml.token_type".to_owned(),
            det_gguf::MetadataValue::ArrayI32(token_types),
        );
        let gguf = det_gguf::Gguf::from_parts(3, metadata, Vec::new(), 0, 0);
        let tok = Tokenizer::from_gguf(&gguf).expect("tokenizer");

        assert_ne!(
            tok.tokenize_bytes(b" hello").expect("tokens"),
            [control_hello]
        );
        assert_eq!(
            tok.detokenize_bytes(&[control_hello]).expect("detok"),
            b" hello"
        );
    }

    #[test]
    fn rejects_token_type_length_mismatch() {
        let tokens = canonical_byte_tokens();
        let mut metadata = BTreeMap::new();
        metadata.insert(
            "tokenizer.ggml.tokens".to_owned(),
            det_gguf::MetadataValue::ArrayString(tokens),
        );
        metadata.insert(
            "tokenizer.ggml.token_type".to_owned(),
            det_gguf::MetadataValue::ArrayI32(vec![6; 255]),
        );
        let gguf = det_gguf::Gguf::from_parts(3, metadata, Vec::new(), 0, 0);

        assert!(matches!(
            Tokenizer::from_gguf(&gguf),
            Err(TokenError::TokenTypeLength)
        ));
    }

    #[test]
    fn rejects_unknown_token_type_values() {
        let tokens = canonical_byte_tokens();
        for value in [-1, 0, 7] {
            let mut metadata = BTreeMap::new();
            metadata.insert(
                "tokenizer.ggml.tokens".to_owned(),
                det_gguf::MetadataValue::ArrayString(tokens.clone()),
            );
            metadata.insert(
                "tokenizer.ggml.token_type".to_owned(),
                det_gguf::MetadataValue::ArrayI32(vec![value; tokens.len()]),
            );
            let gguf = det_gguf::Gguf::from_parts(3, metadata, Vec::new(), 0, 0);
            assert!(matches!(
                Tokenizer::from_gguf(&gguf),
                Err(TokenError::InvalidTokenType(actual)) if actual == i64::from(value)
            ));
        }

        let mut metadata = BTreeMap::new();
        metadata.insert(
            "tokenizer.ggml.tokens".to_owned(),
            det_gguf::MetadataValue::ArrayString(tokens.clone()),
        );
        metadata.insert(
            "tokenizer.ggml.token_type".to_owned(),
            det_gguf::MetadataValue::ArrayU32(vec![7; tokens.len()]),
        );
        let gguf = det_gguf::Gguf::from_parts(3, metadata, Vec::new(), 0, 0);
        assert!(matches!(
            Tokenizer::from_gguf(&gguf),
            Err(TokenError::InvalidTokenType(7))
        ));
    }

    #[test]
    fn rejects_scores_length_mismatch() {
        let tokens = canonical_byte_tokens();
        let mut metadata = BTreeMap::new();
        metadata.insert(
            "tokenizer.ggml.tokens".to_owned(),
            det_gguf::MetadataValue::ArrayString(tokens),
        );
        metadata.insert(
            "tokenizer.ggml.scores".to_owned(),
            det_gguf::MetadataValue::ArrayF32(vec![0.0; 255]),
        );
        let gguf = det_gguf::Gguf::from_parts(3, metadata, Vec::new(), 0, 0);

        assert!(matches!(
            Tokenizer::from_gguf(&gguf),
            Err(TokenError::ScoresLength)
        ));
    }

    #[test]
    fn rejects_nonfinite_scores() {
        let tokens = canonical_byte_tokens();
        let mut scores = vec![0.0; tokens.len()];
        scores[17] = f32::NAN;

        let mut metadata = BTreeMap::new();
        metadata.insert(
            "tokenizer.ggml.tokens".to_owned(),
            det_gguf::MetadataValue::ArrayString(tokens),
        );
        metadata.insert(
            "tokenizer.ggml.scores".to_owned(),
            det_gguf::MetadataValue::ArrayF32(scores),
        );
        let gguf = det_gguf::Gguf::from_parts(3, metadata, Vec::new(), 0, 0);

        assert!(matches!(
            Tokenizer::from_gguf(&gguf),
            Err(TokenError::NonFiniteScore)
        ));
    }

    fn canonical_byte_tokens() -> Vec<String> {
        (0..=255).map(|b| format!("<0x{b:02X}>")).collect()
    }

    fn assert_tokenizer_round_trips_all_bytes(tokenizer: &Tokenizer) {
        let input = (0..=255u8).collect::<Vec<_>>();
        let tokens = tokenizer.tokenize_bytes(&input).expect("tokenize");
        assert_eq!(tokenizer.detokenize_bytes(&tokens).expect("detok"), input);
    }

    fn gpt2_byte_unicode_token_bytes(byte: u8) -> Vec<u8> {
        let code = if (33..=126).contains(&(byte as u32))
            || (161..=172).contains(&(byte as u32))
            || (174..=255).contains(&(byte as u32))
        {
            byte as u32
        } else {
            let mut n = 0u32;
            let mut mapped = None;
            for candidate in 0u32..=255 {
                if (33..=126).contains(&candidate)
                    || (161..=172).contains(&candidate)
                    || (174..=255).contains(&candidate)
                {
                    continue;
                }
                if candidate == byte as u32 {
                    mapped = Some(256 + n);
                    break;
                }
                n += 1;
            }
            mapped.expect("byte mapped")
        };
        char::from_u32(code)
            .expect("codepoint")
            .to_string()
            .into_bytes()
    }
}
