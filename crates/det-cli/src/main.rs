use std::{env, fs, process::ExitCode};

use det_num::{run_canary as run_numeric_canary, Sha256};

const RUNTIME_CANARY_EXPECTED: [u8; 32] = [
    0x05, 0x5d, 0x04, 0xe3, 0x36, 0xf7, 0x48, 0x12, 0x9f, 0x5d, 0x80, 0x25, 0x0f, 0xe9, 0x45, 0xeb,
    0xa6, 0xc8, 0x28, 0x93, 0xc2, 0x89, 0xce, 0x28, 0x4d, 0xc0, 0x9e, 0xcb, 0xc8, 0x03, 0x50, 0x95,
];

fn main() -> ExitCode {
    match real_main() {
        Ok(()) => ExitCode::SUCCESS,
        Err(e) => {
            eprintln!("detllm: {e}");
            ExitCode::FAILURE
        }
    }
}

fn real_main() -> Result<(), String> {
    let mut args = env::args().skip(1);
    let command = args.next();
    if command.as_deref() != Some("selftest") {
        run_canaries()?;
    }
    match command.as_deref() {
        Some("selftest") => {
            run_canaries()?;
            println!("ok");
            Ok(())
        }
        Some("gguf-dump") => gguf_dump(args.collect()),
        Some("sha256") => {
            let path = args.next().ok_or("usage: detllm sha256 file")?;
            let bytes = fs::read(path).map_err(|e| e.to_string())?;
            let mut h = Sha256::new();
            h.update(&bytes);
            println!("{}", hex(&h.finalize()));
            Ok(())
        }
        Some("fixture-logits-hash") => {
            println!("{}", hex(&fixture_logits_hash()?));
            Ok(())
        }
        Some("quant-kernel-hash") => {
            println!("{}", hex(&quant_kernel_hash()?));
            Ok(())
        }
        Some("tokenize") => tokenize(args.collect()),
        Some("logits") => logits(args.collect()),
        Some("compress") => compress(args.collect()),
        Some("decompress") => decompress(args.collect()),
        _ => Err(
            "usage: detllm <selftest|gguf-dump|sha256|fixture-logits-hash|quant-kernel-hash|tokenize|logits|compress|decompress>"
                .to_owned(),
        ),
    }
}

fn gguf_dump(args: Vec<String>) -> Result<(), String> {
    if args.len() != 1 {
        return Err("usage: detllm gguf-dump model.gguf".to_owned());
    }
    let bytes = fs::read(&args[0]).map_err(|e| e.to_string())?;
    let gguf = det_gguf::parse(&bytes).map_err(|e| format!("{e:?}"))?;
    println!("version: {}", gguf.version);
    println!("metadata: {}", gguf.metadata.len());
    for (key, value) in &gguf.metadata {
        println!("metadata {key} {}", metadata_summary(value));
    }
    println!("tensors: {}", gguf.tensors.len());
    for tensor in &gguf.tensors {
        println!(
            "tensor {} {:?} type={} offset={}",
            tensor.name,
            tensor.dimensions,
            tensor.ty.raw(),
            tensor.offset
        );
    }
    Ok(())
}

fn metadata_summary(value: &det_gguf::MetadataValue) -> String {
    match value {
        det_gguf::MetadataValue::U8(v) => format!("u8={v}"),
        det_gguf::MetadataValue::I8(v) => format!("i8={v}"),
        det_gguf::MetadataValue::U16(v) => format!("u16={v}"),
        det_gguf::MetadataValue::I16(v) => format!("i16={v}"),
        det_gguf::MetadataValue::U32(v) => format!("u32={v}"),
        det_gguf::MetadataValue::I32(v) => format!("i32={v}"),
        det_gguf::MetadataValue::U64(v) => format!("u64={v}"),
        det_gguf::MetadataValue::I64(v) => format!("i64={v}"),
        det_gguf::MetadataValue::F32(v) => format!("f32={v:?}"),
        det_gguf::MetadataValue::F64(v) => format!("f64={v:?}"),
        det_gguf::MetadataValue::Bool(v) => format!("bool={v}"),
        det_gguf::MetadataValue::String(v) => format!("string={}", summarize_str(v)),
        det_gguf::MetadataValue::ArrayU8(v) => format!("array<u8>[{}]", v.len()),
        det_gguf::MetadataValue::ArrayI8(v) => format!("array<i8>[{}]", v.len()),
        det_gguf::MetadataValue::ArrayU16(v) => format!("array<u16>[{}]", v.len()),
        det_gguf::MetadataValue::ArrayI16(v) => format!("array<i16>[{}]", v.len()),
        det_gguf::MetadataValue::ArrayU32(v) => format!("array<u32>[{}]", v.len()),
        det_gguf::MetadataValue::ArrayI32(v) => format!("array<i32>[{}]", v.len()),
        det_gguf::MetadataValue::ArrayF32(v) => format!("array<f32>[{}]", v.len()),
        det_gguf::MetadataValue::ArrayBool(v) => format!("array<bool>[{}]", v.len()),
        det_gguf::MetadataValue::ArrayString(v) => format!("array<string>[{}]", v.len()),
        det_gguf::MetadataValue::ArrayU64(v) => format!("array<u64>[{}]", v.len()),
        det_gguf::MetadataValue::ArrayI64(v) => format!("array<i64>[{}]", v.len()),
        det_gguf::MetadataValue::ArrayF64(v) => format!("array<f64>[{}]", v.len()),
    }
}

fn summarize_str(value: &str) -> String {
    const MAX_CHARS: usize = 80;
    let mut out = String::new();
    for (i, ch) in value.chars().enumerate() {
        if i == MAX_CHARS {
            out.push_str("...");
            return out;
        }
        out.push(ch);
    }
    out
}

fn run_canaries() -> Result<(), String> {
    run_numeric_canary().map_err(|e| {
        format!(
            "numeric canary mismatch: expected {}, got {}",
            hex(&e.expected),
            hex(&e.actual)
        )
    })?;

    let actual = runtime_canary_hash()?;
    if actual != RUNTIME_CANARY_EXPECTED {
        return Err(format!(
            "runtime canary mismatch: expected {}, got {}",
            hex(&RUNTIME_CANARY_EXPECTED),
            hex(&actual)
        ));
    }
    Ok(())
}

fn runtime_canary_hash() -> Result<[u8; 32], String> {
    let mut h = Sha256::new();

    for seed in 0..16u32 {
        let q8 = q8_block(seed);
        let q4 = q4_block(seed ^ 0xa5a5_5a5a);
        let a = q8a_block(seed.wrapping_mul(1_664_525).wrapping_add(1_013_904_223));
        let q4k = q4k_block(seed ^ 0x9e37_79b9);
        let a4k = q8a_blocks_for_q4k(seed);
        let q6 = q6_block(seed ^ 0x517c_c1b7);
        let a6 = q8k_block_for_q6(seed);
        let y8 = det_quant::dot_q8_0_q8a_block(q8, a);
        let y4 = det_quant::dot_q4_0_q8a_block(q4, a);
        let y4k = det_quant::dot_q4_k_q8a(&[q4k], &a4k)
            .map_err(|e| format!("runtime canary Q4_K dot error: {e:?}"))?;
        let y6 = det_quant::dot_q6_k_q8k(&[q6], &[a6])
            .map_err(|e| format!("runtime canary Q6_K dot error: {e:?}"))?;
        if !y8.is_finite() || !y4.is_finite() || !y4k.is_finite() || !y6.is_finite() {
            return Err("runtime canary quantized dot produced non-finite output".to_owned());
        }
        h.update(&y8.to_bits().to_le_bytes());
        h.update(&y4.to_bits().to_le_bytes());
        h.update(&y4k.to_bits().to_le_bytes());
        h.update(&y6.to_bits().to_le_bytes());
    }

    let q_input: [f32; det_quant::BLOCK] = core::array::from_fn(|i| match i {
        0 => f32::from_bits(1),
        1 => -f32::from_bits(2),
        _ => (((i as i32 % 11) - 5) as f32) * 0.03125,
    });
    let q = det_quant::quantize_q8a_block(&q_input);
    h.update(&q.d.to_bits().to_le_bytes());
    for &v in &q.q {
        h.update(&[v as u8]);
    }

    let x = [
        f32::from_bits(1),
        -0.75,
        1.25,
        -2.0,
        3.5,
        -4.25,
        0.5,
        -0.125,
    ];
    let w = [1.0, -0.5, 0.25, -1.5, 2.0, -2.5, 0.75, -0.25];
    let mut rms = [0.0f32; 8];
    det_model::rmsnorm(&x, &w, 1e-5, &mut rms)
        .map_err(|e| format!("runtime canary rmsnorm error: {e:?}"))?;
    for &v in &rms {
        h.update(&v.to_bits().to_le_bytes());
    }

    let cdf = det_coder::logits_to_cdf(&[
        0.0,
        -1.0,
        3.25,
        3.25,
        -88.5,
        f32::from_bits(1),
        -12.0,
        0.5,
        -0.25,
        2.0,
        -1000.0,
    ])
    .map_err(|e| format!("runtime canary CDF error: {e:?}"))?;
    for &freq in &cdf.freq {
        h.update(&freq.to_le_bytes());
    }
    for &cum in &cdf.cum {
        h.update(&cum.to_le_bytes());
    }
    h.update(&cdf.total.to_le_bytes());

    Ok(h.finalize())
}

fn fixture_logits_hash() -> Result<[u8; 32], String> {
    let model = fixture_model()?;
    model
        .logits_hash_for_tokens(&[0, 1, 2, 3, 0, 2])
        .map_err(|e| format!("fixture logits error: {e:?}"))
}

fn quant_kernel_hash() -> Result<[u8; 32], String> {
    let mut h = Sha256::new();
    const CASES: u32 = 1_000_000;
    const Q4K_CASES: u32 = 4_096;
    const Q6K_CASES: u32 = 4_096;
    for seed in 0..CASES {
        let q8 = q8_block(seed);
        let q4 = q4_block(seed ^ 0xa5a5_5a5a);
        let a = q8a_block(seed.wrapping_mul(1_664_525).wrapping_add(1_013_904_223));
        let y8 = det_quant::dot_q8_0_q8a_block(q8, a);
        let y4 = det_quant::dot_q4_0_q8a_block(q4, a);
        if !y8.is_finite() || !y4.is_finite() {
            return Err("quant kernel produced non-finite output".to_owned());
        }
        h.update(&y8.to_bits().to_le_bytes());
        h.update(&y4.to_bits().to_le_bytes());
    }
    for seed in 0..Q4K_CASES {
        let q4k = q4k_block(seed ^ 0x9e37_79b9);
        let a4k = q8a_blocks_for_q4k(seed);
        let y4k = det_quant::dot_q4_k_q8a(&[q4k], &a4k)
            .map_err(|e| format!("quant Q4_K kernel error: {e:?}"))?;
        if !y4k.is_finite() {
            return Err("quant Q4_K kernel produced non-finite output".to_owned());
        }
        h.update(&y4k.to_bits().to_le_bytes());
    }
    for seed in 0..Q6K_CASES {
        let q6 = q6_block(seed ^ 0x517c_c1b7);
        let a6 = q8k_block_for_q6(seed);
        let y6 = det_quant::dot_q6_k_q8k(&[q6], &[a6])
            .map_err(|e| format!("quant Q6_K kernel error: {e:?}"))?;
        if !y6.is_finite() {
            return Err("quant Q6_K kernel produced non-finite output".to_owned());
        }
        h.update(&y6.to_bits().to_le_bytes());
    }
    Ok(h.finalize())
}

fn q8_block(seed: u32) -> det_quant::Q8_0Block {
    let mut q = [0i8; det_quant::BLOCK];
    let mut x = seed;
    for dst in &mut q {
        x = x.wrapping_mul(1_103_515_245).wrapping_add(12_345);
        *dst = (((x >> 16) % 255) as i16 - 127) as i8;
    }
    det_quant::Q8_0Block {
        d: f32::from_bits(0x3c00_0000 + (seed & 0xff)),
        q,
    }
}

fn q8a_block(seed: u32) -> det_quant::Q8ABlock {
    let mut q = [0i8; det_quant::BLOCK];
    let mut x = seed;
    for dst in &mut q {
        x = x.wrapping_mul(22_695_477).wrapping_add(1);
        *dst = (((x >> 17) % 255) as i16 - 127) as i8;
    }
    det_quant::Q8ABlock {
        d: f32::from_bits(0x3d00_0000 + (seed & 0xff)),
        q,
    }
}

fn q4_block(seed: u32) -> det_quant::Q4_0Block {
    let mut qs = [0u8; 16];
    let mut x = seed;
    for byte in &mut qs {
        x = x.wrapping_mul(747_796_405).wrapping_add(2_891_336_453);
        let lo = ((x >> 8) & 0x0f) as u8;
        let hi = ((x >> 20) & 0x0f) as u8;
        *byte = lo | (hi << 4);
    }
    det_quant::Q4_0Block {
        d: f32::from_bits(0x3c80_0000 + (seed & 0xff)),
        qs,
    }
}

fn q4k_block(seed: u32) -> det_quant::Q4KBlock {
    let mut scales = [0u8; 12];
    let mut qs = [0u8; 128];
    let mut x = seed;
    for i in 0..4 {
        x = x.wrapping_mul(1_664_525).wrapping_add(1_013_904_223);
        let scale = ((x >> 10) & 0x3f) as u8;
        x = x.wrapping_mul(22_695_477).wrapping_add(1);
        let min = ((x >> 11) & 0x3f) as u8;
        scales[i] = scale & 0x3f;
        scales[i + 4] = min & 0x3f;
    }
    for i in 4..8 {
        x = x.wrapping_mul(747_796_405).wrapping_add(2_891_336_453);
        let scale = ((x >> 8) & 0x3f) as u8;
        x = x.wrapping_mul(1_103_515_245).wrapping_add(12_345);
        let min = ((x >> 12) & 0x3f) as u8;
        scales[i + 4] = (scale & 0x0f) | ((min & 0x0f) << 4);
        scales[i - 4] |= (scale >> 4) << 6;
        scales[i] |= (min >> 4) << 6;
    }
    for byte in &mut qs {
        x = x.wrapping_mul(1_103_515_245).wrapping_add(12_345);
        let lo = ((x >> 16) & 0x0f) as u8;
        x = x.wrapping_mul(747_796_405).wrapping_add(2_891_336_453);
        let hi = ((x >> 20) & 0x0f) as u8;
        *byte = lo | (hi << 4);
    }
    det_quant::Q4KBlock {
        d: f32::from_bits(0x3b00_0000 + (seed & 0xff)),
        dmin: f32::from_bits(0x3a80_0000 + ((seed >> 8) & 0xff)),
        scales,
        qs,
    }
}

fn q6_block(seed: u32) -> det_quant::Q6KBlock {
    let mut ql = [0u8; 128];
    let mut qh = [0u8; 64];
    let mut scales = [0i8; 16];
    let mut x = seed;
    for byte in &mut ql {
        x = x.wrapping_mul(1_664_525).wrapping_add(1_013_904_223);
        *byte = (x >> 13) as u8;
    }
    for byte in &mut qh {
        x = x.wrapping_mul(22_695_477).wrapping_add(1);
        *byte = (x >> 11) as u8;
    }
    for scale in &mut scales {
        x = x.wrapping_mul(747_796_405).wrapping_add(2_891_336_453);
        *scale = (((x >> 16) % 127) as i16 - 63) as i8;
    }
    det_quant::Q6KBlock {
        d: f32::from_bits(0x3b80_0000 + (seed & 0xff)),
        ql,
        qh,
        scales,
    }
}

fn q8a_blocks_for_q4k(seed: u32) -> [det_quant::Q8ABlock; det_quant::Q4K_BLOCK / det_quant::BLOCK] {
    core::array::from_fn(|i| q8a_block(seed.wrapping_add((i as u32).wrapping_mul(0x85eb_ca6b))))
}

fn q8k_block_for_q6(seed: u32) -> det_quant::Q8KBlock {
    let input: [f32; det_quant::Q6K_BLOCK] = core::array::from_fn(|i| {
        let x = seed
            .wrapping_add((i as u32).wrapping_mul(0x9e37_79b9))
            .wrapping_mul(747_796_405)
            .wrapping_add(2_891_336_453);
        (((x >> 9) as i32 % 4093) - 2046) as f32 / 257.0
    });
    det_quant::quantize_q8k_block(&input)
}

fn fixture_model() -> Result<det_model::F32Llama, String> {
    let cfg = det_model::LlamaConfig {
        block_count: 1,
        embedding_length: 4,
        feed_forward_length: 6,
        head_count: 2,
        head_count_kv: 1,
        rms_epsilon: 1e-5,
        attention_scale: 1.0 / 2.0f32.sqrt(),
        rope_freq_base: 10_000.0,
        rope_dimension_count: 2,
        rope_pairing: det_model::RopePairing::Adjacent,
        context_length: 8,
    };
    Ok(det_model::F32Llama {
        config: cfg,
        token_embedding: fixture_matrix(4, 4, 0.01)?.into(),
        layers: vec![det_model::F32LayerWeights {
            attention_norm: vec![1.0; 4],
            wq: fixture_matrix(4, 4, 0.02)?.into(),
            attn_q_bias: None,
            wk: fixture_matrix(2, 4, -0.015)?.into(),
            attn_k_bias: None,
            wv: fixture_matrix(2, 4, 0.025)?.into(),
            attn_v_bias: None,
            wo: fixture_matrix(4, 4, -0.02)?.into(),
            ffn_norm: vec![1.0; 4],
            w_gate: fixture_matrix(6, 4, 0.03)?.into(),
            w_up: fixture_matrix(6, 4, -0.025)?.into(),
            w_down: fixture_matrix(4, 6, 0.018)?.into(),
        }],
        output_norm: vec![1.0; 4],
        output: fixture_matrix(4, 4, 0.022)?.into(),
    })
}

fn fixture_matrix(rows: usize, cols: usize, scale: f32) -> Result<det_model::F32Matrix, String> {
    let data = (0..rows * cols)
        .map(|i| (((i % 9) as f32) - 4.0) * scale)
        .collect();
    det_model::F32Matrix::new(rows, cols, data).map_err(|e| format!("fixture matrix error: {e:?}"))
}

fn compress(args: Vec<String>) -> Result<(), String> {
    let opts = io_model_opts(&args, "compress")?;
    apply_threads(opts.threads)?;
    let input = fs::read(&opts.input).map_err(|e| e.to_string())?;
    let loaded = LoadedModel::load(&opts.model)?;
    let symbols = loaded
        .tokenizer
        .codec_symbols(&input, loaded.model.output.rows())
        .map_err(|e| format!("tokenize error: {e:?}"))?;

    let n_ctx = opts.n_ctx.unwrap_or(loaded.model.config.context_length);
    let overlap = n_ctx / 4;
    validate_window(n_ctx, overlap, loaded.model.config.context_length)?;

    let payload = encode_symbols_with_model(&loaded.model, &symbols, n_ctx, overlap, true)?;
    let header = det_coder::DtlzHeader {
        flags: det_coder::FLAG_BYTE_ESCAPES,
        model_sha256: loaded.model_sha256,
        n_ctx: n_ctx as u32,
        overlap: overlap as u32,
        orig_len: input.len() as u64,
    };
    let mut out = Vec::with_capacity(det_coder::file::HEADER_LEN + payload.len());
    out.extend_from_slice(
        &header
            .encode_checked()
            .map_err(|e| format!("DTLZ header error: {e:?}"))?,
    );
    out.extend_from_slice(&payload);
    fs::write(&opts.output, out).map_err(|e| e.to_string())
}

fn decompress(args: Vec<String>) -> Result<(), String> {
    let opts = io_model_opts(&args, "decompress")?;
    if opts.n_ctx.is_some() {
        return Err("decompress: --n-ctx is stored in the DTLZ header".to_owned());
    }
    apply_threads(opts.threads)?;
    let encoded = fs::read(&opts.input).map_err(|e| e.to_string())?;
    let header =
        det_coder::DtlzHeader::decode(&encoded).map_err(|e| format!("DTLZ header error: {e:?}"))?;
    let payload = encoded
        .get(det_coder::file::HEADER_LEN..)
        .ok_or("truncated DTLZ payload")?;
    let loaded = LoadedModel::load(&opts.model)?;
    if loaded.model_sha256 != header.model_sha256 {
        return Err("model SHA-256 does not match compressed file header".to_owned());
    }

    let n_ctx = header.n_ctx as usize;
    let overlap = header.overlap as usize;
    validate_window(n_ctx, overlap, loaded.model.config.context_length)?;
    let use_byte_escapes = header.flags & det_coder::FLAG_BYTE_ESCAPES != 0;

    let restored_len =
        usize::try_from(header.orig_len).map_err(|_| "orig_len does not fit usize")?;
    let decoded = decode_bytes_with_model(
        &loaded.model,
        &loaded.tokenizer,
        payload,
        restored_len,
        n_ctx,
        overlap,
        use_byte_escapes,
    )?;
    let canonical_payload = encode_symbols_with_model(
        &loaded.model,
        &decoded.symbols,
        n_ctx,
        overlap,
        use_byte_escapes,
    )?;
    if canonical_payload != payload {
        return Err(
            "DTLZ payload is not the canonical encoding for the restored stream".to_owned(),
        );
    }
    fs::write(&opts.output, &decoded.bytes[..restored_len]).map_err(|e| e.to_string())
}

#[cfg(test)]
fn encode_tokens_with_model(
    model: &det_model::F32Llama,
    tokens: &[usize],
    n_ctx: usize,
    overlap: usize,
) -> Result<Vec<u8>, String> {
    encode_symbols_with_model(model, tokens, n_ctx, overlap, true)
}

fn encode_symbols_with_model(
    model: &det_model::F32Llama,
    symbols: &[usize],
    n_ctx: usize,
    overlap: usize,
    use_byte_escapes: bool,
) -> Result<Vec<u8>, String> {
    validate_window(n_ctx, overlap, model.config.context_length)?;
    let mut enc = det_coder::RangeEncoder::new();
    let mut state = WindowedModelState::new(model, n_ctx, use_byte_escapes)?;
    let mut context_tokens = Vec::new();
    let vocab_len = model.output.rows();
    for &symbol in symbols {
        state.sync(context_tokens.len(), &context_tokens, overlap)?;
        let cdf = state.cdf()?;
        encode_symbol(&mut enc, cdf, symbol)?;
        if use_byte_escapes && !det_token::Tokenizer::codec_symbol_is_token(symbol, vocab_len) {
            continue;
        }
        if symbol >= vocab_len {
            return Err(format!("symbol {symbol} is outside vocabulary"));
        }
        {
            context_tokens.push(symbol);
            state.advance(symbol)?;
        }
    }
    Ok(enc.finish())
}

#[cfg(test)]
fn decode_tokens_with_model(
    model: &det_model::F32Llama,
    payload: &[u8],
    token_len: usize,
    n_ctx: usize,
    overlap: usize,
) -> Result<Vec<usize>, String> {
    validate_window(n_ctx, overlap, model.config.context_length)?;
    let mut dec =
        det_coder::RangeDecoder::new(payload).map_err(|e| format!("range decoder error: {e:?}"))?;
    let mut tokens = Vec::with_capacity(token_len);
    let mut state = WindowedModelState::new(model, n_ctx, true)?;
    let vocab_len = model.output.rows();
    while tokens.len() < token_len {
        state.sync(tokens.len(), &tokens, overlap)?;
        let cdf = state.cdf()?;
        let symbol = decode_symbol(&mut dec, cdf)?;
        if !det_token::Tokenizer::codec_symbol_is_token(symbol, vocab_len) {
            return Err(format!("decoded byte escape {symbol} in token-only stream"));
        }
        tokens.push(symbol);
        state.advance(symbol)?;
    }
    Ok(tokens)
}

struct DecodedBytes {
    bytes: Vec<u8>,
    symbols: Vec<usize>,
}

fn decode_bytes_with_model(
    model: &det_model::F32Llama,
    tokenizer: &det_token::Tokenizer,
    payload: &[u8],
    byte_len: usize,
    n_ctx: usize,
    overlap: usize,
    use_byte_escapes: bool,
) -> Result<DecodedBytes, String> {
    validate_window(n_ctx, overlap, model.config.context_length)?;
    let mut dec =
        det_coder::RangeDecoder::new(payload).map_err(|e| format!("range decoder error: {e:?}"))?;
    let mut context_tokens = Vec::new();
    let mut symbols = Vec::new();
    let mut bytes = Vec::with_capacity(byte_len.min(8192));
    let mut state = WindowedModelState::new(model, n_ctx, use_byte_escapes)?;
    let vocab_len = model.output.rows();
    while bytes.len() < byte_len {
        state.sync(context_tokens.len(), &context_tokens, overlap)?;
        let cdf = state.cdf()?;
        let symbol = decode_symbol(&mut dec, cdf)?;
        let piece = if use_byte_escapes {
            tokenizer
                .decode_codec_symbol(symbol, vocab_len)
                .map_err(|e| format!("detokenize error: {e:?}"))?
        } else {
            let token =
                u32::try_from(symbol).map_err(|_| format!("decoded token too large: {symbol}"))?;
            tokenizer
                .detokenize_bytes(&[token])
                .map_err(|e| format!("detokenize error: {e:?}"))?
        };
        if piece.is_empty() {
            return Err(format!("decoded symbol {symbol} produced no bytes"));
        }
        bytes.extend_from_slice(&piece);
        symbols.push(symbol);
        if use_byte_escapes && !det_token::Tokenizer::codec_symbol_is_token(symbol, vocab_len) {
            continue;
        }
        {
            context_tokens.push(symbol);
            state.advance(symbol)?;
        }
    }
    Ok(DecodedBytes { bytes, symbols })
}

struct WindowedModelState<'a> {
    model: &'a det_model::F32Llama,
    n_ctx: usize,
    window_start: usize,
    context_len: usize,
    rope: det_model::RopeTables,
    cache: det_model::KvCache,
    workspace: det_model::ForwardWorkspace,
    logits: Vec<f32>,
    uniform_cdf: det_coder::Cdf,
    cdf_scratch: det_coder::CdfScratch,
    use_byte_escapes: bool,
}

impl<'a> WindowedModelState<'a> {
    fn new(
        model: &'a det_model::F32Llama,
        n_ctx: usize,
        use_byte_escapes: bool,
    ) -> Result<Self, String> {
        let rope = det_model::RopeTables::llama(model.config, n_ctx)
            .map_err(|e| format!("rope error: {e:?}"))?;
        let cache =
            det_model::KvCache::new(model.config).map_err(|e| format!("cache error: {e:?}"))?;
        let workspace = model
            .forward_workspace(n_ctx)
            .map_err(|e| format!("workspace error: {e:?}"))?;
        let uniform_cdf = if use_byte_escapes {
            det_coder::uniform_cdf_with_byte_escapes(model.output.rows())
        } else {
            det_coder::uniform_cdf(model.output.rows())
        }
        .map_err(|e| format!("uniform CDF error: {e:?}"))?;
        Ok(Self {
            model,
            n_ctx,
            window_start: 0,
            context_len: 0,
            rope,
            cache,
            workspace,
            logits: vec![0.0f32; model.output.rows()],
            uniform_cdf,
            cdf_scratch: det_coder::CdfScratch::default(),
            use_byte_escapes,
        })
    }

    fn sync(&mut self, pos: usize, tokens: &[usize], overlap: usize) -> Result<(), String> {
        if pos != tokens.len() {
            return Err("codec state position does not match token prefix length".to_owned());
        }
        let next_start = next_window_start(pos, self.window_start, self.n_ctx, overlap);
        if next_start != self.window_start {
            self.replay(next_start, &tokens[next_start..pos])?;
        }
        Ok(())
    }

    fn replay(&mut self, window_start: usize, context: &[usize]) -> Result<(), String> {
        self.cache = det_model::KvCache::new(self.model.config)
            .map_err(|e| format!("cache error: {e:?}"))?;
        self.logits.fill(0.0);
        self.window_start = window_start;
        self.context_len = 0;
        for &token in context {
            self.advance(token)?;
        }
        Ok(())
    }

    fn cdf(&mut self) -> Result<&det_coder::Cdf, String> {
        if self.context_len == 0 {
            Ok(&self.uniform_cdf)
        } else if self.use_byte_escapes {
            det_coder::logits_to_cdf_with_byte_escapes(&self.logits, &mut self.cdf_scratch)
                .map_err(|e| format!("CDF error: {e:?}"))
        } else {
            det_coder::logits_to_cdf_with_scratch(&self.logits, &mut self.cdf_scratch)
                .map_err(|e| format!("CDF error: {e:?}"))
        }
    }

    fn advance(&mut self, token: usize) -> Result<(), String> {
        if self.context_len >= self.n_ctx {
            return Err("context window invariant violated".to_owned());
        }
        self.model
            .forward_one_with_workspace(
                token,
                self.context_len,
                &self.rope,
                &mut self.cache,
                &mut self.logits,
                &mut self.workspace,
            )
            .map_err(|e| format!("forward error: {e:?}"))?;
        self.context_len += 1;
        Ok(())
    }
}

fn next_window_start(pos: usize, window_start: usize, n_ctx: usize, overlap: usize) -> usize {
    if pos.saturating_sub(window_start) >= n_ctx {
        pos.saturating_sub(overlap)
    } else {
        window_start
    }
}

#[cfg(test)]
fn cdf_for_context(
    model: &det_model::F32Llama,
    context: &[usize],
    n_ctx: usize,
) -> Result<det_coder::Cdf, String> {
    if context.len() > n_ctx || n_ctx > model.config.context_length {
        return Err("context window invariant violated".to_owned());
    }
    let vocab = model.output.rows();
    if context.is_empty() {
        return det_coder::uniform_cdf_with_byte_escapes(vocab)
            .map_err(|e| format!("uniform CDF error: {e:?}"));
    }
    let rope = det_model::RopeTables::llama(model.config, context.len())
        .map_err(|e| format!("rope error: {e:?}"))?;
    let mut cache =
        det_model::KvCache::new(model.config).map_err(|e| format!("cache error: {e:?}"))?;
    let mut logits = vec![0.0f32; vocab];
    for (pos, &token) in context.iter().enumerate() {
        model
            .forward_one(token, pos, &rope, &mut cache, &mut logits)
            .map_err(|e| format!("forward error: {e:?}"))?;
    }
    let mut scratch = det_coder::CdfScratch::default();
    det_coder::logits_to_cdf_with_byte_escapes(&logits, &mut scratch)
        .cloned()
        .map_err(|e| format!("CDF error: {e:?}"))
}

fn validate_window(n_ctx: usize, overlap: usize, model_ctx: usize) -> Result<(), String> {
    if n_ctx == 0 {
        return Err("n_ctx must be greater than zero".to_owned());
    }
    if n_ctx > model_ctx {
        return Err(format!(
            "n_ctx {n_ctx} exceeds model context length {model_ctx}"
        ));
    }
    if overlap >= n_ctx {
        return Err(format!(
            "overlap {overlap} must be smaller than n_ctx {n_ctx}"
        ));
    }
    Ok(())
}

fn encode_symbol(
    enc: &mut det_coder::RangeEncoder,
    cdf: &det_coder::Cdf,
    token: usize,
) -> Result<(), String> {
    let (&cum, &freq) = cdf
        .cum
        .get(token)
        .zip(cdf.freq.get(token))
        .ok_or_else(|| format!("token {token} is outside vocabulary"))?;
    enc.encode(cum, freq as u64, cdf.total)
        .map_err(|e| format!("range encode error: {e:?}"))
}

fn decode_symbol(
    dec: &mut det_coder::RangeDecoder<'_>,
    cdf: &det_coder::Cdf,
) -> Result<usize, String> {
    let value = dec
        .decode_freq(cdf.total)
        .map_err(|e| format!("range decode error: {e:?}"))?;
    let token = cdf
        .symbol_for_validated(value)
        .ok_or_else(|| format!("CDF lookup failed for value {value}"))?;
    dec.advance(cdf.cum[token], cdf.freq[token] as u64, cdf.total)
        .map_err(|e| format!("range advance error: {e:?}"))?;
    Ok(token)
}

struct IoModelOpts {
    model: String,
    input: String,
    output: String,
    n_ctx: Option<usize>,
    threads: Option<usize>,
}

fn io_model_opts(args: &[String], command: &str) -> Result<IoModelOpts, String> {
    let mut model = None;
    let mut input = None;
    let mut output = None;
    let mut n_ctx = None;
    let mut threads = None;
    let mut i = 0usize;
    while i < args.len() {
        match args[i].as_str() {
            "-m" | "--model" => {
                i += 1;
                model = args.get(i).cloned();
            }
            "-i" | "--input" => {
                i += 1;
                input = args.get(i).cloned();
            }
            "-o" | "--output" => {
                i += 1;
                output = args.get(i).cloned();
            }
            "--n-ctx" => {
                i += 1;
                let raw = args
                    .get(i)
                    .ok_or_else(|| format!("{command}: missing value for --n-ctx"))?;
                n_ctx = Some(
                    raw.parse::<usize>()
                        .map_err(|_| format!("{command}: invalid --n-ctx value: {raw}"))?,
                );
            }
            "--threads" => {
                i += 1;
                let raw = args
                    .get(i)
                    .ok_or_else(|| format!("{command}: missing value for --threads"))?;
                let value = raw
                    .parse::<usize>()
                    .map_err(|_| format!("{command}: invalid --threads value: {raw}"))?;
                if value == 0 {
                    return Err(format!("{command}: --threads must be greater than zero"));
                }
                threads = Some(value);
            }
            other => return Err(format!("unknown {command} argument: {other}")),
        }
        i += 1;
    }
    Ok(IoModelOpts {
        model: model.ok_or_else(|| usage_io(command))?,
        input: input.ok_or_else(|| usage_io(command))?,
        output: output.ok_or_else(|| usage_io(command))?,
        n_ctx,
        threads,
    })
}

fn usage_io(command: &str) -> String {
    format!("usage: detllm {command} -m model.gguf -i input -o output")
}

struct LoadedModel {
    model_sha256: [u8; 32],
    model: det_model::F32Llama,
    tokenizer: det_token::Tokenizer,
}

impl LoadedModel {
    fn load(path: &str) -> Result<Self, String> {
        let bytes = fs::read(path).map_err(|e| e.to_string())?;
        let mut hasher = Sha256::new();
        hasher.update(&bytes);
        let model_sha256 = hasher.finalize();
        let gguf = det_gguf::parse(&bytes).map_err(|e| format!("GGUF parse error: {e:?}"))?;
        let tokenizer_vocab_len = gguf_token_vocab_len(&gguf)?;
        let tokenizer =
            det_token::Tokenizer::from_gguf(&gguf).map_err(|e| format!("tokenizer error: {e}"))?;
        let model = det_model::F32Llama::from_gguf(&gguf, &bytes)
            .map_err(|e| format!("model load error: {e:?}"))?;
        validate_tokenizer_model_vocab_len(tokenizer_vocab_len, &model)?;
        validate_codec_vocab_len(model.output.rows())?;
        Ok(Self {
            model_sha256,
            model,
            tokenizer,
        })
    }
}

fn gguf_token_vocab_len(gguf: &det_gguf::Gguf) -> Result<usize, String> {
    match gguf.metadata_value("tokenizer.ggml.tokens") {
        Ok(det_gguf::MetadataValue::ArrayString(tokens)) => Ok(tokens.len()),
        Ok(_) => Err("tokenizer.ggml.tokens has the wrong metadata type".to_owned()),
        Err(e) => Err(format!("tokenizer.ggml.tokens metadata error: {e:?}")),
    }
}

fn validate_tokenizer_model_vocab_len(
    tokenizer_vocab_len: usize,
    model: &det_model::F32Llama,
) -> Result<(), String> {
    validate_vocab_lengths(tokenizer_vocab_len, model.output.rows())
}

fn validate_vocab_lengths(
    tokenizer_vocab_len: usize,
    model_vocab_len: usize,
) -> Result<(), String> {
    if tokenizer_vocab_len != model_vocab_len {
        return Err(format!(
            "tokenizer vocabulary length {tokenizer_vocab_len} does not match model vocabulary {model_vocab_len}",
        ));
    }
    Ok(())
}

fn gguf_model_vocab_len(gguf: &det_gguf::Gguf) -> Result<usize, String> {
    let tensor = match gguf.tensor("output.weight") {
        Ok(tensor) => tensor,
        Err(det_gguf::GgufError::TensorNotFound) => gguf
            .tensor("token_embd.weight")
            .map_err(|e| format!("token_embd.weight tensor error: {e:?}"))?,
        Err(e) => return Err(format!("output.weight tensor error: {e:?}")),
    };
    if tensor.dimensions.len() != 2 {
        return Err(format!("{} tensor must be 2-dimensional", tensor.name));
    }
    usize::try_from(tensor.dimensions[1])
        .map_err(|_| format!("{} vocabulary dimension is too large", tensor.name))
}

fn validate_codec_vocab_len(vocab_len: usize) -> Result<(), String> {
    let symbol_count = vocab_len
        .checked_add(det_coder::BYTE_ESCAPE_SYMBOLS)
        .ok_or_else(|| "codec symbol count overflow".to_owned())?;
    if symbol_count > det_coder::MAX_SYMBOLS {
        return Err(format!(
            "model vocabulary {vocab_len} plus {} byte escapes exceeds codec symbol limit {}",
            det_coder::BYTE_ESCAPE_SYMBOLS,
            det_coder::MAX_SYMBOLS
        ));
    }
    Ok(())
}

fn tokenize(args: Vec<String>) -> Result<(), String> {
    let mut model_path = None;
    let mut prompt = None;
    let mut input_path = None;
    let mut i = 0usize;
    while i < args.len() {
        match args[i].as_str() {
            "-m" | "--model" => {
                i += 1;
                model_path = args.get(i).cloned();
            }
            "-p" | "--prompt" => {
                i += 1;
                prompt = Some(
                    args.get(i)
                        .cloned()
                        .ok_or_else(|| "tokenize: missing value for --prompt".to_owned())?,
                );
            }
            "-i" | "--input" => {
                i += 1;
                input_path = Some(
                    args.get(i)
                        .cloned()
                        .ok_or_else(|| "tokenize: missing value for --input".to_owned())?,
                );
            }
            other => return Err(format!("unknown tokenize argument: {other}")),
        }
        i += 1;
    }

    let model_path =
        model_path.ok_or("usage: detllm tokenize -m model.gguf (-p prompt|--input file)")?;
    if prompt.is_some() == input_path.is_some() {
        return Err("tokenize: provide exactly one of --prompt or --input".to_owned());
    }

    let model_bytes = fs::read(&model_path).map_err(|e| e.to_string())?;
    let gguf = det_gguf::parse(&model_bytes).map_err(|e| format!("GGUF parse error: {e:?}"))?;
    let tokenizer_vocab_len = gguf_token_vocab_len(&gguf)?;
    let tokenizer =
        det_token::Tokenizer::from_gguf(&gguf).map_err(|e| format!("tokenizer error: {e}"))?;
    let model_vocab_len = gguf_model_vocab_len(&gguf)?;
    validate_vocab_lengths(tokenizer_vocab_len, model_vocab_len)?;

    let input = if let Some(prompt) = prompt {
        prompt.into_bytes()
    } else {
        fs::read(input_path.expect("input path")).map_err(|e| e.to_string())?
    };
    let token_ids = tokenizer
        .tokenize_bytes(&input)
        .map_err(|e| format!("tokenize error: {e:?}"))?;
    println!("{}", join_tokens(&token_ids));
    Ok(())
}

fn logits(args: Vec<String>) -> Result<(), String> {
    let mut model_path = None;
    let mut tokens = None;
    let mut prompt = None;
    let mut hash = false;
    let mut dump_path = None;
    let mut threads = None;
    let mut chunk_size = 1usize;
    let mut i = 0usize;
    while i < args.len() {
        match args[i].as_str() {
            "-m" | "--model" => {
                i += 1;
                model_path = args.get(i).cloned();
            }
            "--tokens" => {
                i += 1;
                tokens = args.get(i).cloned();
            }
            "--hash" => hash = true,
            "--dump" => {
                i += 1;
                dump_path = Some(
                    args.get(i)
                        .cloned()
                        .ok_or_else(|| "logits: missing value for --dump".to_owned())?,
                );
            }
            "-p" | "--prompt" => {
                i += 1;
                prompt = args.get(i).cloned();
            }
            "--threads" => {
                i += 1;
                let raw = args
                    .get(i)
                    .ok_or_else(|| "logits: missing value for --threads".to_owned())?;
                let value = raw
                    .parse::<usize>()
                    .map_err(|_| format!("logits: invalid --threads value: {raw}"))?;
                if value == 0 {
                    return Err("logits: --threads must be greater than zero".to_owned());
                }
                threads = Some(value);
            }
            "--chunk-size" => {
                i += 1;
                let raw = args
                    .get(i)
                    .ok_or_else(|| "logits: missing value for --chunk-size".to_owned())?;
                chunk_size = raw
                    .parse::<usize>()
                    .map_err(|_| format!("logits: invalid --chunk-size value: {raw}"))?;
                if chunk_size == 0 {
                    return Err("logits: --chunk-size must be greater than zero".to_owned());
                }
            }
            other => return Err(format!("unknown logits argument: {other}")),
        }
        i += 1;
    }

    if !hash {
        return Err(
            "usage: detllm logits -m model.gguf --tokens 1,2,3 --hash [--dump FILE] [--threads T] [--chunk-size N]".to_owned(),
        );
    }
    let model_path = model_path.ok_or(
        "usage: detllm logits -m model.gguf (--tokens 1,2,3|-p prompt) --hash [--dump FILE] [--threads T] [--chunk-size N]",
    )?;
    if tokens.is_some() == prompt.is_some() {
        return Err("provide exactly one of --tokens or -p/--prompt".to_owned());
    }
    apply_threads(threads)?;

    let bytes = fs::read(model_path).map_err(|e| e.to_string())?;
    let gguf = det_gguf::parse(&bytes).map_err(|e| format!("GGUF parse error: {e:?}"))?;
    let mut tokenizer_vocab_len = None;
    let token_ids = if let Some(raw) = tokens {
        parse_tokens(&raw)?
    } else {
        tokenizer_vocab_len = Some(gguf_token_vocab_len(&gguf)?);
        let tokenizer =
            det_token::Tokenizer::from_gguf(&gguf).map_err(|e| format!("tokenizer error: {e}"))?;
        tokenizer
            .tokenize_bytes(prompt.as_deref().unwrap_or_default().as_bytes())
            .map_err(|e| format!("tokenize error: {e:?}"))?
            .into_iter()
            .map(|token| token as usize)
            .collect()
    };
    let model = det_model::F32Llama::from_gguf(&gguf, &bytes)
        .map_err(|e| format!("F32 model load error: {e:?}"))?;
    if let Some(tokenizer_vocab_len) = tokenizer_vocab_len {
        validate_tokenizer_model_vocab_len(tokenizer_vocab_len, &model)?;
    }
    let logits_bytes = if dump_path.is_some() {
        Some(
            model
                .logits_bytes_for_tokens_chunked(&token_ids, chunk_size)
                .map_err(|e| format!("logits error: {e:?}"))?,
        )
    } else {
        None
    };
    let digest = if let Some(logits_bytes) = logits_bytes.as_ref() {
        let mut h = Sha256::new();
        h.update(logits_bytes);
        h.finalize()
    } else {
        model
            .logits_hash_for_tokens_chunked(&token_ids, chunk_size)
            .map_err(|e| format!("logits error: {e:?}"))?
    };
    if let Some(path) = dump_path {
        fs::write(path, logits_bytes.as_ref().expect("dump bytes")).map_err(|e| e.to_string())?;
    }
    println!("{}", hex(&digest));
    Ok(())
}

fn apply_threads(threads: Option<usize>) -> Result<(), String> {
    det_model::set_thread_count(threads).map_err(|e| format!("thread configuration error: {e:?}"))
}

fn parse_tokens(s: &str) -> Result<Vec<usize>, String> {
    if s.trim().is_empty() {
        return Err("token list must not be empty".to_owned());
    }
    let mut out = Vec::new();
    for part in s.split(',') {
        let token = part
            .trim()
            .parse::<usize>()
            .map_err(|_| format!("invalid token id: {part}"))?;
        out.push(token);
    }
    Ok(out)
}

fn join_tokens(tokens: &[u32]) -> String {
    let mut out = String::new();
    for (idx, token) in tokens.iter().enumerate() {
        if idx != 0 {
            out.push(',');
        }
        out.push_str(&token.to_string());
    }
    out
}

fn hex(bytes: &[u8]) -> String {
    const HEX: &[u8; 16] = b"0123456789abcdef";
    let mut out = String::with_capacity(bytes.len() * 2);
    for &b in bytes {
        out.push(HEX[(b >> 4) as usize] as char);
        out.push(HEX[(b & 0x0f) as usize] as char);
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::sync::atomic::{AtomicU64, Ordering};

    #[test]
    fn runtime_canary_matches_current_cross_crate_kernels() {
        assert_eq!(
            runtime_canary_hash().expect("runtime canary hash"),
            RUNTIME_CANARY_EXPECTED
        );
        run_canaries().expect("all canaries");
    }

    #[test]
    fn model_backed_token_codec_round_trips() {
        let model = fixture_model().expect("fixture model");
        let tokens = [0usize, 1, 2, 3, 0, 2, 1, 3, 2, 0];
        let encoded = encode_tokens_with_model(&model, &tokens, 3, 1).expect("encode");
        let decoded =
            decode_tokens_with_model(&model, &encoded, tokens.len(), 3, 1).expect("decode");
        assert_eq!(decoded, tokens);
    }

    #[test]
    fn streaming_codec_matches_replay_cdf_payload() {
        let model = fixture_model().expect("fixture model");
        let tokens = [0usize, 1, 2, 3, 0, 2, 1, 3, 2, 0];
        let encoded = encode_tokens_with_model(&model, &tokens, 3, 1).expect("encode");
        let reference =
            encode_tokens_with_replayed_context(&model, &tokens, 3, 1).expect("reference encode");
        assert_eq!(encoded, reference);
    }

    #[test]
    fn model_backed_token_codec_rejects_invalid_windows() {
        let model = fixture_model().expect("fixture model");
        let tokens = [0usize, 1];
        let valid = encode_tokens_with_model(&model, &tokens, 3, 1).expect("encode");

        for (n_ctx, overlap, expected) in [
            (0, 0, "n_ctx must be greater than zero"),
            (3, 3, "overlap 3 must be smaller than n_ctx 3"),
            (
                model.config.context_length + 1,
                0,
                "exceeds model context length",
            ),
        ] {
            let encode_err = encode_tokens_with_model(&model, &tokens, n_ctx, overlap)
                .expect_err("invalid encode window");
            assert!(encode_err.contains(expected), "{encode_err}");

            let decode_err = decode_tokens_with_model(&model, &valid, tokens.len(), n_ctx, overlap)
                .expect_err("invalid decode window");
            assert!(decode_err.contains(expected), "{decode_err}");
        }
    }

    #[test]
    fn window_rolls_over_when_context_reaches_n_ctx() {
        let n_ctx = 8;
        let overlap = 2;
        let mut start = 0usize;
        let mut observed = Vec::new();
        for pos in 0..18 {
            start = next_window_start(pos, start, n_ctx, overlap);
            observed.push((pos, start, pos - start));
        }

        assert_eq!(observed[7], (7, 0, 7));
        assert_eq!(observed[8], (8, 6, 2));
        assert_eq!(observed[13], (13, 6, 7));
        assert_eq!(observed[14], (14, 12, 2));
    }

    #[test]
    fn cli_compress_decompress_round_trips_synthetic_gguf_file() {
        let dir = unique_tmp_dir();
        fs::create_dir_all(&dir).expect("mkdir");
        let model_path = dir.join("model.gguf");
        let input_path = dir.join("input.bin");
        let compressed_path = dir.join("out.dtlz");
        let restored_path = dir.join("restored.bin");

        fs::write(&model_path, synthetic_gguf_bytes()).expect("write model");
        let input = b"detllm windowed roundtrip";
        fs::write(&input_path, input).expect("write input");

        compress(vec![
            "-m".to_owned(),
            model_path.to_string_lossy().into_owned(),
            "-i".to_owned(),
            input_path.to_string_lossy().into_owned(),
            "-o".to_owned(),
            compressed_path.to_string_lossy().into_owned(),
            "--n-ctx".to_owned(),
            "8".to_owned(),
        ])
        .expect("compress");
        decompress(vec![
            "-m".to_owned(),
            model_path.to_string_lossy().into_owned(),
            "-i".to_owned(),
            compressed_path.to_string_lossy().into_owned(),
            "-o".to_owned(),
            restored_path.to_string_lossy().into_owned(),
        ])
        .expect("decompress");

        assert_eq!(fs::read(restored_path).expect("restored"), input);
        let encoded = fs::read(compressed_path).expect("compressed");
        let header = det_coder::DtlzHeader::decode(&encoded).expect("header");
        assert_eq!(header.flags, det_coder::FLAG_BYTE_ESCAPES);
        assert_eq!(header.n_ctx, 8);
        assert_eq!(header.overlap, 2);
        assert_eq!(header.orig_len, input.len() as u64);

        let _ = fs::remove_dir_all(dir);
    }

    fn encode_tokens_with_replayed_context(
        model: &det_model::F32Llama,
        tokens: &[usize],
        n_ctx: usize,
        overlap: usize,
    ) -> Result<Vec<u8>, String> {
        validate_window(n_ctx, overlap, model.config.context_length)?;
        let mut enc = det_coder::RangeEncoder::new();
        let mut window_start = 0usize;
        for pos in 0..tokens.len() {
            window_start = next_window_start(pos, window_start, n_ctx, overlap);
            let cdf = cdf_for_context(model, &tokens[window_start..pos], n_ctx)?;
            encode_symbol(&mut enc, &cdf, tokens[pos])?;
        }
        Ok(enc.finish())
    }

    #[test]
    fn cli_compress_rejects_n_ctx_above_model_context() {
        let dir = unique_tmp_dir();
        fs::create_dir_all(&dir).expect("mkdir");
        let model_path = dir.join("model.gguf");
        let input_path = dir.join("input.bin");
        let compressed_path = dir.join("out.dtlz");

        fs::write(&model_path, synthetic_gguf_bytes()).expect("write model");
        fs::write(&input_path, b"context bound").expect("write input");

        let err = compress(vec![
            "-m".to_owned(),
            model_path.to_string_lossy().into_owned(),
            "-i".to_owned(),
            input_path.to_string_lossy().into_owned(),
            "-o".to_owned(),
            compressed_path.to_string_lossy().into_owned(),
            "--n-ctx".to_owned(),
            "17".to_owned(),
        ])
        .expect_err("oversized n_ctx should be rejected");

        assert!(
            err.contains("n_ctx 17 exceeds model context length 16"),
            "{err}"
        );
        assert!(!compressed_path.exists());

        let _ = fs::remove_dir_all(dir);
    }

    #[test]
    fn cli_decompress_rejects_n_ctx_override() {
        let dir = unique_tmp_dir();
        fs::create_dir_all(&dir).expect("mkdir");
        let model_path = dir.join("model.gguf");
        let compressed_path = dir.join("out.dtlz");
        let restored_path = dir.join("restored.bin");

        fs::write(&model_path, synthetic_gguf_bytes()).expect("write model");

        let err = decompress(vec![
            "-m".to_owned(),
            model_path.to_string_lossy().into_owned(),
            "-i".to_owned(),
            compressed_path.to_string_lossy().into_owned(),
            "-o".to_owned(),
            restored_path.to_string_lossy().into_owned(),
            "--n-ctx".to_owned(),
            "8".to_owned(),
        ])
        .expect_err("decompress n_ctx override should be rejected");

        assert!(
            err.contains("--n-ctx is stored in the DTLZ header"),
            "{err}"
        );
        assert!(!restored_path.exists());

        let _ = fs::remove_dir_all(dir);
    }

    #[test]
    fn cli_decompress_rejects_malformed_header_before_model_load() {
        let dir = unique_tmp_dir();
        fs::create_dir_all(&dir).expect("mkdir");
        let missing_model_path = dir.join("missing-model.gguf");
        let compressed_path = dir.join("bad.dtlz");
        let restored_path = dir.join("restored.bin");

        let mut encoded = [0u8; det_coder::file::HEADER_LEN];
        encoded[0..4].copy_from_slice(b"NOPE");
        fs::write(&compressed_path, encoded).expect("write malformed header");

        let err = decompress(vec![
            "-m".to_owned(),
            missing_model_path.to_string_lossy().into_owned(),
            "-i".to_owned(),
            compressed_path.to_string_lossy().into_owned(),
            "-o".to_owned(),
            restored_path.to_string_lossy().into_owned(),
        ])
        .expect_err("malformed header should be rejected before model load");

        assert!(err.contains("DTLZ header error: BadMagic"), "{err}");
        assert!(!restored_path.exists());

        let _ = fs::remove_dir_all(dir);
    }

    #[test]
    fn loaded_model_rejects_tokenizer_model_vocab_mismatch() {
        let dir = unique_tmp_dir();
        fs::create_dir_all(&dir).expect("mkdir");
        let model_path = dir.join("model.gguf");

        fs::write(&model_path, synthetic_gguf_bytes_with_vocab(257, false)).expect("write model");

        let err = match LoadedModel::load(model_path.to_str().expect("model path")) {
            Ok(_) => panic!("vocab mismatch should be rejected"),
            Err(err) => err,
        };
        assert!(
            err.contains("tokenizer vocabulary length 256 does not match model vocabulary 257"),
            "{err}"
        );

        let _ = fs::remove_dir_all(dir);
    }

    #[test]
    fn codec_uses_byte_escape_for_partial_bpe_missing_bytes() {
        let dir = unique_tmp_dir();
        fs::create_dir_all(&dir).expect("mkdir");
        let model_path = dir.join("model.gguf");
        let input_path = dir.join("input.bin");
        let compressed_path = dir.join("out.dtlz");
        let restored_path = dir.join("restored.bin");

        fs::write(
            &model_path,
            synthetic_bpe_gguf_bytes_missing_byte_fallback(),
        )
        .expect("write model");
        let input = b"ab\xffba";
        fs::write(&input_path, input).expect("write input");

        let loaded = LoadedModel::load(model_path.to_str().expect("model path")).expect("load");
        let symbols = loaded
            .tokenizer
            .codec_symbols(input, loaded.model.output.rows())
            .expect("symbols");
        assert!(symbols.contains(&(loaded.model.output.rows() + 0xff)));

        compress(vec![
            "-m".to_owned(),
            model_path.to_string_lossy().into_owned(),
            "-i".to_owned(),
            input_path.to_string_lossy().into_owned(),
            "-o".to_owned(),
            compressed_path.to_string_lossy().into_owned(),
            "--n-ctx".to_owned(),
            "8".to_owned(),
        ])
        .expect("compress");
        decompress(vec![
            "-m".to_owned(),
            model_path.to_string_lossy().into_owned(),
            "-i".to_owned(),
            compressed_path.to_string_lossy().into_owned(),
            "-o".to_owned(),
            restored_path.to_string_lossy().into_owned(),
        ])
        .expect("decompress");

        assert_eq!(fs::read(restored_path).expect("restored"), input);

        let _ = fs::remove_dir_all(dir);
    }

    #[test]
    fn tokenize_allows_partial_bpe_when_input_bytes_are_present() {
        let dir = unique_tmp_dir();
        fs::create_dir_all(&dir).expect("mkdir");
        let model_path = dir.join("model.gguf");

        fs::write(
            &model_path,
            synthetic_bpe_gguf_bytes_missing_byte_fallback(),
        )
        .expect("write model");

        tokenize(vec![
            "-m".to_owned(),
            model_path.to_string_lossy().into_owned(),
            "-p".to_owned(),
            "ab".to_owned(),
        ])
        .expect("tokenize");

        let _ = fs::remove_dir_all(dir);
    }

    #[test]
    fn codec_vocab_limit_matches_cdf_design_limit() {
        validate_codec_vocab_len(det_coder::MAX_SYMBOLS - det_coder::BYTE_ESCAPE_SYMBOLS)
            .expect("design maximum is accepted");
        let err =
            validate_codec_vocab_len(det_coder::MAX_SYMBOLS - det_coder::BYTE_ESCAPE_SYMBOLS + 1)
                .expect_err("larger vocabularies should be rejected for codec use");
        assert!(
            err.contains(
                "model vocabulary 261889 plus 256 byte escapes exceeds codec symbol limit 262144"
            ),
            "{err}"
        );
    }

    #[test]
    fn logits_prompt_rejects_tokenizer_model_vocab_mismatch() {
        let dir = unique_tmp_dir();
        fs::create_dir_all(&dir).expect("mkdir");
        let model_path = dir.join("model.gguf");

        fs::write(&model_path, synthetic_gguf_bytes_with_vocab(257, false)).expect("write model");

        let err = logits(vec![
            "-m".to_owned(),
            model_path.to_string_lossy().into_owned(),
            "-p".to_owned(),
            "a".to_owned(),
            "--hash".to_owned(),
        ])
        .expect_err("prompt logits should reject vocab mismatch");
        assert!(
            err.contains("tokenizer vocabulary length 256 does not match model vocabulary 257"),
            "{err}"
        );

        let _ = fs::remove_dir_all(dir);
    }

    #[test]
    fn tokenize_rejects_tokenizer_model_vocab_mismatch() {
        let dir = unique_tmp_dir();
        fs::create_dir_all(&dir).expect("mkdir");
        let model_path = dir.join("model.gguf");

        fs::write(&model_path, synthetic_gguf_bytes_with_vocab(257, false)).expect("write model");

        let err = tokenize(vec![
            "-m".to_owned(),
            model_path.to_string_lossy().into_owned(),
            "-p".to_owned(),
            "a".to_owned(),
        ])
        .expect_err("tokenize should reject vocab mismatch");
        assert!(
            err.contains("tokenizer vocabulary length 256 does not match model vocabulary 257"),
            "{err}"
        );

        let _ = fs::remove_dir_all(dir);
    }

    #[test]
    fn cli_decompress_stops_after_original_byte_length_for_bpe_tokens() {
        let dir = unique_tmp_dir();
        fs::create_dir_all(&dir).expect("mkdir");
        let model_path = dir.join("model.gguf");
        let input_path = dir.join("input.bin");
        let compressed_path = dir.join("out.dtlz");
        let restored_path = dir.join("restored.bin");

        fs::write(&model_path, synthetic_bpe_gguf_bytes()).expect("write model");
        let input = b"abab";
        fs::write(&input_path, input).expect("write input");

        compress(vec![
            "-m".to_owned(),
            model_path.to_string_lossy().into_owned(),
            "-i".to_owned(),
            input_path.to_string_lossy().into_owned(),
            "-o".to_owned(),
            compressed_path.to_string_lossy().into_owned(),
            "--n-ctx".to_owned(),
            "8".to_owned(),
        ])
        .expect("compress");
        decompress(vec![
            "-m".to_owned(),
            model_path.to_string_lossy().into_owned(),
            "-i".to_owned(),
            compressed_path.to_string_lossy().into_owned(),
            "-o".to_owned(),
            restored_path.to_string_lossy().into_owned(),
        ])
        .expect("decompress");

        assert_eq!(fs::read(restored_path).expect("restored"), input);
        let encoded = fs::read(compressed_path).expect("compressed");
        let header = det_coder::DtlzHeader::decode(&encoded).expect("header");
        assert_eq!(header.flags, det_coder::FLAG_BYTE_ESCAPES);
        assert_eq!(header.orig_len, 4);

        let loaded = LoadedModel::load(model_path.to_str().expect("model path")).expect("load");
        let token_ids = loaded.tokenizer.tokenize_bytes(input).expect("tokenize");
        assert_eq!(token_ids, [256, 256]);

        let _ = fs::remove_dir_all(dir);
    }

    #[test]
    fn cli_decompress_truncates_final_multibyte_token_to_orig_len() {
        let dir = unique_tmp_dir();
        fs::create_dir_all(&dir).expect("mkdir");
        let model_path = dir.join("model.gguf");
        let compressed_path = dir.join("out.dtlz");
        let restored_path = dir.join("restored.bin");

        fs::write(&model_path, synthetic_bpe_gguf_bytes()).expect("write model");
        let loaded = LoadedModel::load(model_path.to_str().expect("model path")).expect("load");
        assert_eq!(
            loaded.tokenizer.tokenize_bytes(b"ab").expect("tokenize"),
            [256]
        );
        assert_eq!(
            loaded.tokenizer.detokenize_bytes(&[256]).expect("detok"),
            b"ab"
        );

        let payload =
            encode_symbols_with_model(&loaded.model, &[256], 8, 2, false).expect("encode");
        let header = det_coder::DtlzHeader {
            flags: 0,
            model_sha256: loaded.model_sha256,
            n_ctx: 8,
            overlap: 2,
            orig_len: 1,
        };
        let mut encoded = Vec::new();
        encoded.extend_from_slice(&header.encode_checked().expect("header"));
        encoded.extend_from_slice(&payload);
        fs::write(&compressed_path, encoded).expect("write compressed");

        decompress(vec![
            "-m".to_owned(),
            model_path.to_string_lossy().into_owned(),
            "-i".to_owned(),
            compressed_path.to_string_lossy().into_owned(),
            "-o".to_owned(),
            restored_path.to_string_lossy().into_owned(),
        ])
        .expect("decompress");

        assert_eq!(fs::read(restored_path).expect("restored"), b"a");

        let _ = fs::remove_dir_all(dir);
    }

    #[test]
    fn cli_round_trips_empty_and_binary_multi_window_inputs() {
        let dir = unique_tmp_dir();
        fs::create_dir_all(&dir).expect("mkdir");
        let model_path = dir.join("model.gguf");
        fs::write(&model_path, synthetic_gguf_bytes()).expect("write model");

        let cases: Vec<(&str, Vec<u8>)> = vec![
            ("empty", Vec::new()),
            (
                "binary",
                (0..=255u8).chain((0..=255u8).rev()).collect::<Vec<_>>(),
            ),
        ];

        for (name, input) in cases {
            let input_path = dir.join(format!("{name}.bin"));
            let compressed_path = dir.join(format!("{name}.dtlz"));
            let restored_path = dir.join(format!("{name}.restored"));
            fs::write(&input_path, &input).expect("write input");

            compress(vec![
                "-m".to_owned(),
                model_path.to_string_lossy().into_owned(),
                "-i".to_owned(),
                input_path.to_string_lossy().into_owned(),
                "-o".to_owned(),
                compressed_path.to_string_lossy().into_owned(),
                "--n-ctx".to_owned(),
                "8".to_owned(),
            ])
            .expect("compress");
            decompress(vec![
                "-m".to_owned(),
                model_path.to_string_lossy().into_owned(),
                "-i".to_owned(),
                compressed_path.to_string_lossy().into_owned(),
                "-o".to_owned(),
                restored_path.to_string_lossy().into_owned(),
            ])
            .expect("decompress");

            assert_eq!(fs::read(restored_path).expect("restored"), input);
            let encoded = fs::read(compressed_path).expect("compressed");
            let header = det_coder::DtlzHeader::decode(&encoded).expect("header");
            assert_eq!(header.flags, det_coder::FLAG_BYTE_ESCAPES);
            assert_eq!(header.n_ctx, 8);
            assert_eq!(header.overlap, 2);
            assert_eq!(header.orig_len, input.len() as u64);
        }

        let _ = fs::remove_dir_all(dir);
    }

    #[test]
    fn cli_decompress_rejects_model_hash_mismatch() {
        let dir = unique_tmp_dir();
        fs::create_dir_all(&dir).expect("mkdir");
        let model_path = dir.join("model.gguf");
        let input_path = dir.join("input.bin");
        let compressed_path = dir.join("out.dtlz");
        let restored_path = dir.join("restored.bin");

        fs::write(&model_path, synthetic_gguf_bytes()).expect("write model");
        fs::write(&input_path, b"hash mismatch").expect("write input");

        compress(vec![
            "-m".to_owned(),
            model_path.to_string_lossy().into_owned(),
            "-i".to_owned(),
            input_path.to_string_lossy().into_owned(),
            "-o".to_owned(),
            compressed_path.to_string_lossy().into_owned(),
            "--n-ctx".to_owned(),
            "8".to_owned(),
        ])
        .expect("compress");

        let mut encoded = fs::read(&compressed_path).expect("compressed");
        encoded[8] ^= 0x80;
        fs::write(&compressed_path, encoded).expect("corrupt header hash");

        let err = decompress(vec![
            "-m".to_owned(),
            model_path.to_string_lossy().into_owned(),
            "-i".to_owned(),
            compressed_path.to_string_lossy().into_owned(),
            "-o".to_owned(),
            restored_path.to_string_lossy().into_owned(),
        ])
        .expect_err("hash mismatch");
        assert!(err.contains("model SHA-256 does not match"));
        assert!(!restored_path.exists());

        let _ = fs::remove_dir_all(dir);
    }

    #[test]
    fn cli_decompress_rejects_inflated_orig_len_without_output() {
        let dir = unique_tmp_dir();
        fs::create_dir_all(&dir).expect("mkdir");
        let model_path = dir.join("model.gguf");
        let input_path = dir.join("input.bin");
        let compressed_path = dir.join("out.dtlz");
        let restored_path = dir.join("restored.bin");

        fs::write(&model_path, synthetic_gguf_bytes()).expect("write model");
        fs::write(&input_path, b"short").expect("write input");

        compress(vec![
            "-m".to_owned(),
            model_path.to_string_lossy().into_owned(),
            "-i".to_owned(),
            input_path.to_string_lossy().into_owned(),
            "-o".to_owned(),
            compressed_path.to_string_lossy().into_owned(),
            "--n-ctx".to_owned(),
            "8".to_owned(),
        ])
        .expect("compress");

        let mut encoded = fs::read(&compressed_path).expect("compressed");
        encoded[48..56].copy_from_slice(&u64::MAX.to_le_bytes());
        fs::write(&compressed_path, encoded).expect("inflate orig_len");

        let err = decompress(vec![
            "-m".to_owned(),
            model_path.to_string_lossy().into_owned(),
            "-i".to_owned(),
            compressed_path.to_string_lossy().into_owned(),
            "-o".to_owned(),
            restored_path.to_string_lossy().into_owned(),
        ])
        .expect_err("inflated orig_len should fail");

        assert!(
            err.contains("orig_len does not fit usize") || err.contains("range "),
            "{err}"
        );
        assert!(!restored_path.exists());

        let _ = fs::remove_dir_all(dir);
    }

    fn unique_tmp_dir() -> std::path::PathBuf {
        static NEXT_TMP_ID: AtomicU64 = AtomicU64::new(0);
        let mut dir = std::env::temp_dir();
        dir.push(format!(
            "detllm-test-{}-{}-{}",
            std::process::id(),
            NEXT_TMP_ID.fetch_add(1, Ordering::Relaxed),
            std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .expect("time")
                .as_nanos()
        ));
        dir
    }

    fn synthetic_gguf_bytes() -> Vec<u8> {
        synthetic_gguf_bytes_with_vocab(256, false)
    }

    fn synthetic_bpe_gguf_bytes() -> Vec<u8> {
        synthetic_gguf_bytes_with_vocab(257, true)
    }

    fn synthetic_bpe_gguf_bytes_missing_byte_fallback() -> Vec<u8> {
        synthetic_gguf_bytes_with_vocab_and_token_override(257, true, Some((255, "not-a-byte")))
    }

    fn synthetic_gguf_bytes_with_vocab(vocab_size: usize, include_bpe: bool) -> Vec<u8> {
        synthetic_gguf_bytes_with_vocab_and_token_override(vocab_size, include_bpe, None)
    }

    fn synthetic_gguf_bytes_with_vocab_and_token_override(
        vocab_size: usize,
        include_bpe: bool,
        token_override: Option<(u32, &str)>,
    ) -> Vec<u8> {
        let mut tensors = Vec::new();
        let mut data = Vec::new();
        push_tensor(
            &mut tensors,
            &mut data,
            "token_embd.weight",
            vocab_size,
            4,
            0.001,
        );
        push_vector(&mut tensors, &mut data, "blk.0.attn_norm.weight", 4, 1.0);
        push_tensor(&mut tensors, &mut data, "blk.0.attn_q.weight", 4, 4, 0.002);
        push_tensor(
            &mut tensors,
            &mut data,
            "blk.0.attn_k.weight",
            2,
            4,
            -0.0015,
        );
        push_tensor(&mut tensors, &mut data, "blk.0.attn_v.weight", 2, 4, 0.0025);
        push_tensor(
            &mut tensors,
            &mut data,
            "blk.0.attn_output.weight",
            4,
            4,
            -0.002,
        );
        push_vector(&mut tensors, &mut data, "blk.0.ffn_norm.weight", 4, 1.0);
        push_tensor(
            &mut tensors,
            &mut data,
            "blk.0.ffn_gate.weight",
            6,
            4,
            0.003,
        );
        push_tensor(
            &mut tensors,
            &mut data,
            "blk.0.ffn_up.weight",
            6,
            4,
            -0.0025,
        );
        push_tensor(
            &mut tensors,
            &mut data,
            "blk.0.ffn_down.weight",
            4,
            6,
            0.0018,
        );
        push_vector(&mut tensors, &mut data, "output_norm.weight", 4, 1.0);
        push_tensor(
            &mut tensors,
            &mut data,
            "output.weight",
            vocab_size,
            4,
            0.0022,
        );

        let mut out = Vec::new();
        out.extend_from_slice(b"GGUF");
        push_u32(&mut out, 3);
        push_u64(&mut out, tensors.len() as u64);
        push_u64(&mut out, if include_bpe { 14 } else { 12 });
        push_meta_string(&mut out, "general.architecture", "llama");
        push_meta_u32(&mut out, "llama.block_count", 1);
        push_meta_u32(&mut out, "llama.embedding_length", 4);
        push_meta_u32(&mut out, "llama.feed_forward_length", 6);
        push_meta_u32(&mut out, "llama.attention.head_count", 2);
        push_meta_u32(&mut out, "llama.attention.head_count_kv", 1);
        push_meta_f32(&mut out, "llama.attention.layer_norm_rms_epsilon", 1e-5);
        push_meta_f32(&mut out, "llama.rope.freq_base", 10_000.0);
        push_meta_u32(&mut out, "llama.rope.dimension_count", 2);
        push_meta_u32(&mut out, "llama.context_length", 16);
        push_meta_u32(&mut out, "llama.vocab_size", vocab_size as u32);
        push_meta_token_array(&mut out, include_bpe, token_override);
        if include_bpe {
            push_meta_string(&mut out, "tokenizer.ggml.model", "gpt2");
            push_meta_merges_array(&mut out);
        }

        for tensor in &tensors {
            push_string(&mut out, &tensor.name);
            push_u32(&mut out, tensor.dims.len() as u32);
            for &dim in &tensor.dims {
                push_u64(&mut out, dim);
            }
            push_u32(&mut out, 0);
            push_u64(&mut out, tensor.offset);
        }
        while out.len() % 32 != 0 {
            out.push(0);
        }
        out.extend_from_slice(&data);
        out
    }

    struct TensorSpec {
        name: String,
        dims: Vec<u64>,
        offset: u64,
    }

    fn push_tensor(
        tensors: &mut Vec<TensorSpec>,
        data: &mut Vec<u8>,
        name: &str,
        rows: usize,
        cols: usize,
        scale: f32,
    ) {
        let offset = data.len() as u64;
        tensors.push(TensorSpec {
            name: name.to_owned(),
            dims: vec![cols as u64, rows as u64],
            offset,
        });
        for i in 0..rows * cols {
            let value = (((i % 13) as f32) - 6.0) * scale;
            data.extend_from_slice(&value.to_le_bytes());
        }
    }

    fn push_vector(
        tensors: &mut Vec<TensorSpec>,
        data: &mut Vec<u8>,
        name: &str,
        len: usize,
        value: f32,
    ) {
        let offset = data.len() as u64;
        tensors.push(TensorSpec {
            name: name.to_owned(),
            dims: vec![len as u64],
            offset,
        });
        for _ in 0..len {
            data.extend_from_slice(&value.to_le_bytes());
        }
    }

    fn push_meta_string(out: &mut Vec<u8>, key: &str, value: &str) {
        push_string(out, key);
        push_u32(out, 8);
        push_string(out, value);
    }

    fn push_meta_u32(out: &mut Vec<u8>, key: &str, value: u32) {
        push_string(out, key);
        push_u32(out, 4);
        push_u32(out, value);
    }

    fn push_meta_f32(out: &mut Vec<u8>, key: &str, value: f32) {
        push_string(out, key);
        push_u32(out, 6);
        out.extend_from_slice(&value.to_bits().to_le_bytes());
    }

    fn push_meta_token_array(
        out: &mut Vec<u8>,
        include_bpe: bool,
        token_override: Option<(u32, &str)>,
    ) {
        push_string(out, "tokenizer.ggml.tokens");
        push_u32(out, 9);
        push_u32(out, 8);
        push_u64(out, if include_bpe { 257 } else { 256 });
        for b in 0..=255u32 {
            if token_override.is_some_and(|(idx, _)| idx == b) {
                push_string(out, token_override.expect("override").1);
            } else {
                push_string(out, &format!("<0x{b:02X}>"));
            }
        }
        if include_bpe {
            push_string(out, "ab");
        }
    }

    fn push_meta_merges_array(out: &mut Vec<u8>) {
        push_string(out, "tokenizer.ggml.merges");
        push_u32(out, 9);
        push_u32(out, 8);
        push_u64(out, 1);
        push_string(out, "<0x61> <0x62>");
    }

    fn push_string(out: &mut Vec<u8>, s: &str) {
        push_u64(out, s.len() as u64);
        out.extend_from_slice(s.as_bytes());
    }

    fn push_u32(out: &mut Vec<u8>, value: u32) {
        out.extend_from_slice(&value.to_le_bytes());
    }

    fn push_u64(out: &mut Vec<u8>, value: u64) {
        out.extend_from_slice(&value.to_le_bytes());
    }
}
