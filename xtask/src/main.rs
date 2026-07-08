use std::{
    collections::BTreeMap,
    env,
    fmt::Write as _,
    fs,
    path::{Path, PathBuf},
    time::Instant,
};

const F32_MODEL_PATH: &str = "testdata/tiny-f32.gguf";
const F32_HASH_PATH: &str = "testdata/tiny-f32.logits.sha256";
const QMIX_MODEL_PATH: &str = "testdata/tiny-qmix.gguf";
const QMIX_HASH_PATH: &str = "testdata/tiny-qmix.logits.sha256";
const TOKENS_PATH: &str = "testdata/tiny.tokens.txt";
const TOKENS: &[usize] = &[0, 1, 2, 3, 0, 2];
const DETERMINISM_CHECK_ROOTS: &[&str] =
    &["crates", "xtask/src", ".github/workflows", "Cargo.toml"];
const DETERMINISM_BANNED_PATTERNS: &[(&str, &str)] = &[
    ("f32::exp", "use det_num::exp_f32 instead of platform libm"), // determinism-allow
    ("f32::sin", "use vendored deterministic libm routines"),      // determinism-allow
    ("f32::cos", "use vendored deterministic libm routines"),      // determinism-allow
    ("f32::ln", "use vendored deterministic libm routines"),       // determinism-allow
    (
        "f32::powf",
        "platform transcendental functions are forbidden",
    ), // determinism-allow
    (
        "f32::tanh",
        "platform transcendental functions are forbidden",
    ), // determinism-allow
    ("f64::exp", "use vendored deterministic libm routines"),      // determinism-allow
    ("f64::sin", "use vendored deterministic libm routines"),      // determinism-allow
    ("f64::cos", "use vendored deterministic libm routines"),      // determinism-allow
    ("f64::ln", "use vendored deterministic libm routines"),       // determinism-allow
    (
        "f64::powf",
        "platform transcendental functions are forbidden",
    ), // determinism-allow
    (
        "f64::tanh",
        "platform transcendental functions are forbidden",
    ), // determinism-allow
    ("mul_add", "FMA changes the specified rounding sequence"),    // determinism-allow
    (
        "HashMap",
        "use BTreeMap/Vec when iteration order can matter",
    ), // determinism-allow
    (
        "HashSet",
        "use BTreeSet/Vec when iteration order can matter",
    ), // determinism-allow
    ("relaxed-simd", "Wasm relaxed SIMD is nondeterministic"),     // determinism-allow
    (
        "target-feature=+relaxed", // determinism-allow
        "relaxed target features are not allowed",
    ),
    (
        ".par_iter().sum", // determinism-allow
        "parallel floating-point reductions are forbidden",
    ),
    (
        ".par_chunks().sum", // determinism-allow
        "parallel floating-point reductions are forbidden",
    ),
];

#[derive(Clone, Copy)]
struct ModelSpec {
    vocab_size: usize,
    embedding_length: u32,
    feed_forward_length: u32,
    head_count: u32,
    head_count_kv: u32,
    context_length: u32,
}

fn main() {
    if let Err(e) = real_main() {
        eprintln!("xtask: {e}");
        std::process::exit(1);
    }
}

fn real_main() -> Result<(), String> {
    let mut args = env::args().skip(1);
    match args.next().as_deref() {
        Some("generate-testdata") => {
            let check = match args.next().as_deref() {
                Some("--check") => true,
                None => false,
                Some(other) => return Err(format!("unknown generate-testdata argument: {other}")),
            };
            generate_testdata(check)
        }
        Some("bench-testdata") => {
            let iters = match args.next().as_deref() {
                Some("--iters") => args
                    .next()
                    .ok_or("bench-testdata: missing value for --iters")?
                    .parse::<usize>()
                    .map_err(|e| format!("bench-testdata: invalid --iters value: {e}"))?,
                None => 100,
                Some(other) => return Err(format!("unknown bench-testdata argument: {other}")),
            };
            bench_testdata(iters)
        }
        Some("model-info") => model_info(parse_model_info_opts(args.collect())?),
        Some("bench-file") => bench_file(parse_bench_file_opts(args.collect())?),
        Some("compare-logits") => compare_logits(parse_compare_logits_opts(args.collect())?),
        Some("verify-logits-hashes") => {
            verify_logits_hashes(parse_verify_logits_hashes_opts(args.collect())?)
        }
        Some("check-ci-workflow") => check_ci_workflow(),
        Some("check-determinism") => check_determinism(),
        _ => Err(
            "usage: cargo run -p xtask -- <generate-testdata [--check]|bench-testdata [--iters N]|model-info --model model.gguf|bench-file --model model.gguf --input file [--limit-bytes N] [--n-ctx N] [--iters N]|compare-logits --actual det.bin --reference ref.bin [--min-cosine X] [--row-size N] [--rows N]|verify-logits-hashes --dir DIR --expected-count N|check-ci-workflow|check-determinism>"
                .to_owned(),
        ),
    }
}

fn check_determinism() -> Result<(), String> {
    let mut files = Vec::new();
    for root in DETERMINISM_CHECK_ROOTS {
        collect_policy_files(Path::new(root), &mut files)?;
    }
    files.sort();

    let mut violations = Vec::new();
    for path in files {
        let text = fs::read_to_string(&path).map_err(|e| format!("{}: {e}", path.display()))?;
        scan_determinism_text(&path, &text, &mut violations);
        if path.file_name().and_then(|name| name.to_str()) == Some("Cargo.toml") {
            scan_dependency_policy_text(&path, &text, &mut violations);
        }
    }

    if violations.is_empty() {
        println!("determinism policy check passed");
        Ok(())
    } else {
        Err(format!(
            "determinism policy violations:\n{}",
            violations.join("\n")
        ))
    }
}

fn collect_policy_files(path: &Path, files: &mut Vec<PathBuf>) -> Result<(), String> {
    let metadata = fs::metadata(path).map_err(|e| format!("{}: {e}", path.display()))?;
    if metadata.is_file() {
        if is_policy_file(path) {
            files.push(path.to_path_buf());
        }
        return Ok(());
    }
    if !metadata.is_dir() {
        return Ok(());
    }

    let mut entries = fs::read_dir(path)
        .map_err(|e| format!("{}: {e}", path.display()))?
        .collect::<Result<Vec<_>, _>>()
        .map_err(|e| format!("{}: {e}", path.display()))?;
    entries.sort_by_key(|entry| entry.path());
    for entry in entries {
        collect_policy_files(&entry.path(), files)?;
    }
    Ok(())
}

fn is_policy_file(path: &Path) -> bool {
    matches!(
        path.extension().and_then(|ext| ext.to_str()),
        Some("rs" | "toml" | "yml" | "yaml")
    )
}

fn scan_determinism_text(path: &Path, text: &str, violations: &mut Vec<String>) {
    let mut in_banned_pattern_table = false;
    for (line_idx, line) in text.lines().enumerate() {
        if line.contains("DETERMINISM_BANNED_PATTERNS") {
            in_banned_pattern_table = true;
        }
        if in_banned_pattern_table {
            if line.trim() == "];" {
                in_banned_pattern_table = false;
            }
            continue;
        }
        for &(pattern, reason) in DETERMINISM_BANNED_PATTERNS {
            if line.contains(pattern) && !line.contains("determinism-allow") {
                violations.push(format!(
                    "{}:{}: banned `{}`: {}",
                    path.display(),
                    line_idx + 1,
                    pattern,
                    reason
                ));
            }
        }
    }
}

fn scan_dependency_policy_text(path: &Path, text: &str, violations: &mut Vec<String>) {
    let mut in_dependency_section = false;
    for (line_idx, line) in text.lines().enumerate() {
        let trimmed = line.trim();
        if trimmed.starts_with('[') && trimmed.ends_with(']') {
            in_dependency_section = matches!(
                trimmed,
                "[dependencies]" | "[dev-dependencies]" | "[build-dependencies]"
            );
            continue;
        }
        if !in_dependency_section
            || trimmed.is_empty()
            || trimmed.starts_with('#')
            || !trimmed.contains('=')
            || trimmed.contains("path")
        {
            continue;
        }
        if !dependency_line_has_exact_version(trimmed) {
            violations.push(format!(
                "{}:{}: dependency versions must be exact (`=x.y.z`) or path-based for deterministic builds",
                path.display(),
                line_idx + 1
            ));
        }
    }
}

fn dependency_line_has_exact_version(line: &str) -> bool {
    let Some((_, rhs)) = line.split_once('=') else {
        return true;
    };
    let rhs = rhs.trim();
    if let Some(version_value) = rhs.strip_prefix('"') {
        return version_value.starts_with('=');
    }
    if let Some(version_pos) = rhs.find("version") {
        let version_rhs = &rhs[version_pos + "version".len()..];
        if let Some((_, value)) = version_rhs.split_once('=') {
            let value = value.trim();
            return value
                .strip_prefix('"')
                .is_some_and(|version_value| version_value.starts_with('='));
        }
    }
    false
}

fn generate_testdata(check: bool) -> Result<(), String> {
    let f32_model = tiny_f32_gguf();
    let f32_hash = logits_hash_text(&f32_model)?;
    let qmix_model = tiny_qmix_gguf();
    let qmix_hash = logits_hash_text(&qmix_model)?;
    let tokens = format!(
        "{}\n",
        TOKENS
            .iter()
            .map(|token| token.to_string())
            .collect::<Vec<_>>()
            .join(",")
    );

    if check {
        check_file(F32_MODEL_PATH, &f32_model)?;
        check_file(F32_HASH_PATH, f32_hash.as_bytes())?;
        check_file(QMIX_MODEL_PATH, &qmix_model)?;
        check_file(QMIX_HASH_PATH, qmix_hash.as_bytes())?;
        check_file(TOKENS_PATH, tokens.as_bytes())?;
    } else {
        fs::create_dir_all("testdata").map_err(|e| e.to_string())?;
        fs::write(F32_MODEL_PATH, f32_model).map_err(|e| e.to_string())?;
        fs::write(F32_HASH_PATH, f32_hash).map_err(|e| e.to_string())?;
        fs::write(QMIX_MODEL_PATH, qmix_model).map_err(|e| e.to_string())?;
        fs::write(QMIX_HASH_PATH, qmix_hash).map_err(|e| e.to_string())?;
        fs::write(TOKENS_PATH, tokens).map_err(|e| e.to_string())?;
    }
    Ok(())
}

fn check_file(path: &str, expected: &[u8]) -> Result<(), String> {
    let actual = fs::read(path).map_err(|e| format!("{path}: {e}"))?;
    if actual != expected {
        return Err(format!(
            "{path} is stale; run `cargo run -p xtask -- generate-testdata`"
        ));
    }
    Ok(())
}

fn logits_hash_text(model_bytes: &[u8]) -> Result<String, String> {
    let gguf = det_gguf::parse(model_bytes).map_err(|e| format!("GGUF parse error: {e:?}"))?;
    let model = det_model::F32Llama::from_gguf(&gguf, model_bytes)
        .map_err(|e| format!("model load error: {e:?}"))?;
    let digest = model
        .logits_hash_for_tokens(TOKENS)
        .map_err(|e| format!("logits hash error: {e:?}"))?;
    Ok(format!("{}\n", hex(&digest)))
}

fn bench_testdata(iters: usize) -> Result<(), String> {
    if iters == 0 {
        return Err("bench-testdata: --iters must be greater than zero".to_owned());
    }
    println!("bench-testdata iters={iters}");

    bench_logits("tiny-f32", F32_MODEL_PATH, iters)?;
    bench_logits("tiny-qmix", QMIX_MODEL_PATH, iters)?;
    bench_codec("tiny-f32", F32_MODEL_PATH, iters)?;
    bench_codec("tiny-qmix", QMIX_MODEL_PATH, iters)?;
    Ok(())
}

#[derive(Debug)]
struct BenchFileOpts {
    model: String,
    input: String,
    limit_bytes: Option<usize>,
    n_ctx: Option<usize>,
    iters: usize,
}

#[derive(Debug)]
struct ModelInfoOpts {
    model: String,
}

#[derive(Debug)]
struct CompareLogitsOpts {
    actual: String,
    reference: String,
    min_cosine: Option<f64>,
    row_size: Option<usize>,
    rows: Option<usize>,
}

struct VerifyLogitsHashesOpts {
    dir: String,
    expected_count: usize,
}

fn parse_model_info_opts(args: Vec<String>) -> Result<ModelInfoOpts, String> {
    let mut model = None;
    let mut i = 0usize;
    while i < args.len() {
        match args[i].as_str() {
            "-m" | "--model" => {
                i += 1;
                model = args.get(i).cloned();
            }
            other => return Err(format!("unknown model-info argument: {other}")),
        }
        i += 1;
    }
    Ok(ModelInfoOpts {
        model: model.ok_or("model-info: missing --model")?,
    })
}

fn model_info(opts: ModelInfoOpts) -> Result<(), String> {
    let bytes = fs::read(&opts.model).map_err(|e| format!("{}: {e}", opts.model))?;
    print!("{}", model_info_text(&opts.model, &bytes)?);
    Ok(())
}

fn model_info_text(path: &str, bytes: &[u8]) -> Result<String, String> {
    let model_sha256 = sha256_hex(bytes);
    let gguf = det_gguf::parse(bytes).map_err(|e| format!("{path}: GGUF parse error: {e:?}"))?;
    let mut out = String::new();
    writeln!(
        out,
        "model-info path={} bytes={} sha256={} gguf_version={} metadata={} tensors={} data_offset={}",
        path,
        bytes.len(),
        model_sha256,
        gguf.version,
        gguf.metadata.len(),
        gguf.tensors.len(),
        gguf.data_offset
    )
    .expect("write to string");
    write_model_metadata_summary(&mut out, &gguf);
    write_tokenizer_summary(&mut out, &gguf);
    write_config_summary(&mut out, &gguf);
    write_tensor_inventory(&mut out, &gguf);
    write_vocab_summary(&mut out, &gguf);
    write_required_tensor_summary(&mut out, &gguf);
    Ok(out)
}

fn write_model_metadata_summary(out: &mut String, gguf: &det_gguf::Gguf) {
    for key in [
        "general.architecture",
        "general.name",
        "tokenizer.ggml.model",
        "llama.vocab_size",
        "qwen2.vocab_size",
        "tokenizer.ggml.add_bos_token",
        "tokenizer.ggml.add_eos_token",
        "tokenizer.ggml.bos_token_id",
        "tokenizer.ggml.eos_token_id",
    ] {
        if let Ok(value) = gguf.metadata_value(key) {
            writeln!(
                out,
                "model-info metadata key={} {}",
                key,
                metadata_summary(value)
            )
            .expect("write to string");
        }
    }
    for key in [
        "tokenizer.ggml.tokens",
        "tokenizer.ggml.merges",
        "tokenizer.ggml.scores",
        "tokenizer.ggml.token_type",
    ] {
        if let Ok(value) = gguf.metadata_value(key) {
            writeln!(
                out,
                "model-info metadata key={} {}",
                key,
                metadata_summary(value)
            )
            .expect("write to string");
        }
    }
}

fn write_tokenizer_summary(out: &mut String, gguf: &det_gguf::Gguf) {
    match det_token::Tokenizer::from_gguf(gguf) {
        Ok(tokenizer) => {
            let kind = match tokenizer {
                det_token::Tokenizer::ByteFallback(_) => "byte_fallback",
                det_token::Tokenizer::ByteBpe(_) => "byte_bpe",
                det_token::Tokenizer::SentencePiece(_) => "sentencepiece",
            };
            writeln!(out, "model-info tokenizer status=ok kind={kind}").expect("write to string");
        }
        Err(e) => {
            writeln!(out, "model-info tokenizer status=error error={e:?}")
                .expect("write to string");
        }
    }
}

fn write_config_summary(out: &mut String, gguf: &det_gguf::Gguf) {
    match det_model::LlamaConfig::from_gguf(gguf) {
        Ok(config) => {
            writeln!(
                out,
                "model-info config status=ok block_count={} embedding_length={} feed_forward_length={} head_count={} head_count_kv={} context_length={} rope_dimension_count={} rope_pairing={:?} rope_freq_base={:?} rms_epsilon={:?} attention_scale={:?}",
                config.block_count,
                config.embedding_length,
                config.feed_forward_length,
                config.head_count,
                config.head_count_kv,
                config.context_length,
                config.rope_dimension_count,
                config.rope_pairing,
                config.rope_freq_base,
                config.rms_epsilon,
                config.attention_scale
            )
            .expect("write to string");
        }
        Err(e) => {
            writeln!(out, "model-info config status=error error={e:?}").expect("write to string");
        }
    }
}

fn write_tensor_inventory(out: &mut String, gguf: &det_gguf::Gguf) {
    let mut counts = BTreeMap::<String, usize>::new();
    let mut encoded_bytes = 0u64;
    let mut encoded_len_errors = 0usize;
    for tensor in &gguf.tensors {
        *counts.entry(ggml_type_label(tensor.ty)).or_default() += 1;
        match tensor.encoded_len() {
            Ok(len) => encoded_bytes = encoded_bytes.saturating_add(len),
            Err(_) => encoded_len_errors += 1,
        }
    }
    write!(
        out,
        "model-info tensor-inventory total={} encoded_bytes={} encoded_len_errors={}",
        gguf.tensors.len(),
        encoded_bytes,
        encoded_len_errors
    )
    .expect("write to string");
    for (ty, count) in counts {
        write!(out, " {}={}", ty, count).expect("write to string");
    }
    out.push('\n');
}

fn write_vocab_summary(out: &mut String, gguf: &det_gguf::Gguf) {
    let tokenizer_vocab = gguf_token_vocab_len(gguf);
    let model_vocab = gguf_model_vocab_len(gguf);
    match (&tokenizer_vocab, &model_vocab) {
        (Ok(tokenizer_vocab), Ok(model_vocab)) => {
            let status = match validate_vocab_lengths(*tokenizer_vocab, *model_vocab) {
                Ok(()) => "ok".to_owned(),
                Err(e) => format!("error error={e:?}"),
            };
            writeln!(
                out,
                "model-info vocab status={} tokenizer={} model={} codec_max_symbols={}",
                status,
                tokenizer_vocab,
                model_vocab,
                det_coder::MAX_SYMBOLS
            )
            .expect("write to string");
        }
        _ => {
            writeln!(
                out,
                "model-info vocab status=error tokenizer={:?} model={:?} codec_max_symbols={}",
                tokenizer_vocab,
                model_vocab,
                det_coder::MAX_SYMBOLS
            )
            .expect("write to string");
        }
    }
}

#[derive(Clone, Copy)]
enum ExpectedTensorKind {
    DenseVector,
    WeightMatrix,
}

struct RequiredTensorSummary {
    checked: usize,
    missing: usize,
    shape_mismatch: usize,
    unsupported_type: usize,
    tied_output: bool,
    issues: Vec<String>,
}

fn write_required_tensor_summary(out: &mut String, gguf: &det_gguf::Gguf) {
    let config = match det_model::LlamaConfig::from_gguf(gguf) {
        Ok(config) => config,
        Err(e) => {
            writeln!(
                out,
                "model-info required-tensors status=skipped reason=config_error error={e:?}"
            )
            .expect("write to string");
            return;
        }
    };
    let model_vocab = match gguf_model_vocab_len(gguf) {
        Ok(model_vocab) => model_vocab,
        Err(e) => {
            writeln!(
                out,
                "model-info required-tensors status=skipped reason=vocab_error error={e:?}"
            )
            .expect("write to string");
            return;
        }
    };

    let summary = required_tensor_summary(gguf, config, model_vocab);
    for issue in &summary.issues {
        writeln!(out, "model-info tensor-issue {issue}").expect("write to string");
    }
    let status =
        if summary.missing == 0 && summary.shape_mismatch == 0 && summary.unsupported_type == 0 {
            "ok"
        } else {
            "error"
        };
    writeln!(
        out,
        "model-info required-tensors status={} checked={} missing={} shape_mismatch={} unsupported_type={} tied_output={}",
        status,
        summary.checked,
        summary.missing,
        summary.shape_mismatch,
        summary.unsupported_type,
        summary.tied_output
    )
    .expect("write to string");
}

fn required_tensor_summary(
    gguf: &det_gguf::Gguf,
    config: det_model::LlamaConfig,
    model_vocab: usize,
) -> RequiredTensorSummary {
    let mut summary = RequiredTensorSummary {
        checked: 0,
        missing: 0,
        shape_mismatch: 0,
        unsupported_type: 0,
        tied_output: false,
        issues: Vec::new(),
    };
    let d = config.embedding_length as u64;
    let d_ff = config.feed_forward_length as u64;
    let head_dim = (config.embedding_length / config.head_count) as u64;
    let q_rows = (config.head_count as u64) * head_dim;
    let kv_rows = (config.head_count_kv as u64) * head_dim;
    let vocab = model_vocab as u64;

    check_expected_tensor(
        gguf,
        &mut summary,
        "token_embd.weight",
        &[d, vocab],
        ExpectedTensorKind::WeightMatrix,
    );
    for layer in 0..config.block_count {
        check_expected_tensor(
            gguf,
            &mut summary,
            &format!("blk.{layer}.attn_norm.weight"),
            &[d],
            ExpectedTensorKind::DenseVector,
        );
        check_expected_tensor(
            gguf,
            &mut summary,
            &format!("blk.{layer}.attn_q.weight"),
            &[d, q_rows],
            ExpectedTensorKind::WeightMatrix,
        );
        check_expected_tensor(
            gguf,
            &mut summary,
            &format!("blk.{layer}.attn_k.weight"),
            &[d, kv_rows],
            ExpectedTensorKind::WeightMatrix,
        );
        check_expected_tensor(
            gguf,
            &mut summary,
            &format!("blk.{layer}.attn_v.weight"),
            &[d, kv_rows],
            ExpectedTensorKind::WeightMatrix,
        );
        check_expected_tensor(
            gguf,
            &mut summary,
            &format!("blk.{layer}.attn_output.weight"),
            &[q_rows, d],
            ExpectedTensorKind::WeightMatrix,
        );
        check_expected_tensor(
            gguf,
            &mut summary,
            &format!("blk.{layer}.ffn_norm.weight"),
            &[d],
            ExpectedTensorKind::DenseVector,
        );
        check_expected_tensor(
            gguf,
            &mut summary,
            &format!("blk.{layer}.ffn_gate.weight"),
            &[d, d_ff],
            ExpectedTensorKind::WeightMatrix,
        );
        check_expected_tensor(
            gguf,
            &mut summary,
            &format!("blk.{layer}.ffn_up.weight"),
            &[d, d_ff],
            ExpectedTensorKind::WeightMatrix,
        );
        check_expected_tensor(
            gguf,
            &mut summary,
            &format!("blk.{layer}.ffn_down.weight"),
            &[d_ff, d],
            ExpectedTensorKind::WeightMatrix,
        );
    }
    check_expected_tensor(
        gguf,
        &mut summary,
        "output_norm.weight",
        &[d],
        ExpectedTensorKind::DenseVector,
    );
    if gguf.tensor("output.weight").is_ok() {
        check_expected_tensor(
            gguf,
            &mut summary,
            "output.weight",
            &[d, vocab],
            ExpectedTensorKind::WeightMatrix,
        );
    } else {
        summary.tied_output = true;
    }
    summary
}

fn check_expected_tensor(
    gguf: &det_gguf::Gguf,
    summary: &mut RequiredTensorSummary,
    name: &str,
    dimensions: &[u64],
    kind: ExpectedTensorKind,
) {
    summary.checked += 1;
    let tensor = match gguf.tensor(name) {
        Ok(tensor) => tensor,
        Err(_) => {
            summary.missing += 1;
            summary.issues.push(format!("name={} issue=missing", name));
            return;
        }
    };
    if tensor.dimensions.as_slice() != dimensions {
        summary.shape_mismatch += 1;
        summary.issues.push(format!(
            "name={} issue=shape expected={:?} actual={:?}",
            name, dimensions, tensor.dimensions
        ));
    }
    if !tensor_type_supported_for(kind, tensor.ty) {
        summary.unsupported_type += 1;
        summary.issues.push(format!(
            "name={} issue=unsupported_type type={}",
            name,
            ggml_type_label(tensor.ty)
        ));
    }
}

fn tensor_type_supported_for(kind: ExpectedTensorKind, ty: det_gguf::GgmlType) -> bool {
    match kind {
        ExpectedTensorKind::DenseVector => {
            matches!(ty, det_gguf::GgmlType::F32 | det_gguf::GgmlType::F16)
        }
        ExpectedTensorKind::WeightMatrix => matches!(
            ty,
            det_gguf::GgmlType::F32
                | det_gguf::GgmlType::F16
                | det_gguf::GgmlType::Q8_0
                | det_gguf::GgmlType::Q4_0
        ),
    }
}

fn gguf_model_vocab_len(gguf: &det_gguf::Gguf) -> Result<usize, String> {
    let arch = gguf
        .metadata_str("general.architecture")
        .map_err(|e| format!("general.architecture metadata error: {e:?}"))?;
    for key in [format!("{arch}.vocab_size"), "llama.vocab_size".to_owned()] {
        match gguf.metadata_u32(&key) {
            Ok(v) => return Ok(v as usize),
            Err(det_gguf::GgufError::MetadataNotFound) => {}
            Err(e) => return Err(format!("{key} metadata error: {e:?}")),
        }
    }
    if let Ok(tokenizer_vocab) = gguf_token_vocab_len(gguf) {
        return Ok(tokenizer_vocab);
    }
    let token_embd = gguf
        .tensor("token_embd.weight")
        .map_err(|e| format!("token_embd.weight tensor error: {e:?}"))?;
    if token_embd.dimensions.len() == 2 {
        return usize::try_from(token_embd.dimensions[1])
            .map_err(|_| "token_embd.weight vocab dimension does not fit usize".to_owned());
    }
    Err("model vocabulary metadata is missing".to_owned())
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

fn ggml_type_label(ty: det_gguf::GgmlType) -> String {
    match ty {
        det_gguf::GgmlType::F32 => "F32".to_owned(),
        det_gguf::GgmlType::F16 => "F16".to_owned(),
        det_gguf::GgmlType::Q4_0 => "Q4_0".to_owned(),
        det_gguf::GgmlType::Q8_0 => "Q8_0".to_owned(),
        det_gguf::GgmlType::Other(raw) => format!("OTHER_{raw}"),
    }
}

fn parse_bench_file_opts(args: Vec<String>) -> Result<BenchFileOpts, String> {
    let mut model = None;
    let mut input = None;
    let mut limit_bytes = None;
    let mut n_ctx = None;
    let mut iters = 1usize;
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
            "--limit-bytes" => {
                i += 1;
                let raw = args
                    .get(i)
                    .ok_or("bench-file: missing value for --limit-bytes")?;
                let value = raw
                    .parse::<usize>()
                    .map_err(|e| format!("bench-file: invalid --limit-bytes value: {e}"))?;
                if value == 0 {
                    return Err("bench-file: --limit-bytes must be greater than zero".to_owned());
                }
                limit_bytes = Some(value);
            }
            "--n-ctx" => {
                i += 1;
                let raw = args.get(i).ok_or("bench-file: missing value for --n-ctx")?;
                n_ctx = Some(
                    raw.parse::<usize>()
                        .map_err(|e| format!("bench-file: invalid --n-ctx value: {e}"))?,
                );
            }
            "--iters" => {
                i += 1;
                let raw = args.get(i).ok_or("bench-file: missing value for --iters")?;
                iters = raw
                    .parse::<usize>()
                    .map_err(|e| format!("bench-file: invalid --iters value: {e}"))?;
            }
            other => return Err(format!("unknown bench-file argument: {other}")),
        }
        i += 1;
    }
    if iters == 0 {
        return Err("bench-file: --iters must be greater than zero".to_owned());
    }
    Ok(BenchFileOpts {
        model: model.ok_or("bench-file: missing --model")?,
        input: input.ok_or("bench-file: missing --input")?,
        limit_bytes,
        n_ctx,
        iters,
    })
}

fn bench_file(opts: BenchFileOpts) -> Result<(), String> {
    let model_bytes = fs::read(&opts.model).map_err(|e| format!("{}: {e}", opts.model))?;
    let model_sha256 = sha256_hex(&model_bytes);
    let gguf = det_gguf::parse(&model_bytes)
        .map_err(|e| format!("{}: GGUF parse error: {e:?}", opts.model))?;
    let model = det_model::F32Llama::from_gguf(&gguf, &model_bytes)
        .map_err(|e| format!("{}: model load error: {e:?}", opts.model))?;
    validate_tokenizer_and_codec_vocab(&gguf, &model)?;
    let tokenizer = det_token::Tokenizer::from_gguf(&gguf)
        .map_err(|e| format!("{}: tokenizer error: {e:?}", opts.model))?;
    let mut input = fs::read(&opts.input).map_err(|e| format!("{}: {e}", opts.input))?;
    let source_input_bytes = input.len();
    if let Some(limit_bytes) = opts.limit_bytes {
        input.truncate(limit_bytes);
    }
    let measured_input_bytes = input.len();
    let input_sha256 = sha256_hex(&input);
    let token_ids: Vec<usize> = tokenizer
        .tokenize_bytes(&input)
        .map_err(|e| format!("{}: tokenize error: {e:?}", opts.input))?
        .into_iter()
        .map(|token| token as usize)
        .collect();

    let n_ctx = opts.n_ctx.unwrap_or(model.config.context_length);
    let overlap = n_ctx / 4;
    validate_window(n_ctx, overlap, model.config.context_length)?;

    let (_, restored) = codec_round_trip(&model, &tokenizer, &token_ids, n_ctx, overlap)?;
    if restored != input {
        return Err("bench-file: warmup did not round-trip".to_owned());
    }

    let start = Instant::now();
    let mut payload_bytes = 0usize;
    for _ in 0..opts.iters {
        let (payload, restored) = codec_round_trip(&model, &tokenizer, &token_ids, n_ctx, overlap)?;
        if restored != input {
            return Err("bench-file: benchmark iteration did not round-trip".to_owned());
        }
        payload_bytes += payload.len();
    }
    let elapsed = start.elapsed();
    let input_bytes = input.len() * opts.iters;
    let dtlz_bytes = payload_bytes + det_coder::file::HEADER_LEN * opts.iters;
    let dtlz_bits_per_byte = if input_bytes == 0 {
        0.0
    } else {
        (dtlz_bytes as f64 * 8.0) / input_bytes as f64
    };
    let payload_bits_per_byte = if input_bytes == 0 {
        0.0
    } else {
        (payload_bytes as f64 * 8.0) / input_bytes as f64
    };
    let compression_ratio = if input_bytes == 0 {
        0.0
    } else {
        dtlz_bytes as f64 / input_bytes as f64
    };
    let limit_label = opts
        .limit_bytes
        .map_or_else(|| "all".to_owned(), |limit_bytes| limit_bytes.to_string());
    println!(
        "bench-file model={} input={} limit_bytes={} iters={} n_ctx={} overlap={} model_sha256={} input_sha256={}",
        opts.model, opts.input, limit_label, opts.iters, n_ctx, overlap, model_sha256, input_sha256
    );
    println!(
        "bench-file: source_input_bytes={} measured_input_bytes={} total_input_bytes={} tokens={} total_tokens={} payload_bytes={} dtlz_bytes={} payload_bits_per_byte={:.6} dtlz_bits_per_byte={:.6} compression_ratio={:.6} elapsed_ms={:.3} input_bytes_per_s={:.3} tokens_per_s={:.3}",
        source_input_bytes,
        measured_input_bytes,
        input_bytes,
        token_ids.len(),
        token_ids.len() * opts.iters,
        payload_bytes,
        dtlz_bytes,
        payload_bits_per_byte,
        dtlz_bits_per_byte,
        compression_ratio,
        elapsed.as_secs_f64() * 1000.0,
        input_bytes as f64 / elapsed.as_secs_f64(),
        (token_ids.len() * opts.iters) as f64 / elapsed.as_secs_f64()
    );
    Ok(())
}

fn parse_compare_logits_opts(args: Vec<String>) -> Result<CompareLogitsOpts, String> {
    let mut actual = None;
    let mut reference = None;
    let mut min_cosine = None;
    let mut row_size = None;
    let mut rows = None;
    let mut i = 0usize;
    while i < args.len() {
        match args[i].as_str() {
            "--actual" => {
                i += 1;
                actual = args.get(i).cloned();
            }
            "--reference" => {
                i += 1;
                reference = args.get(i).cloned();
            }
            "--min-cosine" => {
                i += 1;
                let raw = args
                    .get(i)
                    .ok_or("compare-logits: missing value for --min-cosine")?;
                let value = raw
                    .parse::<f64>()
                    .map_err(|e| format!("compare-logits: invalid --min-cosine value: {e}"))?;
                if !(0.0..=1.0).contains(&value) {
                    return Err("compare-logits: --min-cosine must be in [0, 1]".to_owned());
                }
                min_cosine = Some(value);
            }
            "--row-size" => {
                i += 1;
                let raw = args
                    .get(i)
                    .ok_or("compare-logits: missing value for --row-size")?;
                let value = raw
                    .parse::<usize>()
                    .map_err(|e| format!("compare-logits: invalid --row-size value: {e}"))?;
                if value == 0 {
                    return Err("compare-logits: --row-size must be greater than zero".to_owned());
                }
                row_size = Some(value);
            }
            "--rows" => {
                i += 1;
                let raw = args
                    .get(i)
                    .ok_or("compare-logits: missing value for --rows")?;
                let value = raw
                    .parse::<usize>()
                    .map_err(|e| format!("compare-logits: invalid --rows value: {e}"))?;
                if value == 0 {
                    return Err("compare-logits: --rows must be greater than zero".to_owned());
                }
                rows = Some(value);
            }
            other => return Err(format!("unknown compare-logits argument: {other}")),
        }
        i += 1;
    }
    if rows.is_some() && row_size.is_none() {
        return Err("compare-logits: --rows requires --row-size".to_owned());
    }
    Ok(CompareLogitsOpts {
        actual: actual.ok_or("compare-logits: missing --actual")?,
        reference: reference.ok_or("compare-logits: missing --reference")?,
        min_cosine,
        row_size,
        rows,
    })
}

fn compare_logits(opts: CompareLogitsOpts) -> Result<(), String> {
    let actual_bytes = fs::read(&opts.actual).map_err(|e| format!("{}: {e}", opts.actual))?;
    let reference_bytes =
        fs::read(&opts.reference).map_err(|e| format!("{}: {e}", opts.reference))?;
    let actual = parse_logits_dump(&actual_bytes, &opts.actual)?;
    let reference = parse_logits_dump(&reference_bytes, &opts.reference)?;
    let metrics = compare_logits_values(&actual, &reference)?;
    let row_metrics = match opts.row_size {
        Some(row_size) => Some(compare_logits_rows(&actual, &reference, row_size)?),
        None => None,
    };
    if let Some(min_cosine) = opts.min_cosine {
        if metrics.cosine < min_cosine {
            return Err(format!(
                "compare-logits: cosine {:.9} is below threshold {:.9}",
                metrics.cosine, min_cosine
            ));
        }
        if let Some(row_metrics) = &row_metrics {
            if row_metrics.min_cosine < min_cosine {
                return Err(format!(
                    "compare-logits: min row cosine {:.9} is below threshold {:.9}",
                    row_metrics.min_cosine, min_cosine
                ));
            }
        }
    }
    if let Some(row_metrics) = row_metrics {
        validate_expected_rows(&row_metrics, opts.rows)?;
        println!(
            "compare-logits values={} cosine={:.9} max_abs_diff={:.9} rms_diff={:.9} rows={} row_size={} min_row_cosine={:.9}",
            metrics.values,
            metrics.cosine,
            metrics.max_abs_diff,
            metrics.rms_diff,
            row_metrics.rows,
            row_metrics.row_size,
            row_metrics.min_cosine
        );
    } else {
        println!(
            "compare-logits values={} cosine={:.9} max_abs_diff={:.9} rms_diff={:.9}",
            metrics.values, metrics.cosine, metrics.max_abs_diff, metrics.rms_diff
        );
    }
    Ok(())
}

fn validate_expected_rows(
    row_metrics: &LogitsRowMetrics,
    expected_rows: Option<usize>,
) -> Result<(), String> {
    if let Some(expected_rows) = expected_rows {
        if row_metrics.rows != expected_rows {
            return Err(format!(
                "compare-logits: row count {} does not match expected {}",
                row_metrics.rows, expected_rows
            ));
        }
    }
    Ok(())
}

fn parse_verify_logits_hashes_opts(args: Vec<String>) -> Result<VerifyLogitsHashesOpts, String> {
    let mut dir = None;
    let mut expected_count = None;
    let mut i = 0usize;
    while i < args.len() {
        match args[i].as_str() {
            "--dir" => {
                i += 1;
                dir = args.get(i).cloned();
            }
            "--expected-count" => {
                i += 1;
                let raw = args
                    .get(i)
                    .ok_or("verify-logits-hashes: missing value for --expected-count")?;
                let value = raw.parse::<usize>().map_err(|e| {
                    format!("verify-logits-hashes: invalid --expected-count value: {e}")
                })?;
                if value == 0 {
                    return Err(
                        "verify-logits-hashes: --expected-count must be greater than zero"
                            .to_owned(),
                    );
                }
                expected_count = Some(value);
            }
            other => return Err(format!("unknown verify-logits-hashes argument: {other}")),
        }
        i += 1;
    }
    Ok(VerifyLogitsHashesOpts {
        dir: dir.ok_or("verify-logits-hashes: missing --dir")?,
        expected_count: expected_count.ok_or("verify-logits-hashes: missing --expected-count")?,
    })
}

fn verify_logits_hashes(opts: VerifyLogitsHashesOpts) -> Result<(), String> {
    let mut files = Vec::new();
    collect_logits_hash_files(Path::new(&opts.dir), &mut files)?;
    files.sort();
    if files.len() != opts.expected_count {
        return Err(format!(
            "verify-logits-hashes: expected {} hash artifacts, found {}",
            opts.expected_count,
            files.len()
        ));
    }

    let mut reference = None;
    for file in &files {
        let text = fs::read_to_string(file).map_err(|e| format!("{}: {e}", file.display()))?;
        let hashes = parse_labeled_logits_hashes(&text, &file.display().to_string())?;
        if let Some(reference) = &reference {
            if &hashes != reference {
                return Err(format!(
                    "verify-logits-hashes: {} does not match reference",
                    file.display()
                ));
            }
        } else {
            reference = Some(hashes);
        }
    }
    println!(
        "verify-logits-hashes artifacts={} fixtures={}",
        files.len(),
        EXPECTED_LOGITS_HASH_LABELS.len()
    );
    Ok(())
}

fn check_ci_workflow() -> Result<(), String> {
    let path = Path::new(".github/workflows/ci.yml");
    let text = fs::read_to_string(path).map_err(|e| format!("{}: {e}", path.display()))?;
    validate_ci_workflow_text(&text)?;
    println!("ci workflow structure check passed");
    Ok(())
}

fn validate_ci_workflow_text(text: &str) -> Result<(), String> {
    let required = [
        ("hygiene job", "  hygiene:"),
        ("manual workflow dispatch trigger", "  workflow_dispatch:"),
        ("test job", "  test:"),
        ("logits-hash-match job", "  logits-hash-match:"),
        ("msrv job", "  msrv:"),
        ("toolchain-skew job", "  toolchain-skew:"),
        ("wasm job", "  wasm:"),
        ("native x86_64-linux target", "name: x86_64-linux"),
        ("native aarch64-macos target", "name: aarch64-macos"),
        ("native aarch64-linux target", "name: aarch64-linux"),
        (
            "cross-job hash-match dependencies",
            "needs: [test, toolchain-skew, wasm]",
        ),
        (
            "six-artifact hash verification",
            "verify-logits-hashes --dir logits-hashes --expected-count 6",
        ),
        (
            "native test artifact upload",
            "name: logits-hashes-${{ matrix.name }}",
        ),
        (
            "toolchain artifact upload",
            "name: logits-hashes-toolchain-${{ matrix.toolchain }}",
        ),
        ("wasm artifact upload", "name: logits-hashes-wasm32-wasip1"),
        ("stable toolchain skew entry", "toolchain: [stable,"),
        (
            "wasm target build",
            "cargo build --workspace --target wasm32-wasip1",
        ),
        (
            "wasm selftest execution",
            "wasmtime target/wasm32-wasip1/debug/detllm.wasm selftest",
        ),
        (
            "wasm logits execution",
            "wasmtime --dir . target/wasm32-wasip1/debug/detllm.wasm logits",
        ),
        (
            "wasm codec compress smoke",
            "wasmtime --dir . target/wasm32-wasip1/debug/detllm.wasm compress",
        ),
        (
            "wasm codec decompress smoke",
            "wasmtime --dir . target/wasm32-wasip1/debug/detllm.wasm decompress",
        ),
        (
            "wasm quant-kernel hash comparison",
            "cmp native-quant-kernel-hash.txt wasm-quant-kernel-hash.txt",
        ),
        (
            "workflow self-check in hygiene",
            "cargo run -p xtask -- check-ci-workflow",
        ),
    ];
    for (label, needle) in required {
        if !text.contains(needle) {
            return Err(format!("ci workflow is missing {label}: {needle}"));
        }
    }

    let artifact_uploads = text.matches("uses: actions/upload-artifact@v4").count();
    if artifact_uploads != 3 {
        return Err(format!(
            "ci workflow must upload exactly three logits artifact groups, found {artifact_uploads}"
        ));
    }

    let fixture_hash_blocks = text
        .matches("cargo run -p det-cli -- logits -m testdata/tiny-f32.gguf")
        .count();
    if fixture_hash_blocks < 3 {
        return Err(format!(
            "ci workflow must hash bundled fixtures in native, toolchain-skew, and wasm jobs; found {fixture_hash_blocks} tiny-f32 hash commands"
        ));
    }
    if text.contains("${{ runner.temp }}") {
        return Err(
            "ci workflow must not use runner context in job-level env; use a literal /tmp path"
                .to_owned(),
        );
    }
    Ok(())
}

const EXPECTED_LOGITS_HASH_LABELS: &[&str] = &["tiny-f32", "tiny-qmix"];

fn collect_logits_hash_files(path: &Path, files: &mut Vec<PathBuf>) -> Result<(), String> {
    let metadata = fs::metadata(path).map_err(|e| format!("{}: {e}", path.display()))?;
    if metadata.is_file() {
        if path.file_name().and_then(|name| name.to_str()) == Some("logits-hashes.txt") {
            files.push(path.to_path_buf());
        }
        return Ok(());
    }
    if !metadata.is_dir() {
        return Ok(());
    }

    let mut entries = fs::read_dir(path)
        .map_err(|e| format!("{}: {e}", path.display()))?
        .collect::<Result<Vec<_>, _>>()
        .map_err(|e| format!("{}: {e}", path.display()))?;
    entries.sort_by_key(|entry| entry.path());
    for entry in entries {
        collect_logits_hash_files(&entry.path(), files)?;
    }
    Ok(())
}

fn parse_labeled_logits_hashes(text: &str, label: &str) -> Result<Vec<(String, String)>, String> {
    let mut hashes = Vec::new();
    for (line_idx, line) in text.lines().enumerate() {
        let mut parts = line.split_whitespace();
        let name = parts
            .next()
            .ok_or_else(|| format!("{label}: empty line {}", line_idx + 1))?;
        let hash = parts
            .next()
            .ok_or_else(|| format!("{label}: missing hash on line {}", line_idx + 1))?;
        if parts.next().is_some() {
            return Err(format!("{label}: extra fields on line {}", line_idx + 1));
        }
        if hashes.iter().any(|(seen, _)| seen == name) {
            return Err(format!("{label}: duplicate fixture label `{name}`"));
        }
        if !is_sha256_hex(hash) {
            return Err(format!("{label}: invalid SHA-256 hex for `{name}`"));
        }
        hashes.push((name.to_owned(), hash.to_owned()));
    }
    hashes.sort_by(|a, b| a.0.cmp(&b.0));

    let expected = EXPECTED_LOGITS_HASH_LABELS;
    if hashes.len() != expected.len() {
        return Err(format!(
            "{label}: expected {} fixture hashes, found {}",
            expected.len(),
            hashes.len()
        ));
    }
    for (idx, &expected_label) in expected.iter().enumerate() {
        if hashes[idx].0 != expected_label {
            return Err(format!(
                "{label}: expected fixture label `{expected_label}`, found `{}`",
                hashes[idx].0
            ));
        }
    }
    Ok(hashes)
}

fn is_sha256_hex(value: &str) -> bool {
    value.len() == 64
        && value
            .bytes()
            .all(|byte| byte.is_ascii_digit() || (b'a'..=b'f').contains(&byte))
}

struct LogitsCompareMetrics {
    values: usize,
    cosine: f64,
    max_abs_diff: f64,
    rms_diff: f64,
}

struct LogitsRowMetrics {
    rows: usize,
    row_size: usize,
    min_cosine: f64,
}

fn parse_logits_dump(bytes: &[u8], label: &str) -> Result<Vec<f32>, String> {
    if bytes.is_empty() {
        return Err(format!("{label}: logits dump is empty"));
    }
    if bytes.len() % 4 != 0 {
        return Err(format!(
            "{label}: logits dump length {} is not a multiple of 4",
            bytes.len()
        ));
    }
    let mut out = Vec::with_capacity(bytes.len() / 4);
    for (idx, chunk) in bytes.chunks_exact(4).enumerate() {
        let value = f32::from_le_bytes([chunk[0], chunk[1], chunk[2], chunk[3]]);
        if !value.is_finite() {
            return Err(format!("{label}: non-finite value at f32 index {idx}"));
        }
        out.push(value);
    }
    Ok(out)
}

fn compare_logits_values(
    actual: &[f32],
    reference: &[f32],
) -> Result<LogitsCompareMetrics, String> {
    if actual.len() != reference.len() {
        return Err(format!(
            "compare-logits: length mismatch actual={} reference={}",
            actual.len(),
            reference.len()
        ));
    }
    if actual.is_empty() {
        return Err("compare-logits: no values to compare".to_owned());
    }

    let mut dot = 0.0f64;
    let mut actual_norm = 0.0f64;
    let mut reference_norm = 0.0f64;
    let mut max_abs_diff = 0.0f64;
    let mut sum_sq_diff = 0.0f64;
    for (&a, &r) in actual.iter().zip(reference) {
        let a = a as f64;
        let r = r as f64;
        dot += a * r;
        actual_norm += a * a;
        reference_norm += r * r;
        let diff = (a - r).abs();
        max_abs_diff = max_abs_diff.max(diff);
        sum_sq_diff += diff * diff;
    }
    if actual_norm == 0.0 || reference_norm == 0.0 {
        return Err("compare-logits: cosine is undefined for zero-norm input".to_owned());
    }
    Ok(LogitsCompareMetrics {
        values: actual.len(),
        cosine: dot / (actual_norm.sqrt() * reference_norm.sqrt()),
        max_abs_diff,
        rms_diff: (sum_sq_diff / actual.len() as f64).sqrt(),
    })
}

fn compare_logits_rows(
    actual: &[f32],
    reference: &[f32],
    row_size: usize,
) -> Result<LogitsRowMetrics, String> {
    if row_size == 0 {
        return Err("compare-logits: --row-size must be greater than zero".to_owned());
    }
    if actual.len() != reference.len() {
        return Err(format!(
            "compare-logits: length mismatch actual={} reference={}",
            actual.len(),
            reference.len()
        ));
    }
    if actual.is_empty() || actual.len() % row_size != 0 {
        return Err(format!(
            "compare-logits: value count {} is not divisible by row size {}",
            actual.len(),
            row_size
        ));
    }

    let mut min_cosine = f64::INFINITY;
    for (actual_row, reference_row) in actual
        .chunks_exact(row_size)
        .zip(reference.chunks_exact(row_size))
    {
        let row = compare_logits_values(actual_row, reference_row)?;
        min_cosine = min_cosine.min(row.cosine);
    }
    Ok(LogitsRowMetrics {
        rows: actual.len() / row_size,
        row_size,
        min_cosine,
    })
}

fn codec_round_trip(
    model: &det_model::F32Llama,
    tokenizer: &det_token::Tokenizer,
    token_ids: &[usize],
    n_ctx: usize,
    overlap: usize,
) -> Result<(Vec<u8>, Vec<u8>), String> {
    let payload = encode_tokens_with_model(model, token_ids, n_ctx, overlap)?;
    let decoded = decode_tokens_with_model(model, &payload, token_ids.len(), n_ctx, overlap)?;
    let decoded_u32 = decoded
        .into_iter()
        .map(|token| u32::try_from(token).map_err(|_| format!("decoded token too large: {token}")))
        .collect::<Result<Vec<_>, _>>()?;
    let restored = tokenizer
        .detokenize_bytes(&decoded_u32)
        .map_err(|e| format!("detokenize error: {e:?}"))?;
    Ok((payload, restored))
}

fn bench_logits(label: &str, path: &str, iters: usize) -> Result<(), String> {
    let bytes = fs::read(path).map_err(|e| format!("{path}: {e}"))?;
    let gguf = det_gguf::parse(&bytes).map_err(|e| format!("{label}: GGUF parse error: {e:?}"))?;
    let model = det_model::F32Llama::from_gguf(&gguf, &bytes)
        .map_err(|e| format!("{label}: model load error: {e:?}"))?;
    let digest = model
        .logits_hash_for_tokens(TOKENS)
        .map_err(|e| format!("{label}: warmup logits error: {e:?}"))?;

    let start = Instant::now();
    for _ in 0..iters {
        let next = model
            .logits_hash_for_tokens(TOKENS)
            .map_err(|e| format!("{label}: logits error: {e:?}"))?;
        if next != digest {
            return Err(format!("{label}: logits hash changed during benchmark"));
        }
    }
    let elapsed = start.elapsed();
    let tokens = TOKENS.len() * iters;
    println!(
        "logits {label}: hash={} tokens={} elapsed_ms={:.3} tokens_per_s={:.3}",
        hex(&digest),
        tokens,
        elapsed.as_secs_f64() * 1000.0,
        tokens as f64 / elapsed.as_secs_f64()
    );
    Ok(())
}

fn bench_codec(label: &str, path: &str, iters: usize) -> Result<(), String> {
    let bytes = fs::read(path).map_err(|e| format!("{path}: {e}"))?;
    let gguf = det_gguf::parse(&bytes).map_err(|e| format!("{label}: GGUF parse error: {e:?}"))?;
    let model = det_model::F32Llama::from_gguf(&gguf, &bytes)
        .map_err(|e| format!("{label}: model load error: {e:?}"))?;
    validate_tokenizer_and_codec_vocab(&gguf, &model)?;
    let tokenizer = det_token::Tokenizer::from_gguf(&gguf)
        .map_err(|e| format!("{label}: tokenizer error: {e:?}"))?;
    let input = b"detllm deterministic compression smoke\n";
    let token_ids: Vec<usize> = tokenizer
        .tokenize_bytes(input)
        .map_err(|e| format!("{label}: tokenize error: {e:?}"))?
        .into_iter()
        .map(|token| token as usize)
        .collect();
    let n_ctx = 8;
    let overlap = 2;
    let payload = encode_tokens_with_model(&model, &token_ids, n_ctx, overlap)?;
    let decoded = decode_tokens_with_model(&model, &payload, token_ids.len(), n_ctx, overlap)?;
    if decoded != token_ids {
        return Err(format!("{label}: codec warmup did not round-trip"));
    }

    let start = Instant::now();
    let mut encoded_bytes = 0usize;
    for _ in 0..iters {
        let payload = encode_tokens_with_model(&model, &token_ids, n_ctx, overlap)?;
        encoded_bytes += payload.len();
        let decoded = decode_tokens_with_model(&model, &payload, token_ids.len(), n_ctx, overlap)?;
        if decoded != token_ids {
            return Err(format!("{label}: codec benchmark did not round-trip"));
        }
    }
    let elapsed = start.elapsed();
    let bytes = input.len() * iters;
    println!(
        "codec {label}: input_bytes={} payload_bytes={} elapsed_ms={:.3} input_bytes_per_s={:.3}",
        bytes,
        encoded_bytes,
        elapsed.as_secs_f64() * 1000.0,
        bytes as f64 / elapsed.as_secs_f64()
    );
    Ok(())
}

fn encode_tokens_with_model(
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
        let (&cum, &freq) = cdf
            .cum
            .get(tokens[pos])
            .zip(cdf.freq.get(tokens[pos]))
            .ok_or_else(|| format!("token {} is outside vocabulary", tokens[pos]))?;
        enc.encode(cum, freq as u64, cdf.total)
            .map_err(|e| format!("range encode error: {e:?}"))?;
    }
    Ok(enc.finish())
}

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
    let mut window_start = 0usize;
    for pos in 0..token_len {
        window_start = next_window_start(pos, window_start, n_ctx, overlap);
        let cdf = cdf_for_context(model, &tokens[window_start..pos], n_ctx)?;
        let value = dec
            .decode_freq(cdf.total)
            .map_err(|e| format!("range decode error: {e:?}"))?;
        let token = cdf
            .symbol_for(value)
            .ok_or_else(|| format!("CDF lookup failed for value {value}"))?;
        dec.advance(cdf.cum[token], cdf.freq[token] as u64, cdf.total)
            .map_err(|e| format!("range advance error: {e:?}"))?;
        tokens.push(token);
    }
    Ok(tokens)
}

fn next_window_start(pos: usize, window_start: usize, n_ctx: usize, overlap: usize) -> usize {
    if pos.saturating_sub(window_start) >= n_ctx {
        pos.saturating_sub(overlap)
    } else {
        window_start
    }
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

fn validate_tokenizer_and_codec_vocab(
    gguf: &det_gguf::Gguf,
    model: &det_model::F32Llama,
) -> Result<(), String> {
    let tokenizer_vocab_len = gguf_token_vocab_len(gguf)?;
    let model_vocab_len = model.output.rows();
    validate_vocab_lengths(tokenizer_vocab_len, model_vocab_len)
}

fn validate_vocab_lengths(
    tokenizer_vocab_len: usize,
    model_vocab_len: usize,
) -> Result<(), String> {
    if tokenizer_vocab_len != model_vocab_len {
        return Err(format!(
            "tokenizer vocabulary length {tokenizer_vocab_len} does not match model vocabulary {model_vocab_len}"
        ));
    }
    if model_vocab_len > det_coder::MAX_SYMBOLS {
        return Err(format!(
            "model vocabulary {model_vocab_len} exceeds codec symbol limit {}",
            det_coder::MAX_SYMBOLS
        ));
    }
    Ok(())
}

fn gguf_token_vocab_len(gguf: &det_gguf::Gguf) -> Result<usize, String> {
    match gguf.metadata_value("tokenizer.ggml.tokens") {
        Ok(det_gguf::MetadataValue::ArrayString(tokens)) => Ok(tokens.len()),
        Ok(_) => Err("tokenizer.ggml.tokens has the wrong metadata type".to_owned()),
        Err(e) => Err(format!("tokenizer.ggml.tokens metadata error: {e:?}")),
    }
}

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
        return det_coder::uniform_cdf(vocab).map_err(|e| format!("uniform CDF error: {e:?}"));
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
    det_coder::logits_to_cdf(&logits).map_err(|e| format!("CDF error: {e:?}"))
}

fn tiny_f32_gguf() -> Vec<u8> {
    let mut tensors = Vec::new();
    let mut data = Vec::new();
    let vocab_size = 256;
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

    encode_gguf(
        tensors,
        data,
        ModelSpec {
            vocab_size,
            embedding_length: 4,
            feed_forward_length: 6,
            head_count: 2,
            head_count_kv: 1,
            context_length: 16,
        },
    )
}

fn tiny_qmix_gguf() -> Vec<u8> {
    let mut tensors = Vec::new();
    let mut data = Vec::new();
    let vocab_size = 256;
    push_q8_tensor(
        &mut tensors,
        &mut data,
        "token_embd.weight",
        vocab_size,
        32,
        0x3c00,
        2,
    );
    push_vector(&mut tensors, &mut data, "blk.0.attn_norm.weight", 32, 1.0);
    push_q8_tensor(
        &mut tensors,
        &mut data,
        "blk.0.attn_q.weight",
        32,
        32,
        0x3c00,
        1,
    );
    push_q4_tensor(
        &mut tensors,
        &mut data,
        "blk.0.attn_k.weight",
        16,
        32,
        0x3c00,
        0x99,
    );
    push_q8_tensor(
        &mut tensors,
        &mut data,
        "blk.0.attn_v.weight",
        16,
        32,
        0x3c00,
        -1,
    );
    push_q4_tensor(
        &mut tensors,
        &mut data,
        "blk.0.attn_output.weight",
        32,
        32,
        0x3c00,
        0x99,
    );
    push_vector(&mut tensors, &mut data, "blk.0.ffn_norm.weight", 32, 1.0);
    push_q8_tensor(
        &mut tensors,
        &mut data,
        "blk.0.ffn_gate.weight",
        32,
        32,
        0x3c00,
        2,
    );
    push_q4_tensor(
        &mut tensors,
        &mut data,
        "blk.0.ffn_up.weight",
        32,
        32,
        0x3c00,
        0x99,
    );
    push_q8_tensor(
        &mut tensors,
        &mut data,
        "blk.0.ffn_down.weight",
        32,
        32,
        0x3c00,
        -2,
    );
    push_vector(&mut tensors, &mut data, "output_norm.weight", 32, 1.0);

    encode_gguf(
        tensors,
        data,
        ModelSpec {
            vocab_size,
            embedding_length: 32,
            feed_forward_length: 32,
            head_count: 2,
            head_count_kv: 1,
            context_length: 16,
        },
    )
}

fn encode_gguf(tensors: Vec<TensorSpec>, data: Vec<u8>, spec: ModelSpec) -> Vec<u8> {
    let mut out = Vec::new();
    out.extend_from_slice(b"GGUF");
    push_u32(&mut out, 3);
    push_u64(&mut out, tensors.len() as u64);
    push_u64(&mut out, 12);
    push_meta_string(&mut out, "general.architecture", "llama");
    push_meta_u32(&mut out, "llama.block_count", 1);
    push_meta_u32(&mut out, "llama.embedding_length", spec.embedding_length);
    push_meta_u32(
        &mut out,
        "llama.feed_forward_length",
        spec.feed_forward_length,
    );
    push_meta_u32(&mut out, "llama.attention.head_count", spec.head_count);
    push_meta_u32(
        &mut out,
        "llama.attention.head_count_kv",
        spec.head_count_kv,
    );
    push_meta_f32(&mut out, "llama.attention.layer_norm_rms_epsilon", 1e-5);
    push_meta_f32(&mut out, "llama.rope.freq_base", 10_000.0);
    push_meta_u32(
        &mut out,
        "llama.rope.dimension_count",
        spec.embedding_length / spec.head_count,
    );
    push_meta_u32(&mut out, "llama.context_length", spec.context_length);
    push_meta_u32(&mut out, "llama.vocab_size", spec.vocab_size as u32);
    push_meta_token_array(&mut out);

    for tensor in &tensors {
        push_string(&mut out, &tensor.name);
        push_u32(&mut out, tensor.dims.len() as u32);
        for &dim in &tensor.dims {
            push_u64(&mut out, dim);
        }
        push_u32(&mut out, tensor.ty);
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
    ty: u32,
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
        ty: 0,
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
        ty: 0,
        offset,
    });
    for _ in 0..len {
        data.extend_from_slice(&value.to_le_bytes());
    }
}

fn push_q8_tensor(
    tensors: &mut Vec<TensorSpec>,
    data: &mut Vec<u8>,
    name: &str,
    rows: usize,
    cols: usize,
    scale_f16: u16,
    q: i8,
) {
    let offset = data.len() as u64;
    tensors.push(TensorSpec {
        name: name.to_owned(),
        dims: vec![cols as u64, rows as u64],
        ty: 8,
        offset,
    });
    for _ in 0..rows * (cols / 32) {
        data.extend_from_slice(&scale_f16.to_le_bytes());
        for _ in 0..32 {
            data.push(q as u8);
        }
    }
}

fn push_q4_tensor(
    tensors: &mut Vec<TensorSpec>,
    data: &mut Vec<u8>,
    name: &str,
    rows: usize,
    cols: usize,
    scale_f16: u16,
    packed: u8,
) {
    let offset = data.len() as u64;
    tensors.push(TensorSpec {
        name: name.to_owned(),
        dims: vec![cols as u64, rows as u64],
        ty: 2,
        offset,
    });
    for _ in 0..rows * (cols / 32) {
        data.extend_from_slice(&scale_f16.to_le_bytes());
        for _ in 0..16 {
            data.push(packed);
        }
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

fn push_meta_token_array(out: &mut Vec<u8>) {
    push_string(out, "tokenizer.ggml.tokens");
    push_u32(out, 9);
    push_u32(out, 8);
    push_u64(out, 256);
    for b in 0..=255u32 {
        push_string(out, &format!("<0x{b:02X}>"));
    }
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

fn sha256_hex(bytes: &[u8]) -> String {
    let mut h = det_num::Sha256::new();
    h.update(bytes);
    hex(&h.finalize())
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

    #[test]
    fn parse_logits_dump_rejects_malformed_inputs() {
        assert!(parse_logits_dump(&[], "empty").is_err());
        assert!(parse_logits_dump(&[0, 1, 2], "short").is_err());
        assert!(parse_logits_dump(&f32::NAN.to_le_bytes(), "nan").is_err());
    }

    #[test]
    fn compare_logits_values_reports_cosine_and_diffs() {
        let actual = [1.0f32, 2.0, 3.0, -4.0];
        let reference = [1.0f32, 2.0, 2.5, -4.5];
        let metrics = compare_logits_values(&actual, &reference).expect("metrics");
        assert_eq!(metrics.values, 4);
        assert!(metrics.cosine > 0.99);
        assert_eq!(metrics.max_abs_diff, 0.5);
        assert!((metrics.rms_diff - 0.3535533905932738).abs() < 1e-12);
    }

    #[test]
    fn compare_logits_rows_reports_minimum_row_cosine() {
        let actual = [1.0f32, 0.0, 0.0, 1.0];
        let reference = [1.0f32, 0.0, 1.0, 1.0];
        let rows = compare_logits_rows(&actual, &reference, 2).expect("rows");
        assert_eq!(rows.rows, 2);
        assert_eq!(rows.row_size, 2);
        assert!((rows.min_cosine - core::f64::consts::FRAC_1_SQRT_2).abs() < 1e-12);
    }

    #[test]
    fn parse_compare_logits_opts_validates_expected_rows() {
        let opts = parse_compare_logits_opts(vec![
            "--actual".to_owned(),
            "det.bin".to_owned(),
            "--reference".to_owned(),
            "ref.bin".to_owned(),
            "--row-size".to_owned(),
            "256".to_owned(),
            "--rows".to_owned(),
            "6".to_owned(),
        ])
        .expect("compare-logits options");
        assert_eq!(opts.row_size, Some(256));
        assert_eq!(opts.rows, Some(6));

        let err = parse_compare_logits_opts(vec![
            "--actual".to_owned(),
            "det.bin".to_owned(),
            "--reference".to_owned(),
            "ref.bin".to_owned(),
            "--rows".to_owned(),
            "6".to_owned(),
        ])
        .expect_err("rows without row-size should fail");
        assert_eq!(err, "compare-logits: --rows requires --row-size");

        let err = parse_compare_logits_opts(vec![
            "--actual".to_owned(),
            "det.bin".to_owned(),
            "--reference".to_owned(),
            "ref.bin".to_owned(),
            "--row-size".to_owned(),
            "256".to_owned(),
            "--rows".to_owned(),
            "0".to_owned(),
        ])
        .expect_err("zero rows should fail");
        assert_eq!(err, "compare-logits: --rows must be greater than zero");
    }

    #[test]
    fn compare_logits_expected_rows_rejects_shape_mismatch() {
        let rows = LogitsRowMetrics {
            rows: 2,
            row_size: 256,
            min_cosine: 1.0,
        };
        validate_expected_rows(&rows, Some(2)).expect("matching rows");

        let err = validate_expected_rows(&rows, Some(3)).expect_err("row mismatch should fail");
        assert_eq!(err, "compare-logits: row count 2 does not match expected 3");
    }

    #[test]
    fn compare_logits_values_rejects_bad_shapes_and_zero_norms() {
        assert!(compare_logits_values(&[1.0], &[1.0, 2.0]).is_err());
        assert!(compare_logits_values(&[0.0], &[1.0]).is_err());
        assert!(compare_logits_values(&[1.0], &[0.0]).is_err());
        assert!(compare_logits_rows(&[1.0, 2.0, 3.0], &[1.0, 2.0, 3.0], 2).is_err());
    }

    #[test]
    fn parse_bench_file_opts_accepts_limit_bytes_and_rejects_zero() {
        let opts = parse_bench_file_opts(vec![
            "--model".to_owned(),
            "model.gguf".to_owned(),
            "--input".to_owned(),
            "enwik8".to_owned(),
            "--limit-bytes".to_owned(),
            "1048576".to_owned(),
            "--n-ctx".to_owned(),
            "2048".to_owned(),
            "--iters".to_owned(),
            "2".to_owned(),
        ])
        .expect("bench-file options");
        assert_eq!(opts.model, "model.gguf");
        assert_eq!(opts.input, "enwik8");
        assert_eq!(opts.limit_bytes, Some(1_048_576));
        assert_eq!(opts.n_ctx, Some(2048));
        assert_eq!(opts.iters, 2);

        let err = parse_bench_file_opts(vec![
            "--model".to_owned(),
            "model.gguf".to_owned(),
            "--input".to_owned(),
            "enwik8".to_owned(),
            "--limit-bytes".to_owned(),
            "0".to_owned(),
        ])
        .expect_err("zero limit must be rejected");
        assert_eq!(err, "bench-file: --limit-bytes must be greater than zero");
    }

    #[test]
    fn model_info_reports_fixture_config_and_tensor_status() {
        let model_bytes = tiny_f32_gguf();
        let text = model_info_text("testdata/tiny-f32.gguf", &model_bytes).expect("model info");
        assert!(text.contains("model-info path=testdata/tiny-f32.gguf"));
        assert!(text
            .contains("sha256=ce2aa01900a63585a409ef995a2827dcac81e1678e38a1ab0733302ba82ce79b"));
        assert!(text.contains("model-info tokenizer status=ok kind=byte_fallback"));
        assert!(text.contains("model-info config status=ok block_count=1 embedding_length=4"));
        assert!(text.contains("model-info tensor-inventory total=12"));
        assert!(text.contains("F32=12"));
        assert!(text.contains(
            "model-info vocab status=ok tokenizer=256 model=256 codec_max_symbols=262144"
        ));
        assert!(text.contains(
            "model-info required-tensors status=ok checked=12 missing=0 shape_mismatch=0 unsupported_type=0 tied_output=false"
        ));
    }

    #[test]
    fn parse_model_info_opts_requires_model() {
        let opts = parse_model_info_opts(vec!["--model".to_owned(), "model.gguf".to_owned()])
            .expect("model-info options");
        assert_eq!(opts.model, "model.gguf");

        let err = parse_model_info_opts(Vec::new()).expect_err("missing model should fail");
        assert_eq!(err, "model-info: missing --model");
    }

    #[test]
    fn bench_vocab_validation_matches_codec_limits() {
        validate_vocab_lengths(det_coder::MAX_SYMBOLS, det_coder::MAX_SYMBOLS)
            .expect("design maximum is accepted");

        let mismatch = validate_vocab_lengths(255, 256).expect_err("mismatch");
        assert!(
            mismatch
                .contains("tokenizer vocabulary length 255 does not match model vocabulary 256"),
            "{mismatch}"
        );

        let too_large =
            validate_vocab_lengths(det_coder::MAX_SYMBOLS + 1, det_coder::MAX_SYMBOLS + 1)
                .expect_err("too large");
        assert!(
            too_large.contains("model vocabulary 262145 exceeds codec symbol limit 262144"),
            "{too_large}"
        );
    }

    #[test]
    fn xtask_codec_helpers_reject_invalid_windows() {
        let model_bytes = tiny_f32_gguf();
        let gguf = det_gguf::parse(&model_bytes).expect("gguf");
        let model = det_model::F32Llama::from_gguf(&gguf, &model_bytes).expect("model");
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
    fn determinism_scan_reports_banned_patterns_and_allows_marked_lines() {
        let mut violations = Vec::new();
        let banned = concat!("f32::", "exp");
        let text = format!("let _ = {banned}(1.0);\nlet _ = {banned}(1.0); // determinism-allow\n");
        scan_determinism_text(Path::new("sample.rs"), &text, &mut violations);
        assert_eq!(violations.len(), 1);
        assert!(violations[0].contains("sample.rs:1"));
        assert!(violations[0].contains(banned));
    }

    #[test]
    fn dependency_policy_requires_exact_external_versions() {
        let mut violations = Vec::new();
        let text = r#"
[package]
version = "0.1.0"

[dependencies]
local = { path = "../local" }
exact = "=1.2.3"
exact_table = { version = "=4.5.6", default-features = false }
floating = "1.0"
floating_table = { version = "2.0" }
"#;
        scan_dependency_policy_text(Path::new("Cargo.toml"), text, &mut violations);
        assert_eq!(violations.len(), 2);
        assert!(violations[0].contains("Cargo.toml:9"), "{violations:?}");
        assert!(violations[1].contains("Cargo.toml:10"), "{violations:?}");
    }

    #[test]
    fn parse_labeled_logits_hashes_requires_expected_labels_and_hex() {
        let ok = parse_labeled_logits_hashes(
            "tiny-qmix 2222222222222222222222222222222222222222222222222222222222222222\n\
             tiny-f32 1111111111111111111111111111111111111111111111111111111111111111\n",
            "artifact",
        )
        .expect("hashes");
        assert_eq!(ok[0].0, "tiny-f32");
        assert_eq!(ok[1].0, "tiny-qmix");

        let missing = parse_labeled_logits_hashes(
            "tiny-f32 1111111111111111111111111111111111111111111111111111111111111111\n",
            "artifact",
        )
        .expect_err("missing fixture");
        assert!(
            missing.contains("expected 2 fixture hashes, found 1"),
            "{missing}"
        );

        let bad_hash = parse_labeled_logits_hashes(
            "tiny-f32 1111111111111111111111111111111111111111111111111111111111111111\n\
             tiny-qmix Z222222222222222222222222222222222222222222222222222222222222222\n",
            "artifact",
        )
        .expect_err("bad hash");
        assert!(bad_hash.contains("invalid SHA-256 hex"), "{bad_hash}");
    }

    #[test]
    fn verify_logits_hashes_checks_count_and_exact_artifact_match() {
        let dir = unique_tmp_dir();
        let a = dir.join("a");
        let b = dir.join("b");
        fs::create_dir_all(&a).expect("mkdir a");
        fs::create_dir_all(&b).expect("mkdir b");

        let artifact = "tiny-f32 1111111111111111111111111111111111111111111111111111111111111111\n\
                        tiny-qmix 2222222222222222222222222222222222222222222222222222222222222222\n";
        fs::write(a.join("logits-hashes.txt"), artifact).expect("write a");
        fs::write(b.join("logits-hashes.txt"), artifact).expect("write b");

        verify_logits_hashes(VerifyLogitsHashesOpts {
            dir: dir.to_string_lossy().into_owned(),
            expected_count: 2,
        })
        .expect("matching artifacts");

        let wrong_count = verify_logits_hashes(VerifyLogitsHashesOpts {
            dir: dir.to_string_lossy().into_owned(),
            expected_count: 3,
        })
        .expect_err("wrong count");
        assert!(
            wrong_count.contains("expected 3 hash artifacts, found 2"),
            "{wrong_count}"
        );

        fs::write(
            b.join("logits-hashes.txt"),
            "tiny-f32 1111111111111111111111111111111111111111111111111111111111111111\n\
             tiny-qmix 3333333333333333333333333333333333333333333333333333333333333333\n",
        )
        .expect("write mismatch");
        let mismatch = verify_logits_hashes(VerifyLogitsHashesOpts {
            dir: dir.to_string_lossy().into_owned(),
            expected_count: 2,
        })
        .expect_err("mismatch");
        assert!(mismatch.contains("does not match reference"), "{mismatch}");

        let _ = fs::remove_dir_all(dir);
    }

    #[test]
    fn ci_workflow_check_requires_cross_platform_hash_gate() {
        let valid = valid_ci_workflow_text();
        validate_ci_workflow_text(valid).expect("valid workflow");

        let missing_wasm = valid.replace(
            "wasmtime --dir . target/wasm32-wasip1/debug/detllm.wasm logits",
            "wasmtime target/wasm32-wasip1/debug/detllm.wasm version",
        );
        let err = validate_ci_workflow_text(&missing_wasm).expect_err("missing wasm logits");
        assert!(err.contains("wasm logits execution"), "{err}");

        let missing_artifact = valid.replacen("uses: actions/upload-artifact@v4", "", 1);
        let err = validate_ci_workflow_text(&missing_artifact).expect_err("missing upload");
        assert!(
            err.contains("must upload exactly three logits artifact groups"),
            "{err}"
        );

        let bad_runner_context = valid.replace(
            "XDG_CACHE_HOME: /tmp/detllm-wasmtime-cache",
            "XDG_CACHE_HOME: ${{ runner.temp }}/wasmtime-cache",
        );
        let err = validate_ci_workflow_text(&bad_runner_context).expect_err("bad runner context");
        assert!(
            err.contains("must not use runner context in job-level env"),
            "{err}"
        );
    }

    fn valid_ci_workflow_text() -> &'static str {
        r#"
on:
  workflow_dispatch:
jobs:
  hygiene:
    steps:
      - run: cargo run -p xtask -- check-ci-workflow
  test:
    strategy:
      matrix:
        include:
          - name: x86_64-linux
          - name: aarch64-macos
          - name: aarch64-linux
    steps:
      - run: cargo run -p det-cli -- logits -m testdata/tiny-f32.gguf --tokens "$(cat testdata/tiny.tokens.txt)" --hash --chunk-size 3
      - uses: actions/upload-artifact@v4
        with:
          name: logits-hashes-${{ matrix.name }}
  logits-hash-match:
    needs: [test, toolchain-skew, wasm]
    steps:
      - run: cargo run -p xtask -- verify-logits-hashes --dir logits-hashes --expected-count 6
  msrv:
    steps: []
  toolchain-skew:
    strategy:
      matrix:
        toolchain: [stable, "1.94.0"]
    steps:
      - run: cargo run -p det-cli -- logits -m testdata/tiny-f32.gguf --tokens "$(cat testdata/tiny.tokens.txt)" --hash --chunk-size 3
      - uses: actions/upload-artifact@v4
        with:
          name: logits-hashes-toolchain-${{ matrix.toolchain }}
  wasm:
    env:
      XDG_CACHE_HOME: /tmp/detllm-wasmtime-cache
    steps:
      - run: cargo build --workspace --target wasm32-wasip1
      - run: wasmtime target/wasm32-wasip1/debug/detllm.wasm selftest
      - run: cargo run -p det-cli -- logits -m testdata/tiny-f32.gguf --tokens "$(cat testdata/tiny.tokens.txt)" --hash --chunk-size 3
      - run: wasmtime --dir . target/wasm32-wasip1/debug/detllm.wasm logits -m testdata/tiny-f32.gguf --tokens "$(cat testdata/tiny.tokens.txt)" --hash --chunk-size 3
      - run: wasmtime --dir . target/wasm32-wasip1/debug/detllm.wasm compress -m testdata/tiny-f32.gguf -i testdata/tiny.tokens.txt -o wasm-codec-smoke/tiny-f32.dtlz --n-ctx 8
      - run: wasmtime --dir . target/wasm32-wasip1/debug/detllm.wasm decompress -m testdata/tiny-f32.gguf -i wasm-codec-smoke/tiny-f32.dtlz -o wasm-codec-smoke/tiny-f32.restored
      - run: cmp native-quant-kernel-hash.txt wasm-quant-kernel-hash.txt
      - uses: actions/upload-artifact@v4
        with:
          name: logits-hashes-wasm32-wasip1
"#
    }

    fn unique_tmp_dir() -> PathBuf {
        let mut dir = std::env::temp_dir();
        dir.push(format!(
            "detllm-xtask-test-{}-{}",
            std::process::id(),
            std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .expect("time")
                .as_nanos()
        ));
        dir
    }
}
