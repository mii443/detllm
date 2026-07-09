use std::{
    fs,
    path::{Path, PathBuf},
    process::Command,
    sync::atomic::{AtomicU64, Ordering},
};

use det_num::Sha256;

fn workspace_root() -> PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR"))
        .parent()
        .expect("crates directory")
        .parent()
        .expect("workspace root")
        .to_path_buf()
}

fn detllm() -> &'static str {
    env!("CARGO_BIN_EXE_detllm")
}

#[test]
fn logits_hash_matches_testdata_golden_through_cli() {
    let root = workspace_root();
    let tokens = fs::read_to_string(root.join("testdata/tiny.tokens.txt")).expect("tokens fixture");

    for (model, hash) in [
        ("testdata/tiny-f32.gguf", "testdata/tiny-f32.logits.sha256"),
        (
            "testdata/tiny-qmix.gguf",
            "testdata/tiny-qmix.logits.sha256",
        ),
    ] {
        let expected = fs::read_to_string(root.join(hash)).expect("golden hash");
        for threads in ["1", "2", "7", "16"] {
            for chunk_size in ["1", "2", "3", "6"] {
                let output = Command::new(detllm())
                    .current_dir(&root)
                    .args([
                        "logits",
                        "-m",
                        model,
                        "--tokens",
                        tokens.trim(),
                        "--hash",
                        "--threads",
                        threads,
                        "--chunk-size",
                        chunk_size,
                    ])
                    .output()
                    .expect("run detllm logits");

                assert!(
                    output.status.success(),
                    "detllm logits {model} --threads {threads} --chunk-size {chunk_size} failed: {}",
                    String::from_utf8_lossy(&output.stderr)
                );
                assert_eq!(
                    String::from_utf8(output.stdout).expect("utf8 stdout"),
                    expected
                );
            }
        }
    }
}

#[test]
fn logits_prompt_path_matches_explicit_byte_tokens() {
    let root = workspace_root();

    for model in ["testdata/tiny-f32.gguf", "testdata/tiny-qmix.gguf"] {
        let tokenize_output = Command::new(detllm())
            .current_dir(&root)
            .args(["tokenize", "-m", model, "-p", "abc\n"])
            .output()
            .expect("run detllm tokenize");
        assert!(
            tokenize_output.status.success(),
            "detllm tokenize {model} failed: {}",
            String::from_utf8_lossy(&tokenize_output.stderr)
        );
        let tokenized = String::from_utf8(tokenize_output.stdout).expect("utf8 tokens");
        assert_eq!(tokenized, "97,98,99,10\n");

        let token_output = Command::new(detllm())
            .current_dir(&root)
            .args([
                "logits",
                "-m",
                model,
                "--tokens",
                tokenized.trim(),
                "--hash",
                "--chunk-size",
                "2",
            ])
            .output()
            .expect("run detllm logits --tokens");
        assert!(
            token_output.status.success(),
            "detllm logits --tokens {model} failed: {}",
            String::from_utf8_lossy(&token_output.stderr)
        );

        let prompt_output = Command::new(detllm())
            .current_dir(&root)
            .args([
                "logits",
                "-m",
                model,
                "-p",
                "abc\n",
                "--hash",
                "--chunk-size",
                "2",
            ])
            .output()
            .expect("run detllm logits -p");
        assert!(
            prompt_output.status.success(),
            "detllm logits -p {model} failed: {}",
            String::from_utf8_lossy(&prompt_output.stderr)
        );

        assert_eq!(prompt_output.stdout, token_output.stdout);
    }
}

#[test]
fn logits_dump_matches_hash_stream() {
    let root = workspace_root();
    let dir = unique_tmp_dir();
    fs::create_dir_all(&dir).expect("mkdir");
    let dump_path = dir.join("tiny-f32.logits.bin");
    let tokens = fs::read_to_string(root.join("testdata/tiny.tokens.txt")).expect("tokens fixture");

    let output = Command::new(detllm())
        .current_dir(&root)
        .args([
            "logits",
            "-m",
            "testdata/tiny-f32.gguf",
            "--tokens",
            tokens.trim(),
            "--hash",
            "--chunk-size",
            "3",
            "--dump",
            dump_path.to_str().expect("dump path"),
        ])
        .output()
        .expect("run detllm logits --dump");

    assert!(
        output.status.success(),
        "detllm logits --dump failed: {}",
        String::from_utf8_lossy(&output.stderr)
    );
    let dumped = fs::read(&dump_path).expect("dumped logits");
    let token_count = tokens.trim().split(',').count();
    let model_bytes = fs::read(root.join("testdata/tiny-f32.gguf")).expect("model bytes");
    let gguf = det_gguf::parse(&model_bytes).expect("parse model");
    let model = det_model::F32Llama::from_gguf(&gguf, &model_bytes).expect("load model");
    assert_eq!(dumped.len(), token_count * model.output.rows() * 4);

    let mut hash = Sha256::new();
    hash.update(&dumped);
    assert_eq!(
        String::from_utf8(output.stdout).expect("stdout"),
        format!("{}\n", hex(&hash.finalize()))
    );

    let _ = fs::remove_dir_all(dir);
}

#[test]
fn gguf_dump_lists_metadata_and_all_testdata_tensors() {
    let root = workspace_root();
    let output = Command::new(detllm())
        .current_dir(&root)
        .args(["gguf-dump", "testdata/tiny-qmix.gguf"])
        .output()
        .expect("run detllm gguf-dump");

    assert!(
        output.status.success(),
        "detllm gguf-dump failed: {}",
        String::from_utf8_lossy(&output.stderr)
    );
    let stdout = String::from_utf8(output.stdout).expect("utf8 stdout");
    assert!(stdout.contains("metadata general.architecture string=llama"));
    assert!(stdout.contains("metadata tokenizer.ggml.tokens array<string>[256]"));
    assert!(stdout.contains("tensor token_embd.weight [32, 256] type=8 offset=0"));
    assert!(stdout.contains("tensor output_norm.weight [32] type=0"));
}

#[test]
fn compress_decompress_round_trips_testdata_model_through_cli() {
    let root = workspace_root();
    let dir = unique_tmp_dir();
    fs::create_dir_all(&dir).expect("mkdir");
    let cases = [
        ("empty", Vec::new()),
        (
            "multilingual",
            "detllm deterministic 圧縮 smoke\n復元一致\n"
                .as_bytes()
                .to_vec(),
        ),
        (
            "binary-multi-window",
            (0..=255u8).chain(b"detllm".iter().copied()).collect(),
        ),
    ];

    for (label, model) in [
        ("f32", "testdata/tiny-f32.gguf"),
        ("qmix", "testdata/tiny-qmix.gguf"),
    ] {
        for (case, input) in &cases {
            let input_path = dir.join(format!("{label}.{case}.input.bin"));
            let compressed_path = dir.join(format!("{label}.{case}.dtlz"));
            let restored_path = dir.join(format!("{label}.{case}.restored.bin"));
            fs::write(&input_path, input).expect("write input");

            let compress = Command::new(detllm())
                .current_dir(&root)
                .args([
                    "compress",
                    "-m",
                    model,
                    "-i",
                    input_path.to_str().expect("input path"),
                    "-o",
                    compressed_path.to_str().expect("compressed path"),
                    "--n-ctx",
                    "8",
                    "--threads",
                    "3",
                ])
                .output()
                .expect("run detllm compress");
            assert!(
                compress.status.success(),
                "detllm compress {label}/{case} failed: {}",
                String::from_utf8_lossy(&compress.stderr)
            );

            let decompress = Command::new(detllm())
                .current_dir(&root)
                .args([
                    "decompress",
                    "-m",
                    model,
                    "-i",
                    compressed_path.to_str().expect("compressed path"),
                    "-o",
                    restored_path.to_str().expect("restored path"),
                    "--threads",
                    "3",
                ])
                .output()
                .expect("run detllm decompress");
            assert!(
                decompress.status.success(),
                "detllm decompress {label}/{case} failed: {}",
                String::from_utf8_lossy(&decompress.stderr)
            );

            assert_eq!(
                fs::read(&restored_path).expect("restored").as_slice(),
                input.as_slice()
            );
        }
    }
    let _ = fs::remove_dir_all(dir);
}

#[test]
fn decompress_rejects_noncanonical_payload_with_trailing_bytes() {
    let root = workspace_root();
    let dir = unique_tmp_dir();
    fs::create_dir_all(&dir).expect("mkdir");

    let input_path = dir.join("input.bin");
    let compressed_path = dir.join("compressed.dtlz");
    let restored_path = dir.join("restored.bin");
    fs::write(&input_path, b"canonical payload check").expect("write input");

    let compress = Command::new(detllm())
        .current_dir(&root)
        .args([
            "compress",
            "-m",
            "testdata/tiny-f32.gguf",
            "-i",
            input_path.to_str().expect("input path"),
            "-o",
            compressed_path.to_str().expect("compressed path"),
            "--n-ctx",
            "8",
        ])
        .output()
        .expect("run detllm compress");
    assert!(
        compress.status.success(),
        "detllm compress failed: {}",
        String::from_utf8_lossy(&compress.stderr)
    );

    let mut tampered = fs::read(&compressed_path).expect("compressed");
    tampered.extend_from_slice(&[0xde, 0xad, 0xbe, 0xef]);
    fs::write(&compressed_path, tampered).expect("write tampered");

    let decompress = Command::new(detllm())
        .current_dir(&root)
        .args([
            "decompress",
            "-m",
            "testdata/tiny-f32.gguf",
            "-i",
            compressed_path.to_str().expect("compressed path"),
            "-o",
            restored_path.to_str().expect("restored path"),
        ])
        .output()
        .expect("run detllm decompress");
    assert!(
        !decompress.status.success(),
        "detllm decompress unexpectedly accepted trailing payload"
    );
    assert!(
        String::from_utf8_lossy(&decompress.stderr).contains("canonical encoding"),
        "unexpected stderr: {}",
        String::from_utf8_lossy(&decompress.stderr)
    );

    let _ = fs::remove_dir_all(dir);
}

fn unique_tmp_dir() -> PathBuf {
    static NEXT_TMP_ID: AtomicU64 = AtomicU64::new(0);
    let mut dir = std::env::temp_dir();
    dir.push(format!(
        "detllm-cli-test-{}-{}-{}",
        std::process::id(),
        NEXT_TMP_ID.fetch_add(1, Ordering::Relaxed),
        std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .expect("time")
            .as_nanos()
    ));
    dir
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
