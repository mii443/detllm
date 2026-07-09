// Write llama.cpp C API logits in detllm dump format.
//
// Build example:
//   c++ -std=c++17 -O2 -I/usr/local/include scripts/reference_logits_llamacpp.cpp \
//       -L/usr/local/lib -Wl,-rpath,/usr/local/lib -lllama -lggml -lggml-cpu -lggml-base \
//       -o /tmp/reference_logits_llamacpp
//
// Usage example:
//   /tmp/reference_logits_llamacpp --model model.gguf --tokens 1,2,3 --out llama.logits.bin --quiet
//
// The output is row-major little-endian f32:
//   position 0 vocab logits | position 1 vocab logits | ...

#include "llama.h"

#include <cerrno>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <iostream>
#include <limits>
#include <sstream>
#include <string>
#include <vector>

namespace {

struct Options {
    std::string model;
    std::string tokens;
    std::string out;
    int threads = 1;
    int ctx_size = 0;
    int batch_size = 0;
    int expected_vocab = 0;
    int expected_rows = 0;
    bool quiet = false;
};

[[noreturn]] void fail(const std::string & message) {
    std::cerr << "reference_logits_llamacpp: " << message << "\n";
    std::exit(1);
}

[[noreturn]] void usage() {
    fail("usage: reference_logits_llamacpp --model model.gguf --tokens 1,2,3 --out logits.bin "
         "[--threads N] [--ctx-size N] [--batch-size N] [--expected-vocab N] [--expected-rows N] "
         "[--quiet]");
}

void log_callback(ggml_log_level level, const char * text, void * user_data) {
    const bool quiet = user_data != nullptr && *static_cast<bool *>(user_data);
    if (!quiet || level == GGML_LOG_LEVEL_ERROR) {
        std::fputs(text, stderr);
    }
}

int parse_positive_int(const std::string & value, const char * flag) {
    char * end = nullptr;
    errno = 0;
    long parsed = std::strtol(value.c_str(), &end, 10);
    if (errno != 0 || end == value.c_str() || *end != '\0' || parsed <= 0 ||
        parsed > std::numeric_limits<int>::max()) {
        fail(std::string("invalid ") + flag + " value: " + value);
    }
    return static_cast<int>(parsed);
}

Options parse_options(int argc, char ** argv) {
    Options opts;
    for (int i = 1; i < argc; ++i) {
        std::string arg = argv[i];
        auto value = [&](const char * flag) -> std::string {
            if (++i >= argc) {
                fail(std::string("missing value for ") + flag);
            }
            return argv[i];
        };
        if (arg == "--model" || arg == "-m") {
            opts.model = value(arg.c_str());
        } else if (arg == "--tokens") {
            opts.tokens = value(arg.c_str());
        } else if (arg == "--out" || arg == "-o") {
            opts.out = value(arg.c_str());
        } else if (arg == "--threads") {
            opts.threads = parse_positive_int(value(arg.c_str()), "--threads");
        } else if (arg == "--ctx-size") {
            opts.ctx_size = parse_positive_int(value(arg.c_str()), "--ctx-size");
        } else if (arg == "--batch-size") {
            opts.batch_size = parse_positive_int(value(arg.c_str()), "--batch-size");
        } else if (arg == "--expected-vocab") {
            opts.expected_vocab = parse_positive_int(value(arg.c_str()), "--expected-vocab");
        } else if (arg == "--expected-rows") {
            opts.expected_rows = parse_positive_int(value(arg.c_str()), "--expected-rows");
        } else if (arg == "--quiet") {
            opts.quiet = true;
        } else if (arg == "--help" || arg == "-h") {
            usage();
        } else {
            fail("unknown argument: " + arg);
        }
    }
    if (opts.model.empty() || opts.tokens.empty() || opts.out.empty()) {
        usage();
    }
    return opts;
}

std::vector<llama_token> parse_tokens(const std::string & raw) {
    std::vector<llama_token> tokens;
    std::stringstream ss(raw);
    std::string part;
    while (std::getline(ss, part, ',')) {
        if (part.empty()) {
            fail("empty token id in --tokens");
        }
        char * end = nullptr;
        errno = 0;
        long parsed = std::strtol(part.c_str(), &end, 10);
        if (errno != 0 || end == part.c_str() || *end != '\0' || parsed < 0 ||
            parsed > std::numeric_limits<llama_token>::max()) {
            fail("invalid token id: " + part);
        }
        tokens.push_back(static_cast<llama_token>(parsed));
    }
    if (tokens.empty()) {
        fail("--tokens must not be empty");
    }
    return tokens;
}

void write_f32_dump(const std::string & path, const float * logits, size_t count) {
    std::ofstream out(path, std::ios::binary);
    if (!out) {
        fail("failed to open output file: " + path);
    }
    out.write(reinterpret_cast<const char *>(logits), static_cast<std::streamsize>(count * sizeof(float)));
    if (!out) {
        fail("failed to write output file: " + path);
    }
}

} // namespace

int main(int argc, char ** argv) {
    static_assert(sizeof(float) == 4, "f32 output requires 32-bit float");
#if __BYTE_ORDER__ != __ORDER_LITTLE_ENDIAN__
#error "reference_logits_llamacpp writes host-order floats and requires little-endian host"
#endif

    const Options opts = parse_options(argc, argv);
    const std::vector<llama_token> tokens = parse_tokens(opts.tokens);
    if (opts.expected_rows != 0 && opts.expected_rows != static_cast<int>(tokens.size())) {
        fail("--expected-rows does not match --tokens count");
    }

    bool quiet = opts.quiet;
    llama_log_set(log_callback, &quiet);
    llama_backend_init();

    llama_model_params model_params = llama_model_default_params();
    model_params.n_gpu_layers = 0;
    model_params.use_mmap = true;
    llama_model * model = llama_model_load_from_file(opts.model.c_str(), model_params);
    if (model == nullptr) {
        llama_backend_free();
        fail("failed to load model: " + opts.model);
    }

    const llama_vocab * vocab = llama_model_get_vocab(model);
    const int n_vocab = llama_vocab_n_tokens(vocab);
    if (opts.expected_vocab != 0 && opts.expected_vocab != n_vocab) {
        llama_model_free(model);
        llama_backend_free();
        fail("--expected-vocab does not match llama.cpp vocabulary size");
    }

    llama_context_params ctx_params = llama_context_default_params();
    ctx_params.n_ctx = static_cast<uint32_t>(opts.ctx_size == 0 ? tokens.size() : opts.ctx_size);
    ctx_params.n_batch = static_cast<uint32_t>(opts.batch_size == 0 ? tokens.size() : opts.batch_size);
    ctx_params.n_ubatch = ctx_params.n_batch;
    ctx_params.n_seq_max = 1;
    ctx_params.n_threads = opts.threads;
    ctx_params.n_threads_batch = opts.threads;
    ctx_params.no_perf = true;
    if (ctx_params.n_ctx < tokens.size()) {
        llama_model_free(model);
        llama_backend_free();
        fail("--ctx-size is smaller than token count");
    }

    llama_context * ctx = llama_init_from_model(model, ctx_params);
    if (ctx == nullptr) {
        llama_model_free(model);
        llama_backend_free();
        fail("failed to create llama context");
    }

    llama_batch batch = llama_batch_init(static_cast<int32_t>(tokens.size()), 0, 1);
    for (size_t i = 0; i < tokens.size(); ++i) {
        batch.token[i] = tokens[i];
        batch.pos[i] = static_cast<llama_pos>(i);
        batch.n_seq_id[i] = 1;
        batch.seq_id[i][0] = 0;
        batch.logits[i] = 1;
    }
    batch.n_tokens = static_cast<int32_t>(tokens.size());

    const int rc = llama_decode(ctx, batch);
    if (rc != 0) {
        llama_batch_free(batch);
        llama_free(ctx);
        llama_model_free(model);
        llama_backend_free();
        fail("llama_decode failed with code " + std::to_string(rc));
    }

    const float * logits = llama_get_logits(ctx);
    if (logits == nullptr) {
        llama_batch_free(batch);
        llama_free(ctx);
        llama_model_free(model);
        llama_backend_free();
        fail("llama_get_logits returned null");
    }
    write_f32_dump(opts.out, logits, tokens.size() * static_cast<size_t>(n_vocab));
    std::cout << "reference_logits_llamacpp rows=" << tokens.size() << " vocab=" << n_vocab
              << " values=" << tokens.size() * static_cast<size_t>(n_vocab) << "\n";

    llama_batch_free(batch);
    llama_free(ctx);
    llama_model_free(model);
    llama_backend_free();
    return 0;
}
