---

# detllm — 決定的LLM推論エンジン 設計書 v1.0

**目的**: LLMを算術符号化の確率モデルとして使うロスレス圧縮ツール(llama-zip代替)のために、異なる物理環境(CPUアーキテクチャ・OS・ビルド)間で **bit完全に一致するlogits/CDF** を生成するRust製推論エンジンをスクラッチ実装する。

本書は規範的仕様(normative spec)である。「MUST/MUST NOT/SHOULD」はRFC 2119の意味で用いる。コーディングエージェントは本書の数値仕様から逸脱してはならない。逸脱が必要な場合は仕様の改訂として明示的に記録すること。

---

## 0. ゴールと非ゴール

### ゴール
- G1: 同一のGGUFモデルと同一の入力トークン列に対し、**全対応プラットフォームで全トークン位置のlogitsがbit一致**すること。決定性は性能より常に優先する。
- G2: Llama 2/3系の密なdecoder-only Transformer(RMSNorm, RoPE, GQA, SwiGLU)のCPU推論。
- G3: GGUF形式の `F32` / `Q8_0` / `Q4_0` テンソルの読み込みと推論。
- G4: 算術符号化(range coder)との接合。logits → 整数CDF → 符号化/復号のパイプラインまで含む。
- G5: ターゲット: `x86_64-unknown-linux-gnu`, `aarch64-apple-darwin`, `aarch64-unknown-linux-gnu`, `wasm32-wasip1`(wasmtime実行)。
- G6: 圧縮側(教師強制で全トークンを評価)と復元側(1トークンずつ生成)が**チャンク分割の仕方に依存せず**同一のlogitsを得ること(position invariance)。

### 非ゴール
- GPU対応(将来課題。設計はCPU専用でよい)。
- llama.cppとのbit一致(不要。自己一致のみが要件)。
- MoE、Gemma系soft-capping、sliding window attention。
- 学習・ファインチューニング。
- リアルタイムチャット速度(圧縮ツールなので数tok/sでも許容)。

---

## 1. 決定性モデル(最重要セクション)

### 1.1 決定性不変条件(DETシリーズ)

- **DET-1(値の決定性)**: 全ての浮動小数点演算は IEEE 754-2008 binary32/binary64、丸めモード roundTiesToEven、subnormal有効(FTZ/DAZ無効)で行う。使用してよいFP演算は次の「正しく丸められる演算」のみ: 加算・減算・乗算・除算・`sqrt`・比較・`abs`/`neg`/`copysign`・整数⇔浮動小数点変換・f16→f32変換。これらはIEEE 754が正しい丸めを要求するため、準拠ハードウェア間でbit一致する。
- **DET-2(順序の決定性)**: 全ての浮動小数点縮約(総和・内積)は、§2で定義する**抽象縮約木**に従う。縮約順序は入力長のみの関数であり、スレッド数・SIMD幅・バッチサイズ・チャンク分割に依存してはならない(MUST NOT)。
- **DET-3(超越関数)**: `exp`/`sin`/`cos`/`ln` 等は、プラットフォームlibmを一切呼ばず、§4で指定する固定アルゴリズムの純Rust実装のみを使う。
- **DET-4(NaN非発生)**: 数値パスはNaNを生成してはならない(MUST NOT)。Wasm仕様上、浮動小数点の唯一の非決定性はNaN結果のビットパターンであり、ネイティブでもx86/ARMでNaNペイロードが異なるため、NaN非発生を保証することでこの問題を回避する。debugビルドでは全カーネル出力に `debug_assert!(x.is_finite())` を挿入する。
- **DET-5(整数演算優先)**: 量子化重みとの内積は `i8×i8 → i32` 累積で行う。整数加算は結合的なので、この部分の縮約順序は任意でよい(SIMD最適化の自由度はここに集約する)。
- **DET-6(実行時canary)**: プロセス起動時に数値セルフテスト(§9.4)を実行し、既知入力に対するカーネル出力のハッシュが期待値と一致しない場合は即座にabortする。これによりFTZが設定された環境・壊れたビルド・非準拠FPUを実行前に検出する。

### 1.2 禁止事項(コーディングエージェント向けチェックリスト)

以下は**いかなる理由があっても使用禁止**(MUST NOT):

1. `f32::exp`, `f32::sin`, `f32::cos`, `f32::ln`, `f32::powf`, `f32::tanh` などの標準ライブラリ超越関数(LLVM intrinsic経由でプラットフォームlibmに落ち、OS/libcごとに結果が異なる)。`f32::sqrt` は例外的に許可(IEEEが正しい丸めを要求し、x86 `sqrtss` / AArch64 `fsqrt` / Wasm `f32.sqrt` すべて準拠)。
2. `f32::mul_add` / `f64::mul_add`(FMA)。融合積和は $\mathrm{RN}(a \cdot b + c)$ を1回で丸めるため、mul+addの2回丸めと結果が異なり、FMA非搭載ターゲット(non-relaxed Wasm SIMDにはFMAが存在しない)と一致しなくなる。
3. `-C target-feature` のランタイムディスパッチ(`is_x86_feature_detected!` 等)による**数値結果が変わりうる**コードパス切替。SIMDバックエンドの切替は許可するが、全バックエンドがbit一致することをテストで保証する(§9)。
4. fast-math系の一切: `-ffast-math` 相当のフラグ、`fadd_fast` 等のintrinsic、演算の再結合を仮定した最適化。Rust/LLVMはデフォルトでFP再結合もFP contraction(自動FMA化)も行わないので、デフォルトのまま触らないこと。
5. ハードウェア近似命令: `rsqrtss`/`rcpps`(x86)、`frsqrte`/`frecpe`(NEON)、Wasm relaxed-simd 全命令。Wasmビルドでは `relaxed-simd` を無効化したまま(target-featureに含めない)にすること。relaxed SIMDは仕様上非決定的である。
6. FP値のatomic累積、reduce系の並列ライブラリ関数(`rayon` の `.sum()` 等)。並列化は「独立な出力要素への分割」のみ(§6)。
7. `HashMap`/`HashSet` のイテレーション順序に依存するロジック(乱択ハッシュのため実行ごとに異なる)。順序が意味を持つ箇所は `Vec` / `BTreeMap` を使う。
8. 数値結果に影響するcrateのバージョン非固定。`Cargo.lock` をコミットし、数値に関わる依存(あれば)は `=x.y.z` で完全固定する。原則として**数値カーネルは依存ゼロ**で書く。
9. C/C++依存のリンク(BLAS、libmラッパ等)。MXCSRのFTZビットを設定するCランタイム初期化コードの混入を防ぐため、数値クレートは pure Rust とする。

### 1.3 許可事項・前提

- f16→f32変換: 全てのf16値はf32で正確に表現できるため、正しい変換実装同士は必ずbit一致する。自前のビット操作実装(§4.5)を使う(`half` crateへの依存も可だがバージョン固定)。
- `f32 as i32` / `f32 as u32`: Rustの`as`キャストは「ゼロ方向切り捨て+飽和」と言語仕様で定義されており決定的。ただし丸めが必要な箇所では§4.4の明示的丸め関数を使う。
- 自動ベクトル化: RustはFP再結合を行わないため、LLVMの自動ベクトル化は逐次FPループの縮約を勝手に並列化しない(fast-mathフラグがないとFP reductionはベクトル化されない)。したがってスカラ参照実装は安全。ただし依存しすぎず、順序をコード構造として明示する。
- コンパイラバージョン: 決定性は言語セマンティクスで保証されるべきで、特定rustcバージョンに依存してはならない。CIで2つ以上のtoolchainを回して確認する(§9.5)。`rust-toolchain.toml` はビルド再現性のために置くが、これは決定性の根拠にしない。

---

## 2. 抽象縮約木の仕様(normative)

### 2.1 f32縮約: 8レーン・インターリーブ縮約

長さ $n$ のf32列 $x_0, \dots, x_{n-1}$ の総和(および内積 $\sum_i x_i y_i$)は以下で**定義**する。論理レーン数 $L = 8$ は固定:

$$
a_k = \sum_{\substack{i \equiv k \pmod 8 \\ 0 \le i < n}} x_i \quad (k = 0, \dots, 7)
$$

各 $a_k$ 内の加算は $i$ 昇順の逐次。最終結合は固定の木:

$$
s = \big((a_0 + a_1) + (a_2 + a_3)\big) + \big((a_4 + a_5) + (a_6 + a_7)\big)
$$

**設計根拠**: この定義はSIMD実装と正確に同型になる。
- AVX2(8レーンf32): 1本の `__m256` にアキュムレータを置き、8要素ずつ垂直加算 → 仕様と同一。
- NEON / Wasm simd128(4レーン): 2本のベクトル(レーン0–3とレーン4–7)で8要素/イテレーション → 仕様と同一。
- AVX-512(16レーン): 1本のzmmで16要素を処理すると $i$ と $i+8$ が別レーンに入り仕様と**一致しない**。AVX-512実装は2本のymm(または1本のzmmを8レーン×2グループとして垂直に畳む)で実装しなければならない(MUST)。
- スカラ: `acc: [f32; 8]` の配列で逐次実行 → 仕様と同一。

端数($n \bmod 8$ 要素)はレーン $0..(n \bmod 8)$ に $i$ 昇順で加算する(上の定義から自動的にそうなる)。

参照実装(これが規範。全バックエンドはこれとbit一致すること):

```rust
/// Normative f32 dot product. All SIMD backends MUST be bit-identical to this.
pub fn dot_f32_ref(x: &[f32], y: &[f32]) -> f32 {
    assert_eq!(x.len(), y.len());
    let mut acc = [0.0f32; 8];
    for i in 0..x.len() {
        acc[i % 8] += x[i] * y[i]; // one rounding for mul, one for add; NO mul_add
    }
    ((acc[0] + acc[1]) + (acc[2] + acc[3])) + ((acc[4] + acc[5]) + (acc[6] + acc[7]))
}
```

### 2.2 量子化GEMV: ブロック逐次縮約

出力行 $y_r = \sum_{b=0}^{B-1} c_b$($B$ = Kブロック数、ブロックサイズ32)は $b$ 昇順の**逐次f32加算**で結合する。各ブロック寄与 $c_b$ の計算は§3.3で定義。$K = 4096$ でもブロック数は128なので逐次で十分速い(重み読み出しがボトルネック)。

### 2.3 順序が不要な演算(自由に並列化してよい)

- $\max$ / $\min$ / $\arg$系: 全順序に対して結果不変(NaN非発生が前提)。ただし同値タイの扱いが結果に影響する箇所(§5.6のCDF残差割当など)は「最小インデックス優先」と明示的に定義する。
- i32/i64/u64 の総和: 結合的。オーバーフローしない設計にした上で自由な順序でよい(Q8_0内積のSIMD化はここで稼ぐ)。

---

## 3. 量子化仕様

### 3.1 GGUF `Q8_0`(重み)

GGUF標準レイアウトに従う: ブロックサイズ32、各ブロックは `d: f16`(スケール)+ `qs: [i8; 32]`。逆量子化は $w_i = d \cdot q_i$。

### 3.2 活性化の量子化 `Q8A`(自前形式、メモリ上のみ)

Q8_0重みとの内積のため、活性ベクトルをブロック32で量子化する。参照品質を保つため、整数丸めはhalf-away-from-zeroで固定する。逆数 `1/d` の事前計算は使わず、除算結果を明示的に丸める:

```
for each block of 32 f32 values x[0..32]:
    amax = max_i |x[i]|                      // 順序不変
    if amax == 0.0 { d = 0.0; q[i] = 0 for all i }
    else {
        d = amax / 127.0                     // f32除算(正しい丸め)
        q[i] = clamp(round_half_away_i32(x[i] / d), -127, 127)  // f32除算 → 明示的丸め
    }
    store d as f32 (f16に落とさない)
```

`round_half_away_i32` は§4.4で定義。逆数 `1/d` を掛ける最適化は丸めが1回増えるため禁止(除算はハードウェアで十分速い)。

### 3.3 Q8_0 × Q8A 内積(1ブロック)

ブロック $b$ の寄与:

$$
c_b = \underbrace{(d^{(w)}_b \cdot d^{(a)}_b)}_{\text{f32乗算1回}} \cdot \; \mathrm{f32}\Big(\underbrace{\textstyle\sum_{i=0}^{31} q^{(w)}_i \, q^{(a)}_i}_{\text{i32厳密}}\Big)
$$

- 内側の $\sum q^{(w)}_i q^{(a)}_i$: $|q| \le 127$ なので積は $\le 16129$、32個の和は $\le 516{,}128 < 2^{31}$。i32で厳密、**任意の順序でよい**(DET-5)。SIMDでは `vpmaddubsw`+`vpmaddwd`(AVX2)、`sdot`(NEON)、`i16x8` 経由(Wasm simd128)など自由に実装してよい。ここが性能の主戦場。
- スケール積の順序は $(d^{(w)} \cdot d^{(a)})$ を先に丸め、次に整数和のf32変換値と乗算(f32変換は $|{\cdot}| < 2^{24}$ なので厳密)。
- ブロック間は§2.2の逐次f32加算。

### 3.4 `Q4_0`

GGUF標準: ブロック32、`d: f16` + `qs: [u8; 16]`(4bit×32、オフセット8)。$w_i = d\,(q_i - 8)$。内積はQ8Aとの間で $(q^{(w)}_i - 8) \cdot q^{(a)}_i$ をi32で厳密計算(値域は同様に安全)。以降はQ8_0と同一。

### 3.5 F32テンソル

RMSNorm重み・小さいテンソルはF32のまま保持し、§2.1の縮約で扱う。v1では「全テンソルF32」モードも必ず動くようにする(参照経路・デバッグ用)。

---

## 4. 決定的数学関数(`det-num` クレート)

### 4.1 方針

実行時に必要な超越関数は実質 `exp_f32` のみに絞り込む:
- RoPEの $\sin/\cos$ → **起動時テーブル事前計算**(§4.3)。実行時は表引き+乗算のみ。
- RMSNormの $1/\sqrt{x}$ → `1.0 / x.sqrt()`(正しく丸められた2演算の合成、決定的)。近似命令・Newton法は使わない。
- softmax → `exp_f32` を使用(§5.6)。

### 4.2 `exp_f32` の実装指針

pure Rustの固定アルゴリズムであれば正確さより「固定であること」が重要。実装は次のいずれか(推奨順):

1. **`libm` crate のexpf をベンダリング**: `libm`(MUSL移植の pure Rust、no_std)の `expf` ソースをバージョン固定でリポジトリ内に**コピー**(vendoring)する。crate参照ではなくソースコピーとするのは、将来のcrate更新で数値が変わる事故を構造的に排除するため。コピー元のcrate名・バージョン・ライセンス(MIT/Apache-2.0)をファイルヘッダに記録。
2. RLIBM / CORE-MATH の binary32 expf の移植(correctly rounded)。将来f32以外の表現に拡張する場合や、複数実装間の互換性を最初から確保したい場合はこちら。参考: RLIBM https://github.com/rutgers-apl/rlibm 、CORE-MATH https://core-math.gitlabpages.inria.fr/

いずれの場合も、実装内で `mul_add` を使っていないこと・プラットフォーム分岐がないことをレビューで確認する(MUST)。入力を $[-88.0, 0.0]$ にクランプしてから呼ぶ(softmaxでは $x - \max \le 0$ なので自然に満たす。下限クランプでアンダーフロー→subnormal→0の経路も固定コードで通る)。

### 4.3 RoPEテーブル

パラメータ: `rope_freq_base` $\theta$(GGUFメタデータ、Llama 2は10000、Llama 3は500000)、head_dim $d_h$、最大コンテキスト $N$。周波数 $\omega_j = \theta^{-2j/d_h}$($j = 0..d_h/2$)、角度 $\phi_{p,j} = p \cdot \omega_j$。

テーブル生成は**f64固定アルゴリズム**で行い、最後にf32へ丸める:
- $\omega_j = \exp(-\tfrac{2j}{d_h} \ln \theta)$ を、ベンダリングした `libm::exp`(f64)と `libm::log`(f64)で計算。
- $\cos(\phi), \sin(\phi)$ はベンダリングした `libm::cos` / `libm::sin`(f64)。引数還元まで含めて固定コードなので、$p \cdot \omega_j$ が大きくても環境間で一致する。
- `table_cos[p][j] = cos64(...) as f32`(f64→f32は正しい丸め)。

回転の適用(実行時、f32): $x'_{2j} = x_{2j} c - x_{2j+1} s$、$x'_{2j+1} = x_{2j} s + x_{2j+1} c$。演算順序はこの式の通り(各項の乗算→減算/加算、FMA禁止)。Llama系の "neox" スタイル(前半/後半ペアリング)かどうかはGGUFメタデータに従う。

### 4.4 明示的丸め関数

```rust
/// round-half-to-even, f32 -> i32. Deterministic on all targets.
pub fn round_ties_even_i32(x: f32) -> i32 {
    // f32でtiesToEvenの整数丸めを行う標準技法:
    // |x| < 2^23 の範囲で (x + C) - C, C = 2^23 を使うか、
    // x.round_ties_even()(Rust 1.77+で安定化、LLVM roundeven → 決定的)を使う。
    // ここでは後者を採用し、その後 `as i32` で変換する(丸め済みなので切り捨ては無害)。
    x.round_ties_even() as i32
}
```

`round_ties_even` はIEEEの roundToIntegralTiesToEven であり全ターゲットで一致する。Wasmでは `f32.nearest` に対応する。

Q8Aの量子化では、参照実装との誤差を抑えるためhalf-away-from-zeroを使う。入力は有限かつ `|x| <= 127` の範囲に正規化済みなので、次の演算列で決定的に定義する:

```rust
pub fn round_half_away_i32(x: f32) -> i32 {
    if x < 0.0 { (x - 0.5) as i32 } else { (x + 0.5) as i32 }
}
```

### 4.5 f16→f32

```rust
pub fn f16_to_f32(bits: u16) -> f32 { /* ビット操作による厳密変換。分岐: normal / subnormal / inf / nan */ }
```
GGUFのf16スケールにNaN/Infが含まれていた場合はモデルロード時にエラーとする(DET-4をロード時に強制)。

---

## 5. モデル実装仕様(`det-model` クレート)

### 5.1 対応アーキテクチャ

GGUF `general.architecture == "llama"` の密モデル。読み取るメタデータ:

| キー | 用途 |
|---|---|
| `llama.block_count` | 層数 $L$ |
| `llama.embedding_length` | $d_{model}$ |
| `llama.feed_forward_length` | $d_{ff}$ |
| `llama.attention.head_count` / `head_count_kv` | $H$, $H_{kv}$(GQA) |
| `llama.attention.layer_norm_rms_epsilon` | $\varepsilon$ |
| `llama.rope.freq_base` | $\theta$ |
| `llama.context_length` | 最大コンテキスト |
| `tokenizer.ggml.*` | 語彙・マージ・スコア(§7) |

`output.weight` が存在しないモデルは tied embeddings(`token_embd.weight` を出力射影に流用)。

### 5.2 1トークンのforward(規範的な演算列)

トークン $t$、隠れ状態 $x \in \mathbb{R}^{d_{model}}$(f32):

1. `x = embed[token]`(embedテーブルが量子化されている場合は該当行をブロック単位で逆量子化: $w_i = d \cdot q_i$ の乗算順序で)。
2. 各層 $\ell = 0..L-1$:
   a. `h = rmsnorm(x, w_attn_norm)`(§5.3)
   b. `q = W_q h`, `k = W_k h`, `v = W_v h`(§5.4のGEMV)
   c. `q, k` にRoPE適用(§4.3)、`k, v` をKVキャッシュ(f32、レイアウト `[layer][kv_head][pos][head_dim]`)へ書き込み
   d. `attn_out = attention(q, cache_k, cache_v)`(§5.5)
   e. `x = x + W_o attn_out`(残差加算は要素ごと、順序自明)
   f. `h = rmsnorm(x, w_ffn_norm)`
   g. `x = x + W_down( silu(W_gate h) ⊙ (W_up h) )`(SwiGLU、§5.7)
3. 最終 `h = rmsnorm(x, w_out_norm)`、`logits = W_out h`。

**量子化と再量子化のタイミング**: GEMVの直前に、入力ベクトル(f32)をQ8Aに量子化する(§3.2)。同じ入力に対する複数のGEMV(`W_q/W_k/W_v` や `W_gate/W_up`)では、**1回だけ量子化した同一のQ8Aバッファを共有**する(MUST — 量子化回数が変わると結果が変わるため、仕様として固定)。

### 5.3 RMSNorm

$$
\mathrm{ss} = \sum_i x_i^2 \;(\text{§2.1の8レーン縮約}), \quad
m = \mathrm{ss} / n, \quad
r = 1.0 / \sqrt{m + \varepsilon}, \quad
y_i = (x_i \cdot r) \cdot w_i
$$

$n$ はf32で厳密表現可能($\le 2^{24}$)。$y_i$ の乗算順序は括弧の通り(MUST)。

### 5.4 GEMV

`y = W x`($W$: 量子化またはF32、`x`: f32)。
- 量子化Wの場合: `x` をQ8A化 → 各出力行 $r$ について§3.3+§2.2で計算。
- F32 Wの場合: 各行について§2.1の `dot_f32_ref` 相当。
- 行間は完全独立 → スレッド並列可(§6)。

### 5.5 Attention(1クエリ位置、causal)

ヘッド $h$、クエリ $q \in \mathbb{R}^{d_h}$、キャッシュ内のキー/値 $k_0..k_t, v_0..v_t$(f32)。GQAではヘッド $h$ が参照するKVヘッドは $\lfloor h / (H/H_{kv}) \rfloor$。

1. スコア: $s_j = \mathrm{dot}(q, k_j) \cdot r_s$、ここで $r_s = 1.0 / \sqrt{d_h}$ は起動時に1回計算した定数(f32)。dotは§2.1。$j$ ごとに独立なので順序自由(並列可)。
2. softmax: $m = \max_j s_j$(順序不変)、$e_j = \mathrm{exp\_f32}(s_j - m)$、$Z = \sum_j e_j$(§2.1の8レーン縮約、$j$ 昇順)、$p_j = e_j / Z$。
3. 重み付き和: 出力次元 $d$ ごとに $o_d = \sum_{j=0}^{t} p_j \cdot v_j[d]$ を **$j$ 昇順の逐次加算**で計算する(MUST)。実装は外側ループ $j$、内側で $d$ 方向にベクトル化(`o[d] += p_j * v_j[d]` の垂直SIMD)すると、$d$ ごとの加算順序が仕様と同一のまま高速化できる。$t$ は数千になりうるため8レーン縮約ではなく逐次と定義する(どちらでもよいが、**一方に固定**する。ここでは逐次を採用: $j$ 方向のベクトル化を封じ、$d$ 方向ベクトル化を素直にするため)。

**Position invariance**: 上記は「1クエリ位置」単位で完結しており、プレフィルを何トークンずつチャンクしても、1トークンずつ生成しても、演算列が同一になる。これがG6の根拠である(§9.3で明示的にテストする)。

### 5.6 logits → 整数CDF(算術符号化接合、normative)

語彙サイズ $V$、logitsベクトル $l \in \mathbb{R}^V$。$M = 2^{24}$ とする。

```
m  = max_i l[i]                                  // 順序不変
e[i] = exp_f32(l[i] - m)                          // e[i] ∈ (0, 1]
Z  = 8レーン縮約で Σ e[i]                          // §2.1
p[i] = e[i] / Z                                   // f32除算
g[i] = (p[i] * (M as f32)) as u32                 // f32乗算 → `as`(ゼロ方向切捨て・飽和)
f[i] = g[i] + 1                                   // 最小頻度1を保証(u32)
T   = Σ f[i]  (u64、厳密)                          // 総和 ≤ V + M(1+δ) < 2^26
cum[i] = Σ_{j<i} f[j]  (u64 prefix sum、i昇順)
```

- 全ステップが決定的(DET-1〜3のみで構成)。**正規化して合計をちょうど $M$ にする再配分は行わない** — range coderは任意の $T < 2^{31}$ を総頻度として扱えるため、再配分の複雑さ(タイ処理)を排除できる。
- 復号側は同一のlogitsから同一の `f/cum/T` を再構成できる(これが本プロジェクト全体の正しさの核)。
- $V \le 2^{18}$(Llama 3の128Kまで)を想定し、`assert!(T < (1u64 << 31))` を実行時に置く。

**任意バイトescape拡張**: DTLZヘッダの `FLAG_BYTE_ESCAPES`(§8.3)
が立っているpayloadでは、語彙シンボル `0..V-1` の後ろに256個の
byte escapeシンボル `V + b`($b = 0..255$)を追加する。通常の
logits由来CDFを上の手順で構築したあと、各escapeシンボルに頻度
`1` を割り当て、`cum/T` をi昇順に追記する。初期状態(文脈長0)
の一様CDFも `V + 256` シンボル上で構築する。復元側でescapeを
復号した場合、そのbyteを直接出力し、モデルのKVキャッシュと文脈
トークン列は進めない(MUST)。この拡張により、tokenizerが一部byteを
単独語彙トークンとしてemitできないGGUFでも、モデル語彙を変更せず
任意バイト列をロスレスに扱える。`FLAG_BYTE_ESCAPES` が立っていない
payloadは互換用の従来形式であり、CDFシンボル数は `V` のみである。

### 5.7 SwiGLU

$\mathrm{silu}(x) = x \cdot \sigma(x) = x / (1 + \mathrm{exp\_f32}(-x))$。演算順序: `t = exp_f32(-x); d = 1.0 + t; y = x / d`。gateとupの積は要素ごと `silu(g[i]) * u[i]`。

---

## 6. 並列化設計

原則: **並列化してよいのは「相互に独立な出力要素の集合」への分割のみ**。1つの縮約を複数スレッドで分担することは禁止(MUST NOT)。

| 箇所 | 並列単位 | 備考 |
|---|---|---|
| GEMV | 出力行 $r$ | 各行の縮約は1スレッド内で完結。`rayon::par_chunks` 等で行範囲を分割してよい(チャンク割りは結果に影響しない) |
| Attention スコア計算 | $(h, j)$ | 独立 |
| Attention 重み付き和 | ヘッド $h$、または出力次元ブロック | $j$ 方向の分割は禁止 |
| RMSNorm | 並列化しない | $d_{model} \le 8192$ 程度なので不要 |
| CDF構築 | `e[i]` 計算は並列可、$Z$ と prefix sum は単一スレッド | |

スレッドプールは `rayon` を使用してよい(数値順序に関与しないため)。`RAYON_NUM_THREADS` をいくつに変えても出力が変わらないことをテストで保証する(§9.2)。Wasmビルドのv1はシングルスレッドとする(wasm threads対応はv2、対応時も同じ原則)。

---

## 7. トークナイザ

- GGUF内蔵の語彙(`tokenizer.ggml.tokens/scores/merges/token_type`)から、SentencePiece BPE(Llama 2系)と GPT-2スタイルBPE(Llama 3系、`tokenizer.ggml.model == "gpt2"`)を自前実装する。トークナイザは純関数なので、正しく実装すれば決定性の問題はない(浮動小数点スコア比較のタイは「最小インデックス優先」で固定)。
- **ロスレス性の要件**: 圧縮対象は任意バイト列。`detokenize(tokens) == 元バイト列` を保証するため、byte_fallbackトークン(`<0xNN>`)やGPT-2 byte-unicode単独トークンを持つ語彙ではそれを優先して使う。語彙が一部byteを単独トークンとしてemitできない場合は、§5.6のbyte escapeシンボルを使ってそのbyteを符号化する。復元側は通常語彙トークンをバイト列に戻し、byte escapeは対応する1 byteを直接出力する。tokenize側の一意性は不要(同じテキストに複数のトークン化があってもよい — 圧縮側が選んだシンボル列がそのまま符号化される)。
- v1の対象モデル(TinyLlama, SmolLM2, Qwen2.5)で実際に使われる方式を優先実装する。

---

## 8. 圧縮パイプラインとファイル形式(`det-coder`, `det-cli`)

### 8.1 Range coder

64bit range coder(Subbotin系、carry-less)を実装する:

```rust
pub struct RangeEncoder { low: u64, range: u64, out: Vec<u8> }
impl RangeEncoder {
    /// cum, freq, total: §5.6のu64値。total < 2^31。
    pub fn encode(&mut self, cum: u64, freq: u64, total: u64);
    pub fn finish(self) -> Vec<u8>;
}
pub struct RangeDecoder<'a> { /* 対称 */ }
impl RangeDecoder<'_> {
    pub fn decode_freq(&mut self, total: u64) -> u64;   // 現在のシンボル位置を返す
    pub fn advance(&mut self, cum: u64, freq: u64, total: u64);
}
```

デコーダ側は `decode_freq` の返り値 $u$ に対し `cum` 配列の二分探索(`partition_point`)でトークンを特定する。renormalizationは8bit単位、`range < 2^32` で1バイト出力、という標準構成。整数演算のみなので決定性は自明。ユニットテスト: ランダムな頻度表で $10^6$ シンボルのround-trip。

### 8.2 コンテキスト管理

llama-zip同様の固定ウィンドウ+オーバーラップ方式:
- コンテキストが `n_ctx` に達したら、直近 `overlap` トークン(デフォルト `n_ctx/4`、ヘッダに記録)を残してKVキャッシュを破棄し、その `overlap` トークンを再プレフィルして継続する。
- 圧縮側・復元側が同一の規則を機械的に適用するため、logits一致がそのまま保たれる。

### 8.3 圧縮ファイル形式

```
magic "DTLZ" | version u16 | flags u16
model_sha256 [32]      // GGUFファイル全体のSHA-256
n_ctx u32 | overlap u32
orig_len u64           // 元バイト長(復元終了判定)
payload …              // range coder出力
```

復元時に手元のGGUFのハッシュが `model_sha256` と一致しなければエラー(異なる量子化・異なるモデルでの復元事故を防ぐ)。EOFの扱いは `orig_len` バイトをデトークナイズ出力した時点で停止(トークン境界とバイト境界のずれは最終トークンの部分出力で吸収)。

`flags`:
- bit 0 `FLAG_BYTE_ESCAPES`: §5.6のbyte escape CDF拡張を使う。新規に作成するDTLZ payloadではこのbitを立てる(MUST)。
- 未知のbitが立っているファイルは復元前に拒否する(MUST)。

### 8.4 CLI

```
detllm compress   -m model.gguf -i input.txt -o out.dtlz [--n-ctx N] [--threads T]
detllm decompress -m model.gguf -i out.dtlz  -o restored.txt
detllm logits     -m model.gguf -p "prompt" --hash        // 検証用: 全位置logitsのSHA-256を出力
detllm selftest                                            // §9.4のcanaryを単独実行
```

`logits --hash` はCI・環境間検証の主要ツールになる(logitsのf32ビット列をリトルエンディアンで連結してSHA-256)。

---

## 9. テスト・検証戦略

### 9.1 参照実装 = 規範

`det-num` / `det-quant` の各カーネルにはスカラ参照実装(`*_ref`)を置き、これを規範とする。SIMDバックエンドのテストは全て `assert_eq!(simd_result.to_bits(), ref_result.to_bits())`(ビット比較。`==` はf32比較なので使わない — `-0.0 == 0.0` を見逃す)。proptestでランダム入力(長さ・値域・subnormal含む)を大量に流す。

### 9.2 決定性プロパティテスト

同一入力に対し、(a) スレッド数 {1, 2, 7, 16}、(b) SIMDバックエンド {scalar, simd}、(c) プレフィルチャンク {1, 3, 全量} を変えて `logits --hash` が全一致することをテスト。

### 9.3 Position invariance テスト

トークン列を「一括プレフィル」と「1トークンずつ逐次評価」の両方で流し、全位置のlogitsビット一致を確認(G6の直接検証)。

### 9.4 起動時canary(DET-6)

固定シードで生成した既知の小行列・ベクトルに対して、dot / Q8A量子化 / exp_f32 / rmsnorm / CDF構築を実行し、結果ビット列のSHA-256をコンパイル時定数と比較。不一致なら「この環境ではbit再現を保証できない」旨のエラーでabortする。期待値の更新は仕様変更としてPRレビュー必須。subnormal入力を必ず含める(FTZ検出のため: 例 `f32::from_bits(1) + f32::from_bits(1)` が `f32::from_bits(2)` になること)。

### 9.5 クロスプラットフォームCI(最重要)

GitHub Actions マトリクス:

| ジョブ | ランナー | 内容 |
|---|---|---|
| x86_64-linux | ubuntu-latest | ユニット+統合、`logits --hash` を成果物化 |
| aarch64-macos | macos-14 (M系) | 同上 |
| aarch64-linux | ubuntu-24.04-arm | 同上 |
| wasm32-wasip1 | ubuntu-latest + wasmtime | `cargo build --target wasm32-wasip1` → wasmtimeで `logits --hash` |
| toolchain-skew | ubuntu-latest | stable と stable-1つ前 の2バージョンでビルドしハッシュ比較 |

最終ステップで**全ジョブのハッシュが一致**することをassertするジョブを置く。テストモデルはCIサイズの都合上、(a) 極小の自作ランダム重みGGUF(数MB、リポジトリ同梱)と、(b) nightlyジョブでのみHFから取得するTinyLlama Q8_0、の2段構え。

### 9.6 正しさのサニティ(bit一致とは別軸)

llama.cpp(またはHF transformers)のf32出力とのコサイン類似度 $> 0.999$ を確認し、「決定的だが間違った実装」を検出する。bit一致は要求しない。加えてenwik8等でperplexity/圧縮率を測定し、既知値(参考: ts_zipはRWKV 169Mでenwik8 1.11 bpb)と比較して桁の妥当性を見る。

### 9.7 Round-trip

各対象モデル×各量子化で、多言語テキスト・バイナリ混在データ・空入力・巨大入力(コンテキスト複数周)の compress→decompress→バイト一致 を統合テスト化。

---

## 10. クレート構成

```
detllm/                     (workspace, Rust edition 2021, MSRV明記)
├── crates/
│   ├── det-num/            依存ゼロ, no_std対応。§2縮約, §4数学関数(vendored libm), 丸め, f16
│   ├── det-quant/          Q8_0/Q4_0/Q8A 型と内積カーネル(scalar + feature "simd")
│   ├── det-gguf/           GGUFパーサ。&[u8]上でゼロコピー。ネイティブはmmap(memmap2)、
│   │                       Wasmは読み込み済みバッファ — 入力は `trait ModelBytes: Deref<[u8]>`
│   ├── det-token/          §7 トークナイザ
│   ├── det-model/          §5 forward pass, KVキャッシュ, ウィンドウ管理
│   ├── det-coder/          §8 range coder + CDF builder
│   └── det-cli/            CLI(clap), selftest, logits --hash
├── tests/                  統合テスト(§9.2, 9.3, 9.7)
├── testdata/               自作極小GGUF, goldenハッシュ
└── .github/workflows/      §9.5 CIマトリクス
```

feature flags: `simd`(`core::arch` intrinsicsによるAVX2/NEON/simd128実装。デフォルトoff、CIでon/off両方を検証)、`parallel`(rayon)。**cfgでの数値分岐は「実装の選択」のみで、全実装がbit一致という不変条件をテストが守る。**

## 11. マイルストーン(コーディングエージェント向け、各段階に受け入れ条件)

| M | 内容 | 受け入れ条件 |
|---|---|---|
| M0 | `det-num`: 縮約・丸め・f16・vendored expf/f64 libm | 参照実装のユニットテスト全通過。canaryハッシュ確定。subnormalテスト含む |
| M1 | `det-gguf`: パーサ+テンソルビュー | TinyLlama GGUFのメタデータ/テンソル一覧をdumpできる。自作極小GGUFの生成ツール(`xtask`)完成 |
| M2 | `det-model` F32のみ・スカラ・単スレッドでTinyLlama forward | HF transformersとのlogitsコサイン類似度 > 0.999。`logits --hash` 実装 |
| M3 | Q8_0/Q4_0 + Q8A 経路 | F32経路とのperplexity差が既知の量子化劣化範囲内。§9.3 position invariance通過 |
| M4 | `det-coder` + CLI compress/decompress | §9.7 round-trip通過。enwik8先頭1MBで圧縮率をREADMEに記録 |
| M5 | `parallel` feature | §9.2 スレッド数不変テスト通過 |
| M6 | `simd` feature(AVX2 → NEON → simd128の順) | 各backendがscalarとビット一致(proptest 10^6ケース)。ベンチ(criterion)をREADMEに記録 |
| M7 | wasm32-wasip1ビルド + CIマトリクス完成 | **4ターゲット+2toolchainで `logits --hash` 全一致**(プロジェクトの最終防衛線) |
| M8 | SmolLM2-1.7B / Qwen2.5-1.5B対応(GPT-2 BPE等) | 各モデルでM4〜M7の全テスト通過 |

## 12. 既知のリスクと対応

- **libcのFTZ混入**: pure Rust構成で回避。将来C依存を入れる場合はcanary(§9.4)が検出する。
- **`round_ties_even` の可用性**: Rust 1.77未満では未安定。MSRVを1.77+に設定。
- **AVX-512環境での誤実装**: §2.1に明記した通り16レーン1本での縮約は仕様違反。proptestのビット比較が検出する前提だが、コードレビュー項目にも入れる。
- **Wasm実行系のバグ**: wasmtimeのバージョンもCIで固定しつつ、仕様上の根拠(non-relaxed Wasmの数値決定性、NaN以外)があるため複数ランタイム(wasmer)でのクロスチェックをnightly CIに追加してもよい。
- **モデルのf16重みにInf/NaN**: ロード時に全スケール値を検査して拒否(DET-4)。
- **性能**: 1〜2Bモデル・Q8_0・4スレッドで数〜十数tok/sを想定。圧縮用途では許容。ボトルネックはGEMVのメモリ帯域であり、決定性制約(ブロック逐次f32加算)はi32内積のSIMD化を妨げないため、ネイティブ最適化余地は大きい。

## 13. 参考文献(設計根拠)

- Thinking Machines Lab, "Defeating Nondeterminism in LLM Inference" — batch不変カーネルの設計原理: https://thinkingmachines.ai/blog/defeating-nondeterminism-in-llm-inference/
- Microsoft RepDL — 「正しい丸め+固定縮約順序」で環境間bit再現を達成した先行DLライブラリ: https://github.com/microsoft/RepDL / arXiv:2510.09180
- Fabrice Bellard, ts_zip / LibNC — CPUブランド・OS非依存の再現推論によるLLM圧縮の実証(閉源): https://bellard.org/ts_zip/ , https://bellard.org/libnc/libnc.html
- WebAssembly仕様の非決定性(NaNビットのみ、relaxed-simdは非決定的): https://github.com/WebAssembly/design/blob/main/Nondeterminism.md , https://webassembly.github.io/spec/core/exec/numerics.html
- Wasmtime deterministic execution: https://docs.wasmtime.dev/examples-deterministic-wasm-execution.html
- CORE-MATH / RLIBM(correctly rounded f32超越関数): https://core-math.gitlabpages.inria.fr/ , https://github.com/rutgers-apl/rlibm
- llama.cpp GGUF仕様: https://github.com/ggml-org/ggml/blob/master/docs/gguf.md
- llama-zip(置換対象、CLI/UX参考): https://github.com/AlexBuz/llama-zip
