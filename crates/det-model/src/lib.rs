use det_num::{
    cos_f64, dot_f32_ref, exp_f32, exp_f64, f16_is_finite, f16_to_f32, ln_f64, sin_f64,
    sum_f32_ref, sum_squares_f32_ref, Sha256,
};
use det_quant::{
    dot_q4_0_q8a, dot_q8_0_q8a, q4_0_block_from_gguf, q8_0_block_from_gguf, quantize_q8a,
    Q4_0Block, Q8ABlock, Q8_0Block, BLOCK,
};

static THREAD_COUNT_OVERRIDE: core::sync::atomic::AtomicUsize =
    core::sync::atomic::AtomicUsize::new(0);

#[derive(Debug, Clone, Copy, PartialEq)]
pub struct LlamaConfig {
    pub block_count: usize,
    pub embedding_length: usize,
    pub feed_forward_length: usize,
    pub head_count: usize,
    pub head_count_kv: usize,
    pub rms_epsilon: f32,
    pub attention_scale: f32,
    pub rope_freq_base: f32,
    pub rope_dimension_count: usize,
    pub rope_pairing: RopePairing,
    pub context_length: usize,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum RopePairing {
    Adjacent,
    SplitHalf,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum ModelError {
    Shape,
    NonFinite,
    UnsupportedArchitecture,
    UnsupportedAttentionHeadLength,
    UnsupportedModelFeature,
    UnsupportedRopeScaling,
    MissingMetadata,
    Gguf,
    UnsupportedTensorType,
    ThreadPanicked,
}

#[derive(Debug, Clone, PartialEq)]
pub struct F32Matrix {
    pub rows: usize,
    pub cols: usize,
    pub data: Vec<f32>,
}

#[derive(Debug, Clone, PartialEq)]
pub struct F32LayerWeights {
    pub attention_norm: Vec<f32>,
    pub wq: WeightMatrix,
    pub wk: WeightMatrix,
    pub wv: WeightMatrix,
    pub wo: WeightMatrix,
    pub ffn_norm: Vec<f32>,
    pub w_gate: WeightMatrix,
    pub w_up: WeightMatrix,
    pub w_down: WeightMatrix,
}

#[derive(Debug, Clone, PartialEq)]
pub struct F32Llama {
    pub config: LlamaConfig,
    pub token_embedding: WeightMatrix,
    pub layers: Vec<F32LayerWeights>,
    pub output_norm: Vec<f32>,
    pub output: WeightMatrix,
}

#[derive(Debug, Clone, PartialEq)]
pub enum WeightMatrix {
    F32(F32Matrix),
    Q8_0(Q8Matrix),
    Q4_0(Q4Matrix),
}

#[derive(Debug, Clone, PartialEq)]
pub struct Q8Matrix {
    pub rows: usize,
    pub cols: usize,
    pub blocks_per_row: usize,
    pub blocks: Vec<Q8_0Block>,
}

#[derive(Debug, Clone, PartialEq)]
pub struct Q4Matrix {
    pub rows: usize,
    pub cols: usize,
    pub blocks_per_row: usize,
    pub blocks: Vec<Q4_0Block>,
}

#[derive(Debug, Clone, PartialEq)]
pub struct RopeTables {
    pub positions: usize,
    pub half_head_dim: usize,
    pub pairing: RopePairing,
    pub cos: Vec<f32>,
    pub sin: Vec<f32>,
}

#[derive(Debug, Clone, PartialEq)]
pub struct KvCache {
    config: LlamaConfig,
    head_dim: usize,
    k: Vec<f32>,
    v: Vec<f32>,
}

#[derive(Debug, Clone, Copy)]
pub struct F32TensorView<'a> {
    pub info: &'a det_gguf::TensorInfo,
    data: &'a [u8],
}

pub fn set_thread_count(threads: Option<usize>) -> Result<(), ModelError> {
    let value = match threads {
        Some(0) => return Err(ModelError::Shape),
        Some(value) => value,
        None => 0,
    };
    THREAD_COUNT_OVERRIDE.store(value, core::sync::atomic::Ordering::Relaxed);
    Ok(())
}

impl F32Matrix {
    pub fn new(rows: usize, cols: usize, data: Vec<f32>) -> Result<Self, ModelError> {
        let matrix = Self { rows, cols, data };
        matrix.validate()?;
        Ok(matrix)
    }

    fn validate(&self) -> Result<(), ModelError> {
        if self.rows == 0 || self.cols == 0 || checked_len(self.rows, self.cols)? != self.data.len()
        {
            return Err(ModelError::Shape);
        }
        ensure_finite_slice(&self.data)
    }

    pub fn row(&self, row: usize) -> Result<&[f32], ModelError> {
        self.validate()?;
        if row >= self.rows {
            return Err(ModelError::Shape);
        }
        let start = row.checked_mul(self.cols).ok_or(ModelError::Shape)?;
        let end = start.checked_add(self.cols).ok_or(ModelError::Shape)?;
        self.data.get(start..end).ok_or(ModelError::Shape)
    }

    pub fn gemv(&self, x: &[f32], out: &mut [f32]) -> Result<(), ModelError> {
        self.validate()?;
        if x.len() != self.cols || out.len() != self.rows {
            return Err(ModelError::Shape);
        }
        ensure_finite_slice(x)?;
        gemv_rows(self.rows, out, |r| {
            let start = r * self.cols;
            let end = start + self.cols;
            Ok(dot_f32_ref(&self.data[start..end], x))
        })
    }
}

fn ensure_finite_slice(values: &[f32]) -> Result<(), ModelError> {
    if values.iter().any(|v| !v.is_finite()) {
        return Err(ModelError::NonFinite);
    }
    Ok(())
}

fn gemv_rows<F>(rows: usize, out: &mut [f32], f: F) -> Result<(), ModelError>
where
    F: Fn(usize) -> Result<f32, ModelError> + Sync,
{
    if out.len() != rows {
        return Err(ModelError::Shape);
    }

    #[cfg(all(feature = "parallel", not(target_family = "wasm")))]
    {
        let threads = requested_thread_count();
        gemv_rows_parallel(rows, out, f, threads)
    }

    #[cfg(not(all(feature = "parallel", not(target_family = "wasm"))))]
    {
        gemv_rows_sequential(rows, out, f)
    }
}

#[cfg(all(feature = "parallel", not(target_family = "wasm")))]
fn requested_thread_count() -> usize {
    let configured = THREAD_COUNT_OVERRIDE.load(core::sync::atomic::Ordering::Relaxed);
    if configured != 0 {
        configured
    } else {
        std::thread::available_parallelism()
            .map(usize::from)
            .unwrap_or(1)
    }
}

fn gemv_rows_sequential<F>(rows: usize, out: &mut [f32], f: F) -> Result<(), ModelError>
where
    F: Fn(usize) -> Result<f32, ModelError>,
{
    for (r, dst) in out.iter_mut().enumerate().take(rows) {
        let value = f(r)?;
        if !value.is_finite() {
            return Err(ModelError::NonFinite);
        }
        *dst = value;
    }
    Ok(())
}

#[cfg(all(feature = "parallel", not(target_family = "wasm")))]
fn gemv_rows_parallel<F>(
    rows: usize,
    out: &mut [f32],
    f: F,
    requested_threads: usize,
) -> Result<(), ModelError>
where
    F: Fn(usize) -> Result<f32, ModelError> + Sync,
{
    let threads = requested_threads.clamp(1, rows.max(1));
    if threads == 1 || rows < 2 {
        return gemv_rows_sequential(rows, out, f);
    }

    let chunk_len = rows.div_ceil(threads);
    std::thread::scope(|scope| {
        let mut handles = Vec::new();
        for (chunk_index, chunk) in out.chunks_mut(chunk_len).enumerate() {
            let start = chunk_index * chunk_len;
            let f = &f;
            handles.push(scope.spawn(move || {
                for (offset, dst) in chunk.iter_mut().enumerate() {
                    let value = f(start + offset)?;
                    if !value.is_finite() {
                        return Err(ModelError::NonFinite);
                    }
                    *dst = value;
                }
                Ok(())
            }));
        }

        for handle in handles {
            match handle.join() {
                Ok(Ok(())) => {}
                Ok(Err(e)) => return Err(e),
                Err(_) => return Err(ModelError::ThreadPanicked),
            }
        }
        Ok(())
    })
}

impl From<F32Matrix> for WeightMatrix {
    fn from(value: F32Matrix) -> Self {
        Self::F32(value)
    }
}

impl WeightMatrix {
    pub fn rows(&self) -> usize {
        match self {
            Self::F32(m) => m.rows,
            Self::Q8_0(m) => m.rows,
            Self::Q4_0(m) => m.rows,
        }
    }

    pub fn cols(&self) -> usize {
        match self {
            Self::F32(m) => m.cols,
            Self::Q8_0(m) => m.cols,
            Self::Q4_0(m) => m.cols,
        }
    }

    fn validate(&self) -> Result<(), ModelError> {
        match self {
            Self::F32(m) => m.validate(),
            Self::Q8_0(m) => {
                validate_quant_matrix_shape(m.rows, m.cols, m.blocks_per_row, m.blocks.len())?;
                if m.blocks.iter().any(|block| !block.d.is_finite()) {
                    return Err(ModelError::NonFinite);
                }
                Ok(())
            }
            Self::Q4_0(m) => {
                validate_quant_matrix_shape(m.rows, m.cols, m.blocks_per_row, m.blocks.len())?;
                if m.blocks.iter().any(|block| !block.d.is_finite()) {
                    return Err(ModelError::NonFinite);
                }
                Ok(())
            }
        }
    }

    pub fn gemv(&self, x: &[f32], out: &mut [f32]) -> Result<(), ModelError> {
        self.validate()?;
        if x.len() != self.cols() || out.len() != self.rows() {
            return Err(ModelError::Shape);
        }
        ensure_finite_slice(x)?;
        match self {
            Self::F32(m) => m.gemv(x, out),
            Self::Q8_0(m) => {
                let qx = quantize_q8a(x).map_err(map_quant_error)?;
                gemv_rows(m.rows, out, |r| {
                    let start = r * m.blocks_per_row;
                    let end = start + m.blocks_per_row;
                    dot_q8_0_q8a(&m.blocks[start..end], &qx).map_err(map_quant_error)
                })
            }
            Self::Q4_0(m) => {
                let qx = quantize_q8a(x).map_err(map_quant_error)?;
                gemv_rows(m.rows, out, |r| {
                    let start = r * m.blocks_per_row;
                    let end = start + m.blocks_per_row;
                    dot_q4_0_q8a(&m.blocks[start..end], &qx).map_err(map_quant_error)
                })
            }
        }
    }

    fn needs_q8a(&self) -> bool {
        matches!(self, Self::Q8_0(_) | Self::Q4_0(_))
    }

    fn gemv_with_optional_q8a(
        &self,
        x: &[f32],
        qx: Option<&[Q8ABlock]>,
        out: &mut [f32],
    ) -> Result<(), ModelError> {
        if x.len() != self.cols() || out.len() != self.rows() {
            return Err(ModelError::Shape);
        }
        ensure_finite_slice(x)?;
        match self {
            Self::F32(m) => m.gemv(x, out),
            Self::Q8_0(m) => {
                let qx = qx.ok_or(ModelError::Shape)?;
                if qx.len() != m.blocks_per_row {
                    return Err(ModelError::Shape);
                }
                gemv_rows(m.rows, out, |r| {
                    let start = r * m.blocks_per_row;
                    let end = start + m.blocks_per_row;
                    dot_q8_0_q8a(&m.blocks[start..end], qx).map_err(map_quant_error)
                })
            }
            Self::Q4_0(m) => {
                let qx = qx.ok_or(ModelError::Shape)?;
                if qx.len() != m.blocks_per_row {
                    return Err(ModelError::Shape);
                }
                gemv_rows(m.rows, out, |r| {
                    let start = r * m.blocks_per_row;
                    let end = start + m.blocks_per_row;
                    dot_q4_0_q8a(&m.blocks[start..end], qx).map_err(map_quant_error)
                })
            }
        }
    }

    pub fn row_to_vec(&self, row: usize) -> Result<Vec<f32>, ModelError> {
        self.validate()?;
        if row >= self.rows() {
            return Err(ModelError::Shape);
        }
        match self {
            Self::F32(m) => Ok(m.row(row)?.to_vec()),
            Self::Q8_0(m) => {
                let mut out = Vec::with_capacity(m.cols);
                let start = row * m.blocks_per_row;
                for block in &m.blocks[start..start + m.blocks_per_row] {
                    for &q in &block.q {
                        let value = block.d * (q as f32);
                        if !value.is_finite() {
                            return Err(ModelError::NonFinite);
                        }
                        out.push(value);
                    }
                }
                Ok(out)
            }
            Self::Q4_0(m) => {
                let mut out = Vec::with_capacity(m.cols);
                let start = row * m.blocks_per_row;
                for block in &m.blocks[start..start + m.blocks_per_row] {
                    for i in 0..BLOCK {
                        let byte = block.qs[i / 2];
                        let nibble = if i % 2 == 0 { byte & 0x0f } else { byte >> 4 };
                        let value = block.d * ((nibble as i32 - 8) as f32);
                        if !value.is_finite() {
                            return Err(ModelError::NonFinite);
                        }
                        out.push(value);
                    }
                }
                Ok(out)
            }
        }
    }
}

fn validate_quant_matrix_shape(
    rows: usize,
    cols: usize,
    blocks_per_row: usize,
    block_count: usize,
) -> Result<(), ModelError> {
    if rows == 0
        || cols == 0
        || blocks_per_row == 0
        || cols % BLOCK != 0
        || blocks_per_row != cols / BLOCK
        || checked_len(rows, blocks_per_row)? != block_count
    {
        return Err(ModelError::Shape);
    }
    Ok(())
}

impl RopeTables {
    pub fn identity(positions: usize, head_dim: usize) -> Result<Self, ModelError> {
        if positions == 0 || head_dim == 0 || head_dim % 2 != 0 {
            return Err(ModelError::Shape);
        }
        let half_head_dim = head_dim / 2;
        let table_len = checked_len(positions, half_head_dim)?;
        Ok(Self {
            positions,
            half_head_dim,
            pairing: RopePairing::Adjacent,
            cos: vec![1.0; table_len],
            sin: vec![0.0; table_len],
        })
    }

    pub fn values(&self, pos: usize, j: usize) -> Result<(f32, f32), ModelError> {
        if pos >= self.positions || j >= self.half_head_dim {
            return Err(ModelError::Shape);
        }
        let expected_len = self
            .positions
            .checked_mul(self.half_head_dim)
            .ok_or(ModelError::Shape)?;
        if self.cos.len() != expected_len || self.sin.len() != expected_len {
            return Err(ModelError::Shape);
        }
        let idx = pos * self.half_head_dim + j;
        let c = self.cos[idx];
        let s = self.sin[idx];
        if !c.is_finite() || !s.is_finite() {
            return Err(ModelError::NonFinite);
        }
        Ok((c, s))
    }

    pub fn llama(config: LlamaConfig, positions: usize) -> Result<Self, ModelError> {
        config.validate()?;
        let head_dim = config.head_dim()?;
        let rope_dim = config.rope_dimension_count;
        if positions == 0 || rope_dim == 0 || rope_dim % 2 != 0 || rope_dim > head_dim {
            return Err(ModelError::Shape);
        }
        let half_head_dim = rope_dim / 2;
        let table_len = checked_len(positions, half_head_dim)?;
        let mut cos = Vec::with_capacity(table_len);
        let mut sin = Vec::with_capacity(table_len);
        let ln_base = ln_f64(config.rope_freq_base as f64);
        for pos in 0..positions {
            for j in 0..half_head_dim {
                let exponent = -((2 * j) as f64) / (rope_dim as f64);
                let omega = exp_f64(exponent * ln_base);
                let phi = (pos as f64) * omega;
                let c = cos_f64(phi) as f32;
                let s = sin_f64(phi) as f32;
                if !c.is_finite() || !s.is_finite() {
                    return Err(ModelError::NonFinite);
                }
                cos.push(c);
                sin.push(s);
            }
        }
        Ok(Self {
            positions,
            half_head_dim,
            pairing: config.rope_pairing,
            cos,
            sin,
        })
    }
}

impl KvCache {
    pub fn new(config: LlamaConfig) -> Result<Self, ModelError> {
        config.validate()?;
        let head_dim = config.head_dim()?;
        let len = config
            .block_count
            .checked_mul(config.head_count_kv)
            .and_then(|x| x.checked_mul(config.context_length))
            .and_then(|x| x.checked_mul(head_dim))
            .ok_or(ModelError::Shape)?;
        Ok(Self {
            config,
            head_dim,
            k: vec![0.0; len],
            v: vec![0.0; len],
        })
    }

    fn offset(&self, layer: usize, kv_head: usize, pos: usize) -> Result<usize, ModelError> {
        if layer >= self.config.block_count
            || kv_head >= self.config.head_count_kv
            || pos >= self.config.context_length
        {
            return Err(ModelError::Shape);
        }
        layer
            .checked_mul(self.config.head_count_kv)
            .and_then(|x| x.checked_add(kv_head))
            .and_then(|x| x.checked_mul(self.config.context_length))
            .and_then(|x| x.checked_add(pos))
            .and_then(|x| x.checked_mul(self.head_dim))
            .ok_or(ModelError::Shape)
    }

    pub fn store(
        &mut self,
        layer: usize,
        pos: usize,
        k_layer: &[f32],
        v_layer: &[f32],
    ) -> Result<(), ModelError> {
        let expected = checked_len(self.config.head_count_kv, self.head_dim)?;
        if k_layer.len() != expected || v_layer.len() != expected {
            return Err(ModelError::Shape);
        }
        ensure_finite_slice(k_layer)?;
        ensure_finite_slice(v_layer)?;
        for kv_head in 0..self.config.head_count_kv {
            let src = kv_head
                .checked_mul(self.head_dim)
                .ok_or(ModelError::Shape)?;
            let src_end = src.checked_add(self.head_dim).ok_or(ModelError::Shape)?;
            let dst = self.offset(layer, kv_head, pos)?;
            let dst_end = dst.checked_add(self.head_dim).ok_or(ModelError::Shape)?;
            self.k
                .get_mut(dst..dst_end)
                .ok_or(ModelError::Shape)?
                .copy_from_slice(k_layer.get(src..src_end).ok_or(ModelError::Shape)?);
            self.v
                .get_mut(dst..dst_end)
                .ok_or(ModelError::Shape)?
                .copy_from_slice(v_layer.get(src..src_end).ok_or(ModelError::Shape)?);
        }
        Ok(())
    }

    pub fn key(&self, layer: usize, kv_head: usize, pos: usize) -> Result<&[f32], ModelError> {
        let offset = self.offset(layer, kv_head, pos)?;
        let end = offset.checked_add(self.head_dim).ok_or(ModelError::Shape)?;
        self.k.get(offset..end).ok_or(ModelError::Shape)
    }

    pub fn value(&self, layer: usize, kv_head: usize, pos: usize) -> Result<&[f32], ModelError> {
        let offset = self.offset(layer, kv_head, pos)?;
        let end = offset.checked_add(self.head_dim).ok_or(ModelError::Shape)?;
        self.v.get(offset..end).ok_or(ModelError::Shape)
    }
}

impl F32Llama {
    pub fn from_gguf(gguf: &det_gguf::Gguf, bytes: &[u8]) -> Result<Self, ModelError> {
        let config = LlamaConfig::from_gguf(gguf)?;
        config.validate()?;
        let d = config.embedding_length;
        let d_ff = config.feed_forward_length;
        let head_dim = config.head_dim()?;
        let token_embedding =
            read_weight_matrix(gguf, bytes, "token_embd.weight", vocab_size(gguf)?, d)?;

        let mut layers = Vec::with_capacity(config.block_count);
        for layer in 0..config.block_count {
            layers.push(F32LayerWeights {
                attention_norm: read_f32_vector(
                    gguf,
                    bytes,
                    &format!("blk.{layer}.attn_norm.weight"),
                    d,
                )?,
                wq: read_weight_matrix(
                    gguf,
                    bytes,
                    &format!("blk.{layer}.attn_q.weight"),
                    config.head_count * head_dim,
                    d,
                )?,
                wk: read_weight_matrix(
                    gguf,
                    bytes,
                    &format!("blk.{layer}.attn_k.weight"),
                    config.head_count_kv * head_dim,
                    d,
                )?,
                wv: read_weight_matrix(
                    gguf,
                    bytes,
                    &format!("blk.{layer}.attn_v.weight"),
                    config.head_count_kv * head_dim,
                    d,
                )?,
                wo: read_weight_matrix(
                    gguf,
                    bytes,
                    &format!("blk.{layer}.attn_output.weight"),
                    d,
                    config.head_count * head_dim,
                )?,
                ffn_norm: read_f32_vector(gguf, bytes, &format!("blk.{layer}.ffn_norm.weight"), d)?,
                w_gate: read_weight_matrix(
                    gguf,
                    bytes,
                    &format!("blk.{layer}.ffn_gate.weight"),
                    d_ff,
                    d,
                )?,
                w_up: read_weight_matrix(
                    gguf,
                    bytes,
                    &format!("blk.{layer}.ffn_up.weight"),
                    d_ff,
                    d,
                )?,
                w_down: read_weight_matrix(
                    gguf,
                    bytes,
                    &format!("blk.{layer}.ffn_down.weight"),
                    d,
                    d_ff,
                )?,
            });
        }

        let output_norm = read_f32_vector(gguf, bytes, "output_norm.weight", d)?;
        let output = match gguf.tensor("output.weight") {
            Ok(_) => read_weight_matrix(gguf, bytes, "output.weight", token_embedding.rows(), d)?,
            Err(det_gguf::GgufError::TensorNotFound) => token_embedding.clone(),
            Err(_) => return Err(ModelError::Gguf),
        };

        let model = Self {
            config,
            token_embedding,
            layers,
            output_norm,
            output,
        };
        model.validate()?;
        Ok(model)
    }

    pub fn validate(&self) -> Result<(), ModelError> {
        self.config.validate()?;
        let d = self.config.embedding_length;
        let d_ff = self.config.feed_forward_length;
        let head_dim = self.config.head_dim()?;
        if self.layers.len() != self.config.block_count
            || self.token_embedding.cols() != d
            || self.output.rows() != self.token_embedding.rows()
            || self.output.cols() != d
            || self.output_norm.len() != d
        {
            return Err(ModelError::Shape);
        }
        self.token_embedding.validate()?;
        self.output.validate()?;
        ensure_finite_slice(&self.output_norm)?;
        for layer in &self.layers {
            if layer.attention_norm.len() != d
                || layer.ffn_norm.len() != d
                || layer.wq.rows() != self.config.head_count * head_dim
                || layer.wq.cols() != d
                || layer.wk.rows() != self.config.head_count_kv * head_dim
                || layer.wk.cols() != d
                || layer.wv.rows() != self.config.head_count_kv * head_dim
                || layer.wv.cols() != d
                || layer.wo.rows() != d
                || layer.wo.cols() != self.config.head_count * head_dim
                || layer.w_gate.rows() != d_ff
                || layer.w_gate.cols() != d
                || layer.w_up.rows() != d_ff
                || layer.w_up.cols() != d
                || layer.w_down.rows() != d
                || layer.w_down.cols() != d_ff
            {
                return Err(ModelError::Shape);
            }
            ensure_finite_slice(&layer.attention_norm)?;
            ensure_finite_slice(&layer.ffn_norm)?;
            layer.wq.validate()?;
            layer.wk.validate()?;
            layer.wv.validate()?;
            layer.wo.validate()?;
            layer.w_gate.validate()?;
            layer.w_up.validate()?;
            layer.w_down.validate()?;
        }
        Ok(())
    }

    pub fn forward_one(
        &self,
        token: usize,
        pos: usize,
        rope: &RopeTables,
        cache: &mut KvCache,
        logits: &mut [f32],
    ) -> Result<(), ModelError> {
        self.validate()?;
        if pos >= self.config.context_length
            || logits.len() != self.output.rows()
            || token >= self.token_embedding.rows()
            || cache.config != self.config
        {
            return Err(ModelError::Shape);
        }
        if rope.positions <= pos
            || rope.half_head_dim * 2 != self.config.rope_dimension_count
            || rope.pairing != self.config.rope_pairing
        {
            return Err(ModelError::Shape);
        }
        let d = self.config.embedding_length;
        let head_dim = self.config.head_dim()?;
        let mut x = self.token_embedding.row_to_vec(token)?;

        let mut h = vec![0.0; d];
        let mut q = vec![0.0; self.config.head_count * head_dim];
        let mut k = vec![0.0; self.config.head_count_kv * head_dim];
        let mut v = vec![0.0; self.config.head_count_kv * head_dim];
        let mut attn = vec![0.0; self.config.head_count * head_dim];
        let mut scores = vec![0.0; pos + 1];
        let mut probs = vec![0.0; pos + 1];
        let mut key_window = vec![0.0; (pos + 1) * head_dim];
        let mut value_window = vec![0.0; (pos + 1) * head_dim];
        let mut tmp_d = vec![0.0; d];
        let mut gate = vec![0.0; self.config.feed_forward_length];
        let mut up = vec![0.0; self.config.feed_forward_length];
        let mut ff = vec![0.0; self.config.feed_forward_length];

        for (layer_idx, layer) in self.layers.iter().enumerate() {
            rmsnorm(&x, &layer.attention_norm, self.config.rms_epsilon, &mut h)?;
            let h_q8a = shared_q8a_if_needed(&h, [&layer.wq, &layer.wk, &layer.wv])?;
            layer
                .wq
                .gemv_with_optional_q8a(&h, h_q8a.as_deref(), &mut q)?;
            layer
                .wk
                .gemv_with_optional_q8a(&h, h_q8a.as_deref(), &mut k)?;
            layer
                .wv
                .gemv_with_optional_q8a(&h, h_q8a.as_deref(), &mut v)?;
            apply_rope(&mut q, self.config.head_count, head_dim, pos, rope)?;
            apply_rope(&mut k, self.config.head_count_kv, head_dim, pos, rope)?;
            cache.store(layer_idx, pos, &k, &v)?;

            for head in 0..self.config.head_count {
                let kv_head = head / (self.config.head_count / self.config.head_count_kv);
                let qh = &q[head * head_dim..(head + 1) * head_dim];
                for j in 0..=pos {
                    let dst = j * head_dim;
                    key_window[dst..dst + head_dim]
                        .copy_from_slice(cache.key(layer_idx, kv_head, j)?);
                    value_window[dst..dst + head_dim]
                        .copy_from_slice(cache.value(layer_idx, kv_head, j)?);
                }
                attention_scores_one_head_scaled(
                    qh,
                    &key_window,
                    head_dim,
                    self.config.attention_scale,
                    &mut scores,
                )?;
                probs.copy_from_slice(&scores);
                softmax_in_place(&mut probs)?;
                let out_head = &mut attn[head * head_dim..(head + 1) * head_dim];
                attention_weighted_value(&probs, &value_window, head_dim, out_head)?;
            }

            layer.wo.gemv(&attn, &mut tmp_d)?;
            residual_add(&mut x, &tmp_d)?;

            rmsnorm(&x, &layer.ffn_norm, self.config.rms_epsilon, &mut h)?;
            let h_q8a = shared_q8a_if_needed(&h, [&layer.w_gate, &layer.w_up])?;
            layer
                .w_gate
                .gemv_with_optional_q8a(&h, h_q8a.as_deref(), &mut gate)?;
            layer
                .w_up
                .gemv_with_optional_q8a(&h, h_q8a.as_deref(), &mut up)?;
            swiglu(&gate, &up, &mut ff)?;
            layer.w_down.gemv(&ff, &mut tmp_d)?;
            residual_add(&mut x, &tmp_d)?;
        }

        rmsnorm(&x, &self.output_norm, self.config.rms_epsilon, &mut h)?;
        self.output.gemv(&h, logits)?;
        Ok(())
    }

    pub fn logits_hash_for_tokens(&self, tokens: &[usize]) -> Result<[u8; 32], ModelError> {
        self.logits_hash_for_tokens_chunked(tokens, 1)
    }

    pub fn logits_hash_for_tokens_chunked(
        &self,
        tokens: &[usize],
        chunk_size: usize,
    ) -> Result<[u8; 32], ModelError> {
        let mut hash = Sha256::new();
        self.visit_logits_for_tokens_chunked(tokens, chunk_size, |logits| {
            for &logit in logits {
                hash.update(&logit.to_bits().to_le_bytes());
            }
        })?;
        Ok(hash.finalize())
    }

    pub fn logits_bytes_for_tokens_chunked(
        &self,
        tokens: &[usize],
        chunk_size: usize,
    ) -> Result<Vec<u8>, ModelError> {
        self.validate_logits_request(tokens, chunk_size)?;
        let mut bytes = Vec::with_capacity(logits_byte_len(tokens.len(), self.output.rows())?);
        self.visit_logits_for_tokens_chunked(tokens, chunk_size, |logits| {
            for &logit in logits {
                bytes.extend_from_slice(&logit.to_bits().to_le_bytes());
            }
        })?;
        Ok(bytes)
    }

    pub fn visit_logits_for_tokens_chunked<F>(
        &self,
        tokens: &[usize],
        chunk_size: usize,
        mut visit: F,
    ) -> Result<(), ModelError>
    where
        F: FnMut(&[f32]),
    {
        self.validate_logits_request(tokens, chunk_size)?;
        let rope = RopeTables::llama(self.config, tokens.len())?;
        let mut cache = KvCache::new(self.config)?;
        let mut logits = vec![0.0f32; self.output.rows()];
        let mut pos = 0usize;
        for chunk in tokens.chunks(chunk_size) {
            for &token in chunk {
                self.forward_one(token, pos, &rope, &mut cache, &mut logits)?;
                visit(&logits);
                pos += 1;
            }
        }
        Ok(())
    }

    fn validate_logits_request(
        &self,
        tokens: &[usize],
        chunk_size: usize,
    ) -> Result<(), ModelError> {
        if tokens.is_empty() || chunk_size == 0 || tokens.len() > self.config.context_length {
            return Err(ModelError::Shape);
        }
        if tokens
            .iter()
            .any(|&token| token >= self.token_embedding.rows())
        {
            return Err(ModelError::Shape);
        }
        Ok(())
    }
}

fn checked_len(a: usize, b: usize) -> Result<usize, ModelError> {
    a.checked_mul(b).ok_or(ModelError::Shape)
}

fn checked_byte_len(items: usize, item_size: usize) -> Result<usize, ModelError> {
    items.checked_mul(item_size).ok_or(ModelError::Shape)
}

fn logits_byte_len(positions: usize, vocab: usize) -> Result<usize, ModelError> {
    checked_byte_len(checked_len(positions, vocab)?, core::mem::size_of::<f32>())
}

impl LlamaConfig {
    pub fn from_gguf(gguf: &det_gguf::Gguf) -> Result<Self, ModelError> {
        let arch = gguf
            .metadata_str("general.architecture")
            .map_err(map_metadata_error)?;
        let prefix = match arch {
            "llama" | "qwen2" => arch,
            _ => return Err(ModelError::UnsupportedArchitecture),
        };
        let rope_pairing = match arch {
            "llama" => RopePairing::Adjacent,
            "qwen2" => RopePairing::SplitHalf,
            _ => return Err(ModelError::UnsupportedArchitecture),
        };
        let key = |suffix: &str| format!("{prefix}.{suffix}");
        let head_count = required_u32(gguf, &key("attention.head_count"))? as usize;
        let head_count_kv = optional_u32(gguf, &key("attention.head_count_kv"))?
            .unwrap_or(head_count as u32) as usize;
        let embedding_length = required_u32(gguf, &key("embedding_length"))? as usize;
        if head_count == 0 || embedding_length % head_count != 0 {
            return Err(ModelError::Shape);
        }
        if optional_u32(gguf, &key("embedding_length_out"))?
            .is_some_and(|value| value as usize != embedding_length)
        {
            return Err(ModelError::UnsupportedModelFeature);
        }
        reject_false_optional_bool(gguf, &key("attention.causal"))?;
        if let Some(scaling_type) = optional_str(gguf, &key("rope.scaling.type"))? {
            if scaling_type != "none" {
                return Err(ModelError::UnsupportedRopeScaling);
            }
        }
        reject_true_optional_bool(gguf, &key("use_parallel_residual"))?;
        reject_nonzero_optional_u32(gguf, &key("attention.sliding_window"))?;
        reject_nonzero_optional_f32(gguf, &key("attention.max_alibi_bias"))?;
        reject_nonzero_optional_f32(gguf, &key("attention.clamp_kqv"))?;
        reject_nonzero_optional_f32(gguf, &key("attention.value_scale"))?;
        reject_nonzero_optional_f32(gguf, &key("attn_logit_softcapping"))?;
        reject_nonzero_optional_f32(gguf, &key("final_logit_softcapping"))?;
        reject_nonzero_optional_f32(gguf, &key("logit_scale"))?;
        reject_nonzero_optional_f32(gguf, &key("embedding_scale"))?;
        reject_nonzero_optional_f32(gguf, &key("residual_scale"))?;
        let head_dim = embedding_length / head_count;
        for length_key in ["attention.key_length", "attention.value_length"] {
            if optional_u32(gguf, &key(length_key))?.is_some_and(|value| value as usize != head_dim)
            {
                return Err(ModelError::UnsupportedAttentionHeadLength);
            }
        }
        let attention_scale = optional_f32(gguf, &key("attention.scale"))?
            .unwrap_or_else(|| default_attention_scale(head_dim));
        let config = Self {
            block_count: required_u32(gguf, &key("block_count"))? as usize,
            embedding_length,
            feed_forward_length: required_u32(gguf, &key("feed_forward_length"))? as usize,
            head_count,
            head_count_kv,
            rms_epsilon: required_f32(gguf, &key("attention.layer_norm_rms_epsilon"))?,
            attention_scale,
            rope_freq_base: optional_f32(gguf, &key("rope.freq_base"))?.unwrap_or(10_000.0),
            rope_dimension_count: optional_u32(gguf, &key("rope.dimension_count"))?
                .unwrap_or(head_dim as u32) as usize,
            rope_pairing,
            context_length: required_u32(gguf, &key("context_length"))? as usize,
        };
        config.validate()?;
        Ok(config)
    }

    pub fn head_dim(self) -> Result<usize, ModelError> {
        if self.head_count == 0 || self.embedding_length % self.head_count != 0 {
            return Err(ModelError::Shape);
        }
        Ok(self.embedding_length / self.head_count)
    }

    pub fn validate(self) -> Result<(), ModelError> {
        if self.block_count == 0
            || self.embedding_length == 0
            || self.feed_forward_length == 0
            || self.head_count == 0
            || self.head_count_kv == 0
            || self.context_length == 0
            || self.head_count % self.head_count_kv != 0
            || self.rope_dimension_count == 0
            || self.rope_dimension_count % 2 != 0
            || !self.rms_epsilon.is_finite()
            || self.rms_epsilon <= 0.0
            || !self.attention_scale.is_finite()
            || self.attention_scale <= 0.0
            || !self.rope_freq_base.is_finite()
            || self.rope_freq_base <= 0.0
        {
            return Err(ModelError::Shape);
        }
        if self.rope_dimension_count > self.head_dim()? {
            return Err(ModelError::Shape);
        }
        Ok(())
    }
}

impl<'a> F32TensorView<'a> {
    pub fn from_gguf(
        gguf: &'a det_gguf::Gguf,
        bytes: &'a [u8],
        name: &str,
    ) -> Result<Self, ModelError> {
        let info = gguf.tensor(name).map_err(|_| ModelError::Gguf)?;
        if info.ty != det_gguf::GgmlType::F32 {
            return Err(ModelError::UnsupportedTensorType);
        }
        let data = gguf
            .tensor_data(bytes, name)
            .map_err(|_| ModelError::Gguf)?;
        Ok(Self { info, data })
    }

    pub fn len(self) -> Result<usize, ModelError> {
        usize::try_from(self.info.element_count().map_err(|_| ModelError::Shape)?)
            .map_err(|_| ModelError::Shape)
    }

    pub fn is_empty(self) -> bool {
        self.info.element_count().unwrap_or(0) == 0
    }

    pub fn get(self, index: usize) -> Result<f32, ModelError> {
        if index >= self.len()? {
            return Err(ModelError::Shape);
        }
        let start = index * 4;
        let b = self.data.get(start..start + 4).ok_or(ModelError::Shape)?;
        let value = f32::from_le_bytes([b[0], b[1], b[2], b[3]]);
        if !value.is_finite() {
            return Err(ModelError::NonFinite);
        }
        Ok(value)
    }

    pub fn read_all(self) -> Result<Vec<f32>, ModelError> {
        let mut out = Vec::with_capacity(self.len()?);
        for chunk in self.data.chunks_exact(4) {
            let value = f32::from_le_bytes([chunk[0], chunk[1], chunk[2], chunk[3]]);
            if !value.is_finite() {
                return Err(ModelError::NonFinite);
            }
            out.push(value);
        }
        Ok(out)
    }
}

pub fn f32_gemv_from_view(
    matrix: F32TensorView<'_>,
    rows: usize,
    cols: usize,
    x: &[f32],
    out: &mut [f32],
) -> Result<(), ModelError> {
    if x.len() != cols || out.len() != rows || matrix.len()? != checked_len(rows, cols)? {
        return Err(ModelError::Shape);
    }
    ensure_finite_slice(x)?;
    let weights = matrix.read_all()?;
    gemv_rows(rows, out, |r| {
        Ok(dot_f32_ref(&weights[r * cols..(r + 1) * cols], x))
    })
}

fn vocab_size(gguf: &det_gguf::Gguf) -> Result<usize, ModelError> {
    let arch = gguf
        .metadata_str("general.architecture")
        .map_err(map_metadata_error)?;
    for key in [format!("{arch}.vocab_size"), "llama.vocab_size".to_owned()] {
        match gguf.metadata_u32(&key) {
            Ok(v) => return Ok(v as usize),
            Err(det_gguf::GgufError::MetadataNotFound) => {}
            Err(e) => return Err(map_metadata_error(e)),
        }
    }
    match gguf.metadata_value("tokenizer.ggml.tokens") {
        Ok(det_gguf::MetadataValue::ArrayString(tokens)) => return Ok(tokens.len()),
        Ok(_) => return Err(ModelError::Gguf),
        Err(det_gguf::GgufError::MetadataNotFound) => {}
        Err(e) => return Err(map_metadata_error(e)),
    }
    let token_embd = gguf
        .tensor("token_embd.weight")
        .map_err(|_| ModelError::Gguf)?;
    if token_embd.dimensions.len() == 2 {
        return usize::try_from(token_embd.dimensions[1]).map_err(|_| ModelError::Shape);
    }
    Err(ModelError::MissingMetadata)
}

fn read_f32_vector(
    gguf: &det_gguf::Gguf,
    bytes: &[u8],
    name: &str,
    len: usize,
) -> Result<Vec<f32>, ModelError> {
    let info = gguf.tensor(name).map_err(|_| ModelError::Gguf)?;
    if info.dimensions.as_slice() != [len as u64] {
        return Err(ModelError::Shape);
    }
    read_dense_tensor_as_f32(gguf, bytes, name)
}

fn read_f32_matrix(
    gguf: &det_gguf::Gguf,
    bytes: &[u8],
    name: &str,
    rows: usize,
    cols: usize,
) -> Result<F32Matrix, ModelError> {
    let info = gguf.tensor(name).map_err(|_| ModelError::Gguf)?;
    if info.dimensions.as_slice() != [cols as u64, rows as u64] {
        return Err(ModelError::Shape);
    }
    F32Matrix::new(rows, cols, read_dense_tensor_as_f32(gguf, bytes, name)?)
}

fn read_dense_tensor_as_f32(
    gguf: &det_gguf::Gguf,
    bytes: &[u8],
    name: &str,
) -> Result<Vec<f32>, ModelError> {
    let info = gguf.tensor(name).map_err(|_| ModelError::Gguf)?;
    let data = gguf
        .tensor_data(bytes, name)
        .map_err(|_| ModelError::Gguf)?;
    let len = usize::try_from(info.element_count().map_err(|_| ModelError::Shape)?)
        .map_err(|_| ModelError::Shape)?;
    match info.ty {
        det_gguf::GgmlType::F32 => {
            if data.len() != checked_byte_len(len, 4)? {
                return Err(ModelError::Shape);
            }
            let mut out = Vec::with_capacity(len);
            for chunk in data.chunks_exact(4) {
                let value = f32::from_le_bytes([chunk[0], chunk[1], chunk[2], chunk[3]]);
                if !value.is_finite() {
                    return Err(ModelError::NonFinite);
                }
                out.push(value);
            }
            Ok(out)
        }
        det_gguf::GgmlType::F16 => {
            if data.len() != checked_byte_len(len, 2)? {
                return Err(ModelError::Shape);
            }
            let mut out = Vec::with_capacity(len);
            for chunk in data.chunks_exact(2) {
                let bits = u16::from_le_bytes([chunk[0], chunk[1]]);
                if !f16_is_finite(bits) {
                    return Err(ModelError::NonFinite);
                }
                out.push(f16_to_f32(bits));
            }
            Ok(out)
        }
        _ => Err(ModelError::UnsupportedTensorType),
    }
}

fn read_weight_matrix(
    gguf: &det_gguf::Gguf,
    bytes: &[u8],
    name: &str,
    rows: usize,
    cols: usize,
) -> Result<WeightMatrix, ModelError> {
    let info = gguf.tensor(name).map_err(|_| ModelError::Gguf)?;
    if info.dimensions.as_slice() != [cols as u64, rows as u64] {
        return Err(ModelError::Shape);
    }
    match info.ty {
        det_gguf::GgmlType::F32 => Ok(WeightMatrix::F32(read_f32_matrix(
            gguf, bytes, name, rows, cols,
        )?)),
        det_gguf::GgmlType::F16 => Ok(WeightMatrix::F32(read_f32_matrix(
            gguf, bytes, name, rows, cols,
        )?)),
        det_gguf::GgmlType::Q8_0 => {
            read_q8_matrix(gguf, bytes, name, rows, cols).map(WeightMatrix::Q8_0)
        }
        det_gguf::GgmlType::Q4_0 => {
            read_q4_matrix(gguf, bytes, name, rows, cols).map(WeightMatrix::Q4_0)
        }
        _ => Err(ModelError::UnsupportedTensorType),
    }
}

fn read_q8_matrix(
    gguf: &det_gguf::Gguf,
    bytes: &[u8],
    name: &str,
    rows: usize,
    cols: usize,
) -> Result<Q8Matrix, ModelError> {
    if cols % BLOCK != 0 {
        return Err(ModelError::Shape);
    }
    let data = gguf
        .tensor_data(bytes, name)
        .map_err(|_| ModelError::Gguf)?;
    let blocks_per_row = cols / BLOCK;
    let expected_blocks = checked_len(rows, blocks_per_row)?;
    if data.len() != checked_byte_len(expected_blocks, 34)? {
        return Err(ModelError::Shape);
    }
    let mut blocks = Vec::with_capacity(expected_blocks);
    for chunk in data.chunks_exact(34) {
        let scale = u16::from_le_bytes([chunk[0], chunk[1]]);
        let mut q = [0i8; BLOCK];
        for i in 0..BLOCK {
            q[i] = chunk[2 + i] as i8;
        }
        blocks.push(q8_0_block_from_gguf(scale, q).map_err(|_| ModelError::NonFinite)?);
    }
    Ok(Q8Matrix {
        rows,
        cols,
        blocks_per_row,
        blocks,
    })
}

fn read_q4_matrix(
    gguf: &det_gguf::Gguf,
    bytes: &[u8],
    name: &str,
    rows: usize,
    cols: usize,
) -> Result<Q4Matrix, ModelError> {
    if cols % BLOCK != 0 {
        return Err(ModelError::Shape);
    }
    let data = gguf
        .tensor_data(bytes, name)
        .map_err(|_| ModelError::Gguf)?;
    let blocks_per_row = cols / BLOCK;
    let expected_blocks = checked_len(rows, blocks_per_row)?;
    if data.len() != checked_byte_len(expected_blocks, 18)? {
        return Err(ModelError::Shape);
    }
    let mut blocks = Vec::with_capacity(expected_blocks);
    for chunk in data.chunks_exact(18) {
        let scale = u16::from_le_bytes([chunk[0], chunk[1]]);
        let mut qs = [0u8; 16];
        qs.copy_from_slice(&chunk[2..18]);
        blocks.push(q4_0_block_from_gguf(scale, qs).map_err(|_| ModelError::NonFinite)?);
    }
    Ok(Q4Matrix {
        rows,
        cols,
        blocks_per_row,
        blocks,
    })
}

fn shared_q8a_if_needed<const N: usize>(
    x: &[f32],
    matrices: [&WeightMatrix; N],
) -> Result<Option<Vec<Q8ABlock>>, ModelError> {
    if matrices.iter().any(|matrix| matrix.needs_q8a()) {
        Ok(Some(quantize_q8a(x).map_err(map_quant_error)?))
    } else {
        Ok(None)
    }
}

fn residual_add(x: &mut [f32], residual: &[f32]) -> Result<(), ModelError> {
    if x.len() != residual.len() {
        return Err(ModelError::Shape);
    }
    ensure_finite_slice(x)?;
    ensure_finite_slice(residual)?;
    for (dst, &value) in x.iter_mut().zip(residual) {
        *dst += value;
        if !dst.is_finite() {
            return Err(ModelError::NonFinite);
        }
    }
    Ok(())
}

fn map_quant_error(e: det_quant::QuantError) -> ModelError {
    match e {
        det_quant::QuantError::NonFiniteInput
        | det_quant::QuantError::NonFiniteScale
        | det_quant::QuantError::NonFiniteOutput => ModelError::NonFinite,
        det_quant::QuantError::LengthMismatch | det_quant::QuantError::InvalidBlockLength => {
            ModelError::Shape
        }
    }
}

pub fn apply_rope(
    x: &mut [f32],
    heads: usize,
    head_dim: usize,
    pos: usize,
    rope: &RopeTables,
) -> Result<(), ModelError> {
    let expected_len = checked_len(heads, head_dim)?;
    let rope_width = checked_len(rope.half_head_dim, 2)?;
    if heads == 0
        || head_dim == 0
        || x.len() != expected_len
        || rope.half_head_dim == 0
        || rope_width > head_dim
    {
        return Err(ModelError::Shape);
    }
    ensure_finite_slice(x)?;
    let half = rope.half_head_dim;
    for head in 0..heads {
        let base = head * head_dim;
        for j in 0..half {
            let (first, second) = match rope.pairing {
                RopePairing::Adjacent => {
                    let first = base + 2 * j;
                    (first, first + 1)
                }
                RopePairing::SplitHalf => (base + j, base + j + half),
            };
            let x0 = x[first];
            let x1 = x[second];
            let (c, s) = rope.values(pos, j)?;
            x[first] = x0 * c - x1 * s;
            x[second] = x0 * s + x1 * c;
            if !x[first].is_finite() || !x[second].is_finite() {
                return Err(ModelError::NonFinite);
            }
        }
    }
    Ok(())
}

pub fn rmsnorm(x: &[f32], weight: &[f32], epsilon: f32, out: &mut [f32]) -> Result<(), ModelError> {
    if x.len() != weight.len() || x.len() != out.len() || x.is_empty() {
        return Err(ModelError::Shape);
    }
    if x.iter().chain(weight).any(|v| !v.is_finite()) || !epsilon.is_finite() {
        return Err(ModelError::NonFinite);
    }
    let ss = sum_squares_f32_ref(x);
    let m = ss / (x.len() as f32);
    let r = 1.0 / (m + epsilon).sqrt();
    if !r.is_finite() {
        return Err(ModelError::NonFinite);
    }
    for i in 0..x.len() {
        out[i] = (x[i] * r) * weight[i];
        if !out[i].is_finite() {
            return Err(ModelError::NonFinite);
        }
    }
    Ok(())
}

fn required_u32(gguf: &det_gguf::Gguf, key: &str) -> Result<u32, ModelError> {
    gguf.metadata_u32(key).map_err(map_metadata_error)
}

fn optional_u32(gguf: &det_gguf::Gguf, key: &str) -> Result<Option<u32>, ModelError> {
    match gguf.metadata_u32(key) {
        Ok(v) => Ok(Some(v)),
        Err(det_gguf::GgufError::MetadataNotFound) => Ok(None),
        Err(e) => Err(map_metadata_error(e)),
    }
}

fn required_f32(gguf: &det_gguf::Gguf, key: &str) -> Result<f32, ModelError> {
    gguf.metadata_f32(key).map_err(map_metadata_error)
}

fn optional_f32(gguf: &det_gguf::Gguf, key: &str) -> Result<Option<f32>, ModelError> {
    match gguf.metadata_f32(key) {
        Ok(v) => Ok(Some(v)),
        Err(det_gguf::GgufError::MetadataNotFound) => Ok(None),
        Err(e) => Err(map_metadata_error(e)),
    }
}

fn optional_str<'a>(gguf: &'a det_gguf::Gguf, key: &str) -> Result<Option<&'a str>, ModelError> {
    match gguf.metadata_str(key) {
        Ok(v) => Ok(Some(v)),
        Err(det_gguf::GgufError::MetadataNotFound) => Ok(None),
        Err(e) => Err(map_metadata_error(e)),
    }
}

fn optional_bool(gguf: &det_gguf::Gguf, key: &str) -> Result<Option<bool>, ModelError> {
    match gguf.metadata_value(key) {
        Ok(det_gguf::MetadataValue::Bool(v)) => Ok(Some(*v)),
        Ok(_) => Err(ModelError::Gguf),
        Err(det_gguf::GgufError::MetadataNotFound) => Ok(None),
        Err(e) => Err(map_metadata_error(e)),
    }
}

fn reject_true_optional_bool(gguf: &det_gguf::Gguf, key: &str) -> Result<(), ModelError> {
    if optional_bool(gguf, key)?.unwrap_or(false) {
        return Err(ModelError::UnsupportedModelFeature);
    }
    Ok(())
}

fn reject_false_optional_bool(gguf: &det_gguf::Gguf, key: &str) -> Result<(), ModelError> {
    if optional_bool(gguf, key)?.is_some_and(|value| !value) {
        return Err(ModelError::UnsupportedModelFeature);
    }
    Ok(())
}

fn reject_nonzero_optional_u32(gguf: &det_gguf::Gguf, key: &str) -> Result<(), ModelError> {
    if optional_u32(gguf, key)?.unwrap_or(0) != 0 {
        return Err(ModelError::UnsupportedModelFeature);
    }
    Ok(())
}

fn reject_nonzero_optional_f32(gguf: &det_gguf::Gguf, key: &str) -> Result<(), ModelError> {
    let value = match gguf.metadata_value(key) {
        Ok(det_gguf::MetadataValue::F32(value)) => *value as f64,
        Ok(det_gguf::MetadataValue::F64(value)) => *value,
        Ok(_) => return Err(ModelError::Gguf),
        Err(det_gguf::GgufError::MetadataNotFound) => return Ok(()),
        Err(e) => return Err(map_metadata_error(e)),
    };
    if !value.is_finite() {
        return Err(ModelError::Shape);
    }
    if value != 0.0 {
        return Err(ModelError::UnsupportedModelFeature);
    }
    Ok(())
}

fn default_attention_scale(head_dim: usize) -> f32 {
    1.0 / (head_dim as f32).sqrt()
}

fn map_metadata_error(e: det_gguf::GgufError) -> ModelError {
    match e {
        det_gguf::GgufError::MetadataNotFound => ModelError::MissingMetadata,
        _ => ModelError::Gguf,
    }
}

pub fn attention_scores_one_head(
    q: &[f32],
    keys: &[f32],
    head_dim: usize,
    out: &mut [f32],
) -> Result<(), ModelError> {
    attention_scores_one_head_scaled(q, keys, head_dim, default_attention_scale(head_dim), out)
}

pub fn attention_scores_one_head_scaled(
    q: &[f32],
    keys: &[f32],
    head_dim: usize,
    scale: f32,
    out: &mut [f32],
) -> Result<(), ModelError> {
    let expected_keys = checked_len(out.len(), head_dim)?;
    if head_dim == 0 || out.is_empty() || q.len() != head_dim || keys.len() != expected_keys {
        return Err(ModelError::Shape);
    }
    if !scale.is_finite() || scale <= 0.0 {
        return Err(ModelError::Shape);
    }
    if q.iter().chain(keys).any(|v| !v.is_finite()) {
        return Err(ModelError::NonFinite);
    }
    for (j, key) in keys.chunks_exact(head_dim).enumerate() {
        out[j] = dot_f32_ref(q, key) * scale;
        if !out[j].is_finite() {
            return Err(ModelError::NonFinite);
        }
    }
    Ok(())
}

pub fn softmax_in_place(x: &mut [f32]) -> Result<(), ModelError> {
    if x.is_empty() {
        return Err(ModelError::Shape);
    }
    let mut max = f32::NEG_INFINITY;
    for &v in x.iter() {
        if !v.is_finite() {
            return Err(ModelError::NonFinite);
        }
        if v > max {
            max = v;
        }
    }
    for v in x.iter_mut() {
        *v = exp_f32((*v - max).max(-88.0));
    }
    let z = sum_f32_ref(x);
    if !z.is_finite() || z == 0.0 {
        return Err(ModelError::NonFinite);
    }
    for v in x.iter_mut() {
        *v /= z;
        if !v.is_finite() {
            return Err(ModelError::NonFinite);
        }
    }
    Ok(())
}

pub fn attention_weighted_value(
    p: &[f32],
    values: &[f32],
    head_dim: usize,
    out: &mut [f32],
) -> Result<(), ModelError> {
    let expected_values = checked_len(p.len(), head_dim)?;
    if head_dim == 0 || p.is_empty() || out.len() != head_dim || values.len() != expected_values {
        return Err(ModelError::Shape);
    }
    if p.iter().chain(values).any(|v| !v.is_finite()) {
        return Err(ModelError::NonFinite);
    }
    out.fill(0.0);
    for (j, value) in values.chunks_exact(head_dim).enumerate() {
        let pj = p[j];
        for d in 0..head_dim {
            out[d] += pj * value[d];
            if !out[d].is_finite() {
                return Err(ModelError::NonFinite);
            }
        }
    }
    Ok(())
}

pub fn swiglu(gate: &[f32], up: &[f32], out: &mut [f32]) -> Result<(), ModelError> {
    if gate.len() != up.len() || gate.len() != out.len() {
        return Err(ModelError::Shape);
    }
    if gate.iter().chain(up).any(|v| !v.is_finite()) {
        return Err(ModelError::NonFinite);
    }
    for i in 0..gate.len() {
        out[i] = det_num::silu_f32(gate[i]) * up[i];
        if !out[i].is_finite() {
            return Err(ModelError::NonFinite);
        }
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::collections::BTreeMap;
    use std::path::{Path, PathBuf};

    #[test]
    fn rmsnorm_runs_in_specified_order() {
        let x = [1.0, -2.0, 3.0, -4.0];
        let w = [0.5, 1.0, 1.5, 2.0];
        let mut out = [0.0; 4];
        rmsnorm(&x, &w, 1e-5, &mut out).expect("rmsnorm");
        let ss = det_num::sum_squares_f32_ref(&x);
        let r = 1.0 / (ss / 4.0 + 1e-5).sqrt();
        let expected = [
            (x[0] * r) * w[0],
            (x[1] * r) * w[1],
            (x[2] * r) * w[2],
            (x[3] * r) * w[3],
        ];
        for i in 0..4 {
            assert_eq!(out[i].to_bits(), expected[i].to_bits());
        }
    }

    #[test]
    fn softmax_uses_det_exp_and_8lane_sum() {
        let mut x = [1.0, 0.0, -2.0, 3.0, -88.0, 3.0, -1.0, 2.0, 0.5];
        softmax_in_place(&mut x).expect("softmax");
        let z = det_num::sum_f32_ref(&x);
        assert!((z - 1.0).abs() < 0.000001);
        assert_eq!(x[3].to_bits(), x[5].to_bits());
    }

    #[test]
    fn public_kernels_reject_nonfinite_inputs() {
        let mut rms = [0.0; 2];
        assert_eq!(
            rmsnorm(&[1.0, f32::NAN], &[1.0, 1.0], 1e-5, &mut rms),
            Err(ModelError::NonFinite)
        );

        let mut scores = [0.0; 1];
        assert_eq!(
            attention_scores_one_head(&[1.0, f32::INFINITY], &[1.0, 2.0], 2, &mut scores),
            Err(ModelError::NonFinite)
        );
        assert_eq!(
            attention_scores_one_head(&[1.0], &[1.0, 2.0], 2, &mut scores),
            Err(ModelError::Shape)
        );
        attention_scores_one_head_scaled(&[2.0, 4.0], &[3.0, 5.0], 2, 0.25, &mut scores)
            .expect("scaled attention score");
        assert_eq!(scores[0].to_bits(), 6.5f32.to_bits());
        assert_eq!(
            attention_scores_one_head_scaled(&[1.0, 2.0], &[3.0, 4.0], 2, 0.0, &mut scores),
            Err(ModelError::Shape)
        );

        let mut out = [0.0; 2];
        assert_eq!(
            attention_weighted_value(&[1.0], &[1.0, f32::NAN], 2, &mut out),
            Err(ModelError::NonFinite)
        );
        assert_eq!(
            attention_weighted_value(&[1.0], &[1.0], 2, &mut out),
            Err(ModelError::Shape)
        );

        let mut sw = [0.0; 2];
        assert_eq!(
            swiglu(&[1.0, f32::NAN], &[1.0, 2.0], &mut sw),
            Err(ModelError::NonFinite)
        );

        let mut sm = [0.0, f32::INFINITY];
        assert_eq!(softmax_in_place(&mut sm), Err(ModelError::NonFinite));
    }

    #[test]
    fn public_kernels_reject_shape_overflow_or_empty_attention() {
        let rope = RopeTables::identity(1, 2).expect("rope");
        let mut empty = [];
        assert_eq!(
            apply_rope(&mut empty, usize::MAX, 2, 0, &rope),
            Err(ModelError::Shape)
        );

        let huge_rope = RopeTables {
            positions: 1,
            half_head_dim: usize::MAX,
            pairing: RopePairing::Adjacent,
            cos: Vec::new(),
            sin: Vec::new(),
        };
        assert_eq!(
            apply_rope(&mut empty, 0, 0, 0, &huge_rope),
            Err(ModelError::Shape)
        );

        let mut no_scores = [];
        assert_eq!(
            attention_scores_one_head_scaled(&[1.0], &[], 1, 1.0, &mut no_scores),
            Err(ModelError::Shape)
        );

        let mut out = [0.0];
        assert_eq!(
            attention_weighted_value(&[], &[], 1, &mut out),
            Err(ModelError::Shape)
        );
    }

    #[test]
    fn gemv_and_residual_add_reject_nonfinite_outputs() {
        assert_eq!(
            F32Matrix::new(1, 1, vec![f32::NAN]),
            Err(ModelError::NonFinite)
        );

        let m = F32Matrix::new(1, 1, vec![1.0]).expect("matrix");
        let mut out = [0.0];
        assert_eq!(m.gemv(&[f32::NAN], &mut out), Err(ModelError::NonFinite));

        let huge = F32Matrix::new(1, 1, vec![f32::MAX]).expect("huge matrix");
        assert_eq!(huge.gemv(&[2.0], &mut out), Err(ModelError::NonFinite));

        let malformed_f32 = F32Matrix {
            rows: 1,
            cols: 2,
            data: vec![1.0],
        };
        let mut malformed_out = [0.0];
        assert_eq!(malformed_f32.row(0), Err(ModelError::Shape));
        assert_eq!(
            malformed_f32.gemv(&[1.0, 2.0], &mut malformed_out),
            Err(ModelError::Shape)
        );
        let nonfinite_f32 = F32Matrix {
            rows: 1,
            cols: 1,
            data: vec![f32::INFINITY],
        };
        assert_eq!(nonfinite_f32.row(0), Err(ModelError::NonFinite));
        assert_eq!(
            nonfinite_f32.gemv(&[1.0], &mut malformed_out),
            Err(ModelError::NonFinite)
        );

        let q8 = WeightMatrix::Q8_0(Q8Matrix {
            rows: 1,
            cols: BLOCK,
            blocks_per_row: 1,
            blocks: vec![Q8_0Block {
                d: 1.0,
                q: [1; BLOCK],
            }],
        });
        let mut qout = [0.0];
        let mut bad = [0.0; BLOCK];
        bad[0] = f32::INFINITY;
        assert_eq!(q8.gemv(&bad, &mut qout), Err(ModelError::NonFinite));

        let q8_overflow = WeightMatrix::Q8_0(Q8Matrix {
            rows: 1,
            cols: BLOCK,
            blocks_per_row: 1,
            blocks: vec![Q8_0Block {
                d: f32::MAX,
                q: [127; BLOCK],
            }],
        });
        let q4_overflow = WeightMatrix::Q4_0(Q4Matrix {
            rows: 1,
            cols: BLOCK,
            blocks_per_row: 1,
            blocks: vec![Q4_0Block {
                d: f32::MAX,
                qs: [0xff; BLOCK / 2],
            }],
        });
        let x = [127.0; BLOCK];
        assert_eq!(q8_overflow.gemv(&x, &mut qout), Err(ModelError::NonFinite));
        assert_eq!(q4_overflow.gemv(&x, &mut qout), Err(ModelError::NonFinite));
        assert_eq!(q8_overflow.row_to_vec(0), Err(ModelError::NonFinite));

        let malformed_q8 = WeightMatrix::Q8_0(Q8Matrix {
            rows: 1,
            cols: BLOCK,
            blocks_per_row: 1,
            blocks: Vec::new(),
        });
        assert_eq!(
            malformed_q8.gemv(&[0.0; BLOCK], &mut qout),
            Err(ModelError::Shape)
        );
        assert_eq!(malformed_q8.row_to_vec(0), Err(ModelError::Shape));

        let mut x = [f32::MAX];
        assert_eq!(
            residual_add(&mut x, &[f32::MAX]),
            Err(ModelError::NonFinite)
        );
        assert_eq!(
            residual_add(&mut [0.0], &[0.0, 1.0]),
            Err(ModelError::Shape)
        );
    }

    #[test]
    fn kv_cache_store_rejects_nonfinite_values() {
        let cfg = LlamaConfig {
            block_count: 1,
            embedding_length: 4,
            feed_forward_length: 4,
            head_count: 2,
            head_count_kv: 1,
            rms_epsilon: 1e-5,
            attention_scale: default_attention_scale(2),
            rope_freq_base: 10_000.0,
            rope_dimension_count: 2,
            rope_pairing: RopePairing::Adjacent,
            context_length: 2,
        };
        let mut cache = KvCache::new(cfg).expect("cache");
        assert_eq!(
            cache.store(0, 0, &[f32::NAN, 0.0], &[0.0, 0.0]),
            Err(ModelError::NonFinite)
        );
        assert_eq!(
            cache.store(0, 0, &[0.0, 0.0], &[0.0, f32::INFINITY]),
            Err(ModelError::NonFinite)
        );
    }

    #[test]
    fn kv_cache_rejects_out_of_bounds_indices_and_bad_lengths() {
        let cfg = LlamaConfig {
            block_count: 1,
            embedding_length: 4,
            feed_forward_length: 4,
            head_count: 2,
            head_count_kv: 1,
            rms_epsilon: 1e-5,
            attention_scale: default_attention_scale(2),
            rope_freq_base: 10_000.0,
            rope_dimension_count: 2,
            rope_pairing: RopePairing::Adjacent,
            context_length: 2,
        };
        let mut cache = KvCache::new(cfg).expect("cache");
        let k = [0.1, 0.2];
        let v = [0.3, 0.4];
        cache.store(0, 0, &k, &v).expect("valid store");

        assert_eq!(cache.store(0, 0, &[0.1], &v), Err(ModelError::Shape));
        assert_eq!(cache.store(0, 0, &k, &[0.3]), Err(ModelError::Shape));
        assert_eq!(cache.store(1, 0, &k, &v), Err(ModelError::Shape));
        assert_eq!(cache.store(0, 2, &k, &v), Err(ModelError::Shape));

        assert_eq!(cache.key(1, 0, 0), Err(ModelError::Shape));
        assert_eq!(cache.key(0, 1, 0), Err(ModelError::Shape));
        assert_eq!(cache.key(0, 0, 2), Err(ModelError::Shape));
        assert_eq!(cache.value(1, 0, 0), Err(ModelError::Shape));
        assert_eq!(cache.value(0, 1, 0), Err(ModelError::Shape));
        assert_eq!(cache.value(0, 0, 2), Err(ModelError::Shape));
    }

    #[test]
    fn rope_applies_adjacent_pairs_in_specified_order() {
        let rope = RopeTables {
            positions: 2,
            half_head_dim: 2,
            pairing: RopePairing::Adjacent,
            cos: vec![1.0, 1.0, 0.0, 0.5],
            sin: vec![0.0, 0.0, 1.0, -0.25],
        };
        let mut x = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0];
        apply_rope(&mut x, 2, 4, 1, &rope).expect("rope");

        let expected = [-2.0f32, 1.0, 2.5, 1.25, -6.0, 5.0, 5.5, 2.25];
        for i in 0..x.len() {
            assert_eq!(x[i].to_bits(), expected[i].to_bits(), "index {i}");
        }
    }

    #[test]
    fn rope_applies_split_half_pairs_for_qwen2_style() {
        let rope = RopeTables {
            positions: 2,
            half_head_dim: 2,
            pairing: RopePairing::SplitHalf,
            cos: vec![1.0, 1.0, 0.0, 0.5],
            sin: vec![0.0, 0.0, 1.0, -0.25],
        };
        let mut x = [1.0, 2.0, 3.0, 4.0];
        apply_rope(&mut x, 1, 4, 1, &rope).expect("rope");

        let expected = [-3.0f32, 2.0, 1.0, 1.5];
        for i in 0..x.len() {
            assert_eq!(x[i].to_bits(), expected[i].to_bits(), "index {i}");
        }
    }

    #[test]
    fn rope_leaves_non_rotary_tail_unchanged() {
        let rope = RopeTables {
            positions: 2,
            half_head_dim: 2,
            pairing: RopePairing::Adjacent,
            cos: vec![1.0, 1.0, 0.0, 0.5],
            sin: vec![0.0, 0.0, 1.0, -0.25],
        };
        let mut x = [1.0, 2.0, 3.0, 4.0, 9.0, -9.0];
        apply_rope(&mut x, 1, 6, 1, &rope).expect("partial adjacent rope");
        let expected = [-2.0f32, 1.0, 2.5, 1.25, 9.0, -9.0];
        for i in 0..x.len() {
            assert_eq!(x[i].to_bits(), expected[i].to_bits(), "adjacent index {i}");
        }

        let rope = RopeTables {
            positions: 2,
            half_head_dim: 2,
            pairing: RopePairing::SplitHalf,
            cos: vec![1.0, 1.0, 0.0, 0.5],
            sin: vec![0.0, 0.0, 1.0, -0.25],
        };
        let mut x = [1.0, 2.0, 3.0, 4.0, 9.0, -9.0];
        apply_rope(&mut x, 1, 6, 1, &rope).expect("partial split-half rope");
        let expected = [-3.0f32, 2.0, 1.0, 1.5, 9.0, -9.0];
        for i in 0..x.len() {
            assert_eq!(
                x[i].to_bits(),
                expected[i].to_bits(),
                "split-half index {i}"
            );
        }
    }

    #[test]
    fn rope_rejects_malformed_or_nonfinite_values() {
        let short = RopeTables {
            positions: 1,
            half_head_dim: 1,
            pairing: RopePairing::Adjacent,
            cos: Vec::new(),
            sin: vec![0.0],
        };
        assert_eq!(short.values(0, 0), Err(ModelError::Shape));

        let nonfinite = RopeTables {
            positions: 1,
            half_head_dim: 1,
            pairing: RopePairing::Adjacent,
            cos: vec![f32::NAN],
            sin: vec![0.0],
        };
        assert_eq!(nonfinite.values(0, 0), Err(ModelError::NonFinite));

        let rope = RopeTables::identity(1, 2).expect("rope");
        let mut x = [f32::INFINITY, 0.0];
        assert_eq!(
            apply_rope(&mut x, 1, 2, 0, &rope),
            Err(ModelError::NonFinite)
        );

        let rope = RopeTables {
            positions: 1,
            half_head_dim: 1,
            pairing: RopePairing::Adjacent,
            cos: vec![1.0],
            sin: vec![1.0],
        };
        let mut x = [f32::MAX, -f32::MAX];
        assert_eq!(
            apply_rope(&mut x, 1, 2, 0, &rope),
            Err(ModelError::NonFinite)
        );
    }

    #[test]
    fn f32_llama_forward_one_runs_through_kv_cache() {
        let cfg = LlamaConfig {
            block_count: 1,
            embedding_length: 4,
            feed_forward_length: 6,
            head_count: 2,
            head_count_kv: 1,
            rms_epsilon: 1e-5,
            attention_scale: default_attention_scale(2),
            rope_freq_base: 10_000.0,
            rope_dimension_count: 2,
            rope_pairing: RopePairing::Adjacent,
            context_length: 4,
        };
        let model = F32Llama {
            config: cfg,
            token_embedding: F32Matrix::new(
                3,
                4,
                vec![
                    0.1, 0.2, 0.3, 0.4, //
                    -0.2, 0.5, 0.7, -0.1, //
                    0.9, -0.3, 0.2, 0.8,
                ],
            )
            .expect("emb")
            .into(),
            layers: vec![F32LayerWeights {
                attention_norm: vec![1.0; 4],
                wq: patterned_matrix(4, 4, 0.01).into(),
                wk: patterned_matrix(2, 4, -0.02).into(),
                wv: patterned_matrix(2, 4, 0.03).into(),
                wo: patterned_matrix(4, 4, -0.015).into(),
                ffn_norm: vec![1.0; 4],
                w_gate: patterned_matrix(6, 4, 0.025).into(),
                w_up: patterned_matrix(6, 4, -0.018).into(),
                w_down: patterned_matrix(4, 6, 0.012).into(),
            }],
            output_norm: vec![1.0; 4],
            output: patterned_matrix(3, 4, 0.02).into(),
        };
        let rope = RopeTables::identity(4, 2).expect("rope");
        let mut cache = KvCache::new(cfg).expect("cache");
        let mut logits0 = [0.0f32; 3];
        let mut logits1 = [0.0f32; 3];
        model
            .forward_one(0, 0, &rope, &mut cache, &mut logits0)
            .expect("forward pos0");
        model
            .forward_one(1, 1, &rope, &mut cache, &mut logits1)
            .expect("forward pos1");
        assert!(logits0.iter().chain(logits1.iter()).all(|x| x.is_finite()));
        assert_ne!(logits0.map(f32::to_bits), logits1.map(f32::to_bits));
    }

    #[test]
    fn forward_one_rejects_mismatched_rope_or_cache_config() {
        let model = small_valid_model();
        let mut cache = KvCache::new(model.config).expect("cache");
        let mut logits = vec![0.0; model.output.rows()];

        let mut wrong_pairing = RopeTables::llama(model.config, 1).expect("rope");
        wrong_pairing.pairing = RopePairing::SplitHalf;
        assert_eq!(
            model.forward_one(0, 0, &wrong_pairing, &mut cache, &mut logits),
            Err(ModelError::Shape)
        );

        let rope = RopeTables::llama(model.config, 1).expect("rope");
        let mut other_config = model.config;
        other_config.context_length += 1;
        let mut wrong_cache = KvCache::new(other_config).expect("cache");
        assert_eq!(
            model.forward_one(0, 0, &rope, &mut wrong_cache, &mut logits),
            Err(ModelError::Shape)
        );
    }

    #[test]
    fn validate_rejects_mismatched_input_and_output_vocab_rows() {
        let mut model = small_valid_model();
        model.output = patterned_matrix(4, 4, 0.02).into();

        assert_eq!(model.validate(), Err(ModelError::Shape));
    }

    #[test]
    fn validate_rejects_nonfinite_or_malformed_model_weights() {
        let mut nonfinite_norm = small_valid_model();
        nonfinite_norm.layers[0].attention_norm[0] = f32::NAN;
        assert_eq!(nonfinite_norm.validate(), Err(ModelError::NonFinite));

        let mut nonfinite_weight = small_valid_model();
        let mut data = vec![0.0f32; 12];
        data[0] = f32::INFINITY;
        nonfinite_weight.output = WeightMatrix::F32(F32Matrix {
            rows: 3,
            cols: 4,
            data,
        });
        assert_eq!(nonfinite_weight.validate(), Err(ModelError::NonFinite));

        let mut malformed_quant = small_valid_model();
        malformed_quant.layers[0].wq = WeightMatrix::Q8_0(Q8Matrix {
            rows: 4,
            cols: 4,
            blocks_per_row: 1,
            blocks: vec![
                Q8_0Block {
                    d: 1.0,
                    q: [0; BLOCK],
                };
                4
            ],
        });
        assert_eq!(malformed_quant.validate(), Err(ModelError::Shape));
    }

    fn small_valid_model() -> F32Llama {
        let cfg = LlamaConfig {
            block_count: 1,
            embedding_length: 4,
            feed_forward_length: 6,
            head_count: 2,
            head_count_kv: 1,
            rms_epsilon: 1e-5,
            attention_scale: default_attention_scale(2),
            rope_freq_base: 10_000.0,
            rope_dimension_count: 2,
            rope_pairing: RopePairing::Adjacent,
            context_length: 4,
        };
        F32Llama {
            config: cfg,
            token_embedding: patterned_matrix(3, 4, 0.01).into(),
            layers: vec![F32LayerWeights {
                attention_norm: vec![1.0; 4],
                wq: patterned_matrix(4, 4, 0.01).into(),
                wk: patterned_matrix(2, 4, -0.02).into(),
                wv: patterned_matrix(2, 4, 0.03).into(),
                wo: patterned_matrix(4, 4, -0.015).into(),
                ffn_norm: vec![1.0; 4],
                w_gate: patterned_matrix(6, 4, 0.025).into(),
                w_up: patterned_matrix(6, 4, -0.018).into(),
                w_down: patterned_matrix(4, 6, 0.012).into(),
            }],
            output_norm: vec![1.0; 4],
            output: patterned_matrix(3, 4, 0.02).into(),
        }
    }

    fn patterned_matrix(rows: usize, cols: usize, scale: f32) -> F32Matrix {
        let data = (0..rows * cols)
            .map(|i| (((i % 7) as f32) - 3.0) * scale)
            .collect();
        F32Matrix::new(rows, cols, data).expect("matrix")
    }

    #[test]
    fn loads_f32_llama_from_gguf_parts_and_hashes_logits() {
        let cfg = LlamaConfig {
            block_count: 1,
            embedding_length: 4,
            feed_forward_length: 6,
            head_count: 2,
            head_count_kv: 1,
            rms_epsilon: 1e-5,
            attention_scale: default_attention_scale(2),
            rope_freq_base: 10_000.0,
            rope_dimension_count: 2,
            rope_pairing: RopePairing::Adjacent,
            context_length: 4,
        };
        let (gguf, bytes) = synthetic_f32_gguf(cfg);
        let model = F32Llama::from_gguf(&gguf, &bytes).expect("model");
        assert_eq!(model.token_embedding.rows(), 3);
        assert_eq!(model.output.rows(), 3);
        let hash = model.logits_hash_for_tokens(&[0, 1]).expect("hash");
        assert_ne!(hash, [0u8; 32]);
    }

    #[test]
    fn vocab_size_rejects_malformed_metadata_before_fallback() {
        let mut metadata = BTreeMap::new();
        metadata.insert(
            "general.architecture".to_owned(),
            det_gguf::MetadataValue::String("llama".to_owned()),
        );
        metadata.insert(
            "llama.vocab_size".to_owned(),
            det_gguf::MetadataValue::String("3".to_owned()),
        );
        metadata.insert(
            "tokenizer.ggml.tokens".to_owned(),
            det_gguf::MetadataValue::ArrayString(vec!["a".to_owned(), "b".to_owned()]),
        );
        let gguf = det_gguf::Gguf::from_parts(3, metadata, Vec::new(), 0, 0);
        assert_eq!(vocab_size(&gguf), Err(ModelError::Gguf));

        let mut metadata = BTreeMap::new();
        metadata.insert(
            "general.architecture".to_owned(),
            det_gguf::MetadataValue::String("llama".to_owned()),
        );
        metadata.insert(
            "tokenizer.ggml.tokens".to_owned(),
            det_gguf::MetadataValue::ArrayU32(vec![0, 1, 2]),
        );
        let mut tensors = Vec::new();
        let mut bytes = Vec::new();
        push_tensor(&mut tensors, &mut bytes, "token_embd.weight", 3, 4, 0.01);
        let gguf = det_gguf::Gguf::from_parts(3, metadata, tensors, 0, bytes.len());
        assert_eq!(vocab_size(&gguf), Err(ModelError::Gguf));
    }

    #[test]
    fn logits_bytes_hash_matches_hash_api() {
        let cfg = LlamaConfig {
            block_count: 1,
            embedding_length: 4,
            feed_forward_length: 6,
            head_count: 2,
            head_count_kv: 1,
            rms_epsilon: 1e-5,
            attention_scale: default_attention_scale(2),
            rope_freq_base: 10_000.0,
            rope_dimension_count: 2,
            rope_pairing: RopePairing::Adjacent,
            context_length: 4,
        };
        let (gguf, bytes) = synthetic_f32_gguf(cfg);
        let model = F32Llama::from_gguf(&gguf, &bytes).expect("model");
        let logits = model
            .logits_bytes_for_tokens_chunked(&[0, 1], 2)
            .expect("logits bytes");
        assert_eq!(logits.len(), 2 * model.output.rows() * 4);

        let mut hash = Sha256::new();
        hash.update(&logits);
        assert_eq!(
            hash.finalize(),
            model
                .logits_hash_for_tokens_chunked(&[0, 1], 2)
                .expect("hash")
        );
    }

    #[test]
    fn logits_request_rejects_token_ids_outside_vocabulary() {
        let cfg = LlamaConfig {
            block_count: 1,
            embedding_length: 4,
            feed_forward_length: 6,
            head_count: 2,
            head_count_kv: 1,
            rms_epsilon: 1e-5,
            attention_scale: default_attention_scale(2),
            rope_freq_base: 10_000.0,
            rope_dimension_count: 2,
            rope_pairing: RopePairing::Adjacent,
            context_length: 4,
        };
        let (gguf, bytes) = synthetic_f32_gguf(cfg);
        let model = F32Llama::from_gguf(&gguf, &bytes).expect("model");

        assert_eq!(model.logits_hash_for_tokens(&[3]), Err(ModelError::Shape));
        assert_eq!(
            model.logits_bytes_for_tokens_chunked(&[0, 3], 1),
            Err(ModelError::Shape)
        );
    }

    #[test]
    fn loads_qwen2_prefixed_dense_decoder_metadata() {
        let cfg = LlamaConfig {
            block_count: 1,
            embedding_length: 4,
            feed_forward_length: 6,
            head_count: 2,
            head_count_kv: 1,
            rms_epsilon: 1e-6,
            attention_scale: default_attention_scale(2),
            rope_freq_base: 1_000_000.0,
            rope_dimension_count: 2,
            rope_pairing: RopePairing::SplitHalf,
            context_length: 4,
        };
        let (gguf, bytes) = synthetic_f32_gguf_with_arch(cfg, true, "qwen2");
        let loaded_cfg = LlamaConfig::from_gguf(&gguf).expect("config");
        assert_eq!(loaded_cfg, cfg);

        let model = F32Llama::from_gguf(&gguf, &bytes).expect("model");
        assert_eq!(model.token_embedding.rows(), 3);
        assert_eq!(model.output.rows(), 3);
        let hash = model.logits_hash_for_tokens(&[0, 1]).expect("hash");
        assert_ne!(hash, [0u8; 32]);
    }

    #[test]
    fn loads_f16_dense_tensors_as_f32() {
        let cfg = LlamaConfig {
            block_count: 1,
            embedding_length: 4,
            feed_forward_length: 6,
            head_count: 2,
            head_count_kv: 1,
            rms_epsilon: 1e-5,
            attention_scale: default_attention_scale(2),
            rope_freq_base: 10_000.0,
            rope_dimension_count: 2,
            rope_pairing: RopePairing::Adjacent,
            context_length: 4,
        };
        let (gguf, bytes) = synthetic_f16_gguf(cfg);
        let model = F32Llama::from_gguf(&gguf, &bytes).expect("model");
        assert!(matches!(model.token_embedding, WeightMatrix::F32(_)));
        assert!(matches!(model.output, WeightMatrix::F32(_)));
        let hash = model.logits_hash_for_tokens(&[0, 1]).expect("hash");
        assert_ne!(hash, [0u8; 32]);
    }

    #[test]
    fn rejects_nonfinite_f16_dense_tensor_values() {
        let mut tensors = Vec::new();
        let mut bytes = Vec::new();
        push_f16_vector(&mut tensors, &mut bytes, "bad.weight", 2, 0x7c00);
        let gguf = det_gguf::Gguf::from_parts(3, BTreeMap::new(), tensors, 0, bytes.len());

        assert_eq!(
            read_f32_vector(&gguf, &bytes, "bad.weight", 2),
            Err(ModelError::NonFinite)
        );
    }

    #[test]
    fn f32_gemv_from_view_rejects_nonfinite_boundaries() {
        let (gguf, bytes) = single_f32_tensor_gguf("w.weight", &[1.0, 2.0], &[2, 1]);
        let view = F32TensorView::from_gguf(&gguf, &bytes, "w.weight").expect("view");
        let mut out = [0.0f32];
        f32_gemv_from_view(view, 1, 2, &[3.0, 4.0], &mut out).expect("gemv");
        assert_eq!(out[0].to_bits(), 11.0f32.to_bits());

        let view = F32TensorView::from_gguf(&gguf, &bytes, "w.weight").expect("view");
        assert_eq!(
            f32_gemv_from_view(view, 1, 2, &[f32::NAN, 4.0], &mut out),
            Err(ModelError::NonFinite)
        );

        let (huge_gguf, huge_bytes) = single_f32_tensor_gguf("huge.weight", &[f32::MAX], &[1, 1]);
        let huge_view =
            F32TensorView::from_gguf(&huge_gguf, &huge_bytes, "huge.weight").expect("huge view");
        assert_eq!(
            f32_gemv_from_view(huge_view, 1, 1, &[2.0], &mut out),
            Err(ModelError::NonFinite)
        );
    }

    #[test]
    fn loads_tied_output_embedding_when_output_weight_is_missing() {
        let cfg = LlamaConfig {
            block_count: 1,
            embedding_length: 4,
            feed_forward_length: 6,
            head_count: 2,
            head_count_kv: 1,
            rms_epsilon: 1e-5,
            attention_scale: default_attention_scale(2),
            rope_freq_base: 10_000.0,
            rope_dimension_count: 2,
            rope_pairing: RopePairing::Adjacent,
            context_length: 4,
        };
        let (gguf, bytes) = synthetic_f32_gguf_without_output_weight(cfg);
        let model = F32Llama::from_gguf(&gguf, &bytes).expect("model");
        assert_eq!(model.output, model.token_embedding);
        let hash = model.logits_hash_for_tokens(&[0, 1]).expect("hash");
        assert_ne!(hash, [0u8; 32]);
    }

    #[test]
    fn loads_q8_and_q4_weight_matrices_from_gguf_parts() {
        let mut tensors = Vec::new();
        let mut bytes = Vec::new();
        push_q8_tensor(&mut tensors, &mut bytes, "q8.weight", 2, 32, 0x3c00, 2);
        push_q4_tensor(&mut tensors, &mut bytes, "q4.weight", 2, 32, 0x3c00, 0x99);
        let gguf = det_gguf::Gguf::from_parts(3, BTreeMap::new(), tensors, 0, bytes.len());
        let q8 = read_weight_matrix(&gguf, &bytes, "q8.weight", 2, 32).expect("q8");
        let q4 = read_weight_matrix(&gguf, &bytes, "q4.weight", 2, 32).expect("q4");

        let x = vec![1.0f32; 32];
        let mut y8 = [0.0f32; 2];
        let mut y4 = [0.0f32; 2];
        q8.gemv(&x, &mut y8).expect("q8 gemv");
        q4.gemv(&x, &mut y4).expect("q4 gemv");
        assert_eq!(q8.row_to_vec(0).expect("row").len(), 32);
        assert_eq!(q4.row_to_vec(0).expect("row").len(), 32);
        assert!(y8.iter().chain(y4.iter()).all(|v| v.is_finite()));
        assert_ne!(y8.map(f32::to_bits), [0u32; 2]);
        assert_ne!(y4.map(f32::to_bits), [0u32; 2]);
    }

    #[test]
    fn shared_q8a_path_matches_standalone_quantized_gemv() {
        let mut tensors = Vec::new();
        let mut bytes = Vec::new();
        push_q8_tensor(&mut tensors, &mut bytes, "q8.weight", 2, 32, 0x3c00, 2);
        push_q4_tensor(&mut tensors, &mut bytes, "q4.weight", 2, 32, 0x3c00, 0x99);
        let gguf = det_gguf::Gguf::from_parts(3, BTreeMap::new(), tensors, 0, bytes.len());
        let q8 = read_weight_matrix(&gguf, &bytes, "q8.weight", 2, 32).expect("q8");
        let q4 = read_weight_matrix(&gguf, &bytes, "q4.weight", 2, 32).expect("q4");
        let x: Vec<f32> = (0..32).map(|i| ((i as f32) - 16.0) / 8.0).collect();
        let shared = shared_q8a_if_needed(&x, [&q8, &q4])
            .expect("shared")
            .expect("q8a");

        let mut y8_standalone = [0.0f32; 2];
        let mut y8_shared = [0.0f32; 2];
        let mut y4_standalone = [0.0f32; 2];
        let mut y4_shared = [0.0f32; 2];
        q8.gemv(&x, &mut y8_standalone).expect("q8 standalone");
        q8.gemv_with_optional_q8a(&x, Some(&shared), &mut y8_shared)
            .expect("q8 shared");
        q4.gemv(&x, &mut y4_standalone).expect("q4 standalone");
        q4.gemv_with_optional_q8a(&x, Some(&shared), &mut y4_shared)
            .expect("q4 shared");

        assert_eq!(y8_shared.map(f32::to_bits), y8_standalone.map(f32::to_bits));
        assert_eq!(y4_shared.map(f32::to_bits), y4_standalone.map(f32::to_bits));

        let f32: WeightMatrix = patterned_matrix(2, 32, 0.01).into();
        assert!(shared_q8a_if_needed(&x, [&f32]).expect("f32").is_none());
        assert_eq!(
            shared_q8a_if_needed(&x, [&f32, &q8])
                .expect("mixed shared")
                .expect("mixed q8a"),
            quantize_q8a(&x).expect("standalone q8a")
        );
        assert_eq!(
            q8.gemv_with_optional_q8a(&x, None, &mut y8_shared),
            Err(ModelError::Shape)
        );
        assert_eq!(
            q8.gemv_with_optional_q8a(&x, Some(&[]), &mut y8_shared),
            Err(ModelError::Shape)
        );
    }

    #[cfg(all(feature = "parallel", not(target_family = "wasm")))]
    #[test]
    fn parallel_gemv_thread_counts_are_bit_invariant() {
        let rows = 17;
        let cols = 32;
        let x: Vec<f32> = (0..cols).map(|i| ((i as f32) - 15.0) / 9.0).collect();
        let f32 = patterned_matrix(rows, cols, 0.007);
        let mut expected_f32 = vec![0.0f32; rows];
        gemv_rows_sequential(rows, &mut expected_f32, |r| {
            f32.row(r).map(|row| dot_f32_ref(row, &x))
        })
        .expect("f32 reference");

        for threads in [1usize, 2, 7, 16] {
            let mut out = vec![0.0f32; rows];
            gemv_rows_parallel(
                rows,
                &mut out,
                |r| f32.row(r).map(|row| dot_f32_ref(row, &x)),
                threads,
            )
            .expect("parallel f32");
            assert_eq!(bits(&out), bits(&expected_f32));
        }

        let qx = quantize_q8a(&x).expect("q8a");
        let q8 = Q8Matrix {
            rows,
            cols,
            blocks_per_row: 1,
            blocks: (0..rows).map(q8_block_for_row).collect(),
        };
        let q4 = Q4Matrix {
            rows,
            cols,
            blocks_per_row: 1,
            blocks: (0..rows).map(q4_block_for_row).collect(),
        };
        let expected_q8: Vec<f32> = (0..rows)
            .map(|r| dot_q8_0_q8a(&q8.blocks[r..r + 1], &qx).expect("q8 dot"))
            .collect();
        let expected_q4: Vec<f32> = (0..rows)
            .map(|r| dot_q4_0_q8a(&q4.blocks[r..r + 1], &qx).expect("q4 dot"))
            .collect();

        for threads in [1usize, 2, 7, 16] {
            let mut out_q8 = vec![0.0f32; rows];
            gemv_rows_parallel(
                rows,
                &mut out_q8,
                |r| dot_q8_0_q8a(&q8.blocks[r..r + 1], &qx).map_err(map_quant_error),
                threads,
            )
            .expect("parallel q8");
            assert_eq!(bits(&out_q8), bits(&expected_q8));

            let mut out_q4 = vec![0.0f32; rows];
            gemv_rows_parallel(
                rows,
                &mut out_q4,
                |r| dot_q4_0_q8a(&q4.blocks[r..r + 1], &qx).map_err(map_quant_error),
                threads,
            )
            .expect("parallel q4");
            assert_eq!(bits(&out_q4), bits(&expected_q4));
        }
    }

    #[cfg(all(feature = "parallel", not(target_family = "wasm")))]
    fn bits(values: &[f32]) -> Vec<u32> {
        values.iter().map(|v| v.to_bits()).collect()
    }

    #[cfg(all(feature = "parallel", not(target_family = "wasm")))]
    fn q8_block_for_row(row: usize) -> Q8_0Block {
        let mut q = [0i8; BLOCK];
        for (i, v) in q.iter_mut().enumerate() {
            *v = (((row * 3 + i) % 11) as i8) - 5;
        }
        Q8_0Block { d: 0.125, q }
    }

    #[cfg(all(feature = "parallel", not(target_family = "wasm")))]
    fn q4_block_for_row(row: usize) -> Q4_0Block {
        let mut qs = [0u8; 16];
        for (i, byte) in qs.iter_mut().enumerate() {
            let lo = ((row + i * 2) % 16) as u8;
            let hi = ((row + i * 2 + 1) % 16) as u8;
            *byte = lo | (hi << 4);
        }
        Q4_0Block { d: 0.25, qs }
    }

    #[test]
    fn quantized_llama_forward_runs_with_q8_and_q4_weights() {
        let cfg = LlamaConfig {
            block_count: 1,
            embedding_length: 32,
            feed_forward_length: 32,
            head_count: 2,
            head_count_kv: 1,
            rms_epsilon: 1e-5,
            attention_scale: default_attention_scale(16),
            rope_freq_base: 10_000.0,
            rope_dimension_count: 16,
            rope_pairing: RopePairing::Adjacent,
            context_length: 4,
        };
        let (gguf, bytes) = synthetic_quant_gguf(cfg);
        let model = F32Llama::from_gguf(&gguf, &bytes).expect("quant model");
        let hash = model.logits_hash_for_tokens(&[0, 1]).expect("hash");
        assert_ne!(hash, [0u8; 32]);
    }

    #[test]
    fn quantized_forward_is_position_invariant_under_prefix_replay() {
        let cfg = LlamaConfig {
            block_count: 1,
            embedding_length: 32,
            feed_forward_length: 32,
            head_count: 2,
            head_count_kv: 1,
            rms_epsilon: 1e-5,
            attention_scale: default_attention_scale(16),
            rope_freq_base: 10_000.0,
            rope_dimension_count: 16,
            rope_pairing: RopePairing::Adjacent,
            context_length: 4,
        };
        let (gguf, bytes) = synthetic_quant_gguf(cfg);
        let model = F32Llama::from_gguf(&gguf, &bytes).expect("quant model");
        let tokens = [0usize, 1, 2, 0];
        let rope = RopeTables::llama(cfg, tokens.len()).expect("rope");

        let mut cache = KvCache::new(cfg).expect("cache");
        let mut logits = vec![0.0f32; model.output.rows()];
        let mut continuous = Vec::new();
        for (pos, &token) in tokens.iter().enumerate() {
            model
                .forward_one(token, pos, &rope, &mut cache, &mut logits)
                .expect("continuous forward");
            continuous.push(logits.iter().map(|x| x.to_bits()).collect::<Vec<_>>());
        }

        for end in 0..tokens.len() {
            let mut replay_cache = KvCache::new(cfg).expect("replay cache");
            let mut replay_logits = vec![0.0f32; model.output.rows()];
            for (pos, &token) in tokens[..=end].iter().enumerate() {
                model
                    .forward_one(token, pos, &rope, &mut replay_cache, &mut replay_logits)
                    .expect("replay forward");
            }
            let replay_bits = replay_logits
                .iter()
                .map(|x| x.to_bits())
                .collect::<Vec<_>>();
            assert_eq!(replay_bits, continuous[end]);
        }
    }

    #[test]
    fn testdata_logits_hash_is_invariant_to_chunks_and_threads() {
        let root = workspace_root();
        let tokens = read_tokens(&root.join("testdata/tiny.tokens.txt"));

        for (model_path, hash_path) in [
            ("testdata/tiny-f32.gguf", "testdata/tiny-f32.logits.sha256"),
            (
                "testdata/tiny-qmix.gguf",
                "testdata/tiny-qmix.logits.sha256",
            ),
        ] {
            let bytes = std::fs::read(root.join(model_path)).expect("model fixture");
            let gguf = det_gguf::parse(&bytes).expect("parse fixture");
            let model = F32Llama::from_gguf(&gguf, &bytes).expect("load fixture");
            let expected = std::fs::read_to_string(root.join(hash_path)).expect("golden hash");

            for threads in thread_counts_to_check() {
                set_thread_count(Some(threads)).expect("thread count");
                for chunk_size in [1, 2, 3, tokens.len()] {
                    let digest = model
                        .logits_hash_for_tokens_chunked(&tokens, chunk_size)
                        .expect("hash");
                    assert_eq!(
                        format!("{}\n", hex(&digest)),
                        expected,
                        "{model_path} threads={threads} chunk_size={chunk_size}"
                    );
                }
            }
        }
        set_thread_count(None).expect("reset thread count");
    }

    #[test]
    fn testdata_logits_are_position_invariant_under_replay() {
        let root = workspace_root();
        let tokens = read_tokens(&root.join("testdata/tiny.tokens.txt"));

        for model_path in ["testdata/tiny-f32.gguf", "testdata/tiny-qmix.gguf"] {
            let bytes = std::fs::read(root.join(model_path)).expect("model fixture");
            let gguf = det_gguf::parse(&bytes).expect("parse fixture");
            let model = F32Llama::from_gguf(&gguf, &bytes).expect("load fixture");
            let rope = RopeTables::llama(model.config, tokens.len()).expect("rope");

            let mut cache = KvCache::new(model.config).expect("cache");
            let mut logits = vec![0.0f32; model.output.rows()];
            let mut continuous = Vec::new();
            for (pos, &token) in tokens.iter().enumerate() {
                model
                    .forward_one(token, pos, &rope, &mut cache, &mut logits)
                    .expect("continuous forward");
                continuous.push(logits.iter().map(|x| x.to_bits()).collect::<Vec<_>>());
            }

            for end in 0..tokens.len() {
                let mut replay_cache = KvCache::new(model.config).expect("replay cache");
                let mut replay_logits = vec![0.0f32; model.output.rows()];
                for (pos, &token) in tokens[..=end].iter().enumerate() {
                    model
                        .forward_one(token, pos, &rope, &mut replay_cache, &mut replay_logits)
                        .expect("replay forward");
                }
                let replay_bits = replay_logits
                    .iter()
                    .map(|x| x.to_bits())
                    .collect::<Vec<_>>();
                assert_eq!(replay_bits, continuous[end], "{model_path} pos={end}");
            }
        }
    }

    fn workspace_root() -> PathBuf {
        Path::new(env!("CARGO_MANIFEST_DIR"))
            .parent()
            .expect("crates directory")
            .parent()
            .expect("workspace root")
            .to_path_buf()
    }

    fn read_tokens(path: &Path) -> Vec<usize> {
        std::fs::read_to_string(path)
            .expect("tokens fixture")
            .trim()
            .split(',')
            .map(|part| part.parse::<usize>().expect("token id"))
            .collect()
    }

    fn thread_counts_to_check() -> Vec<usize> {
        if cfg!(all(feature = "parallel", not(target_family = "wasm"))) {
            vec![1, 2, 7, 16]
        } else {
            vec![1]
        }
    }

    fn hex(bytes: &[u8]) -> String {
        const HEX: &[u8; 16] = b"0123456789abcdef";
        let mut out = String::with_capacity(bytes.len() * 2);
        for &byte in bytes {
            out.push(HEX[(byte >> 4) as usize] as char);
            out.push(HEX[(byte & 0x0f) as usize] as char);
        }
        out
    }

    fn synthetic_f32_gguf(config: LlamaConfig) -> (det_gguf::Gguf, Vec<u8>) {
        synthetic_f32_gguf_with_output(config, true)
    }

    fn synthetic_f32_gguf_without_output_weight(config: LlamaConfig) -> (det_gguf::Gguf, Vec<u8>) {
        synthetic_f32_gguf_with_output(config, false)
    }

    fn synthetic_f16_gguf(config: LlamaConfig) -> (det_gguf::Gguf, Vec<u8>) {
        let mut metadata = BTreeMap::new();
        metadata.insert(
            "general.architecture".to_owned(),
            det_gguf::MetadataValue::String("llama".to_owned()),
        );
        metadata.insert(
            "llama.block_count".to_owned(),
            det_gguf::MetadataValue::U32(config.block_count as u32),
        );
        metadata.insert(
            "llama.embedding_length".to_owned(),
            det_gguf::MetadataValue::U32(config.embedding_length as u32),
        );
        metadata.insert(
            "llama.feed_forward_length".to_owned(),
            det_gguf::MetadataValue::U32(config.feed_forward_length as u32),
        );
        metadata.insert(
            "llama.attention.head_count".to_owned(),
            det_gguf::MetadataValue::U32(config.head_count as u32),
        );
        metadata.insert(
            "llama.attention.head_count_kv".to_owned(),
            det_gguf::MetadataValue::U32(config.head_count_kv as u32),
        );
        metadata.insert(
            "llama.attention.layer_norm_rms_epsilon".to_owned(),
            det_gguf::MetadataValue::F32(config.rms_epsilon),
        );
        metadata.insert(
            "llama.rope.freq_base".to_owned(),
            det_gguf::MetadataValue::F32(config.rope_freq_base),
        );
        metadata.insert(
            "llama.rope.dimension_count".to_owned(),
            det_gguf::MetadataValue::U32(config.rope_dimension_count as u32),
        );
        metadata.insert(
            "llama.context_length".to_owned(),
            det_gguf::MetadataValue::U32(config.context_length as u32),
        );
        metadata.insert(
            "llama.vocab_size".to_owned(),
            det_gguf::MetadataValue::U32(3),
        );

        let mut tensors = Vec::new();
        let mut bytes = Vec::new();
        push_f16_tensor(&mut tensors, &mut bytes, "token_embd.weight", 3, 4);
        push_f16_vector(
            &mut tensors,
            &mut bytes,
            "blk.0.attn_norm.weight",
            4,
            0x3c00,
        );
        push_f16_tensor(&mut tensors, &mut bytes, "blk.0.attn_q.weight", 4, 4);
        push_f16_tensor(&mut tensors, &mut bytes, "blk.0.attn_k.weight", 2, 4);
        push_f16_tensor(&mut tensors, &mut bytes, "blk.0.attn_v.weight", 2, 4);
        push_f16_tensor(&mut tensors, &mut bytes, "blk.0.attn_output.weight", 4, 4);
        push_f16_vector(&mut tensors, &mut bytes, "blk.0.ffn_norm.weight", 4, 0x3c00);
        push_f16_tensor(&mut tensors, &mut bytes, "blk.0.ffn_gate.weight", 6, 4);
        push_f16_tensor(&mut tensors, &mut bytes, "blk.0.ffn_up.weight", 6, 4);
        push_f16_tensor(&mut tensors, &mut bytes, "blk.0.ffn_down.weight", 4, 6);
        push_f16_vector(&mut tensors, &mut bytes, "output_norm.weight", 4, 0x3c00);
        push_f16_tensor(&mut tensors, &mut bytes, "output.weight", 3, 4);

        let gguf = det_gguf::Gguf::from_parts(3, metadata, tensors, 0, bytes.len());
        (gguf, bytes)
    }

    fn synthetic_f32_gguf_with_output(
        config: LlamaConfig,
        include_output_weight: bool,
    ) -> (det_gguf::Gguf, Vec<u8>) {
        synthetic_f32_gguf_with_arch(config, include_output_weight, "llama")
    }

    fn synthetic_f32_gguf_with_arch(
        config: LlamaConfig,
        include_output_weight: bool,
        arch: &str,
    ) -> (det_gguf::Gguf, Vec<u8>) {
        let mut metadata = BTreeMap::new();
        metadata.insert(
            "general.architecture".to_owned(),
            det_gguf::MetadataValue::String(arch.to_owned()),
        );
        let key = |suffix: &str| format!("{arch}.{suffix}");
        metadata.insert(
            key("block_count"),
            det_gguf::MetadataValue::U32(config.block_count as u32),
        );
        metadata.insert(
            key("embedding_length"),
            det_gguf::MetadataValue::U32(config.embedding_length as u32),
        );
        metadata.insert(
            key("feed_forward_length"),
            det_gguf::MetadataValue::U32(config.feed_forward_length as u32),
        );
        metadata.insert(
            key("attention.head_count"),
            det_gguf::MetadataValue::U32(config.head_count as u32),
        );
        metadata.insert(
            key("attention.head_count_kv"),
            det_gguf::MetadataValue::U32(config.head_count_kv as u32),
        );
        metadata.insert(
            key("attention.layer_norm_rms_epsilon"),
            det_gguf::MetadataValue::F32(config.rms_epsilon),
        );
        metadata.insert(
            key("rope.freq_base"),
            det_gguf::MetadataValue::F32(config.rope_freq_base),
        );
        metadata.insert(
            key("rope.dimension_count"),
            det_gguf::MetadataValue::U32(config.rope_dimension_count as u32),
        );
        metadata.insert(
            key("context_length"),
            det_gguf::MetadataValue::U32(config.context_length as u32),
        );
        metadata.insert(key("vocab_size"), det_gguf::MetadataValue::U32(3));

        let mut tensors = Vec::new();
        let mut bytes = Vec::new();
        push_tensor(&mut tensors, &mut bytes, "token_embd.weight", 3, 4, 0.01);
        push_vector(&mut tensors, &mut bytes, "blk.0.attn_norm.weight", 4, 1.0);
        push_tensor(&mut tensors, &mut bytes, "blk.0.attn_q.weight", 4, 4, 0.02);
        push_tensor(
            &mut tensors,
            &mut bytes,
            "blk.0.attn_k.weight",
            2,
            4,
            -0.015,
        );
        push_tensor(&mut tensors, &mut bytes, "blk.0.attn_v.weight", 2, 4, 0.025);
        push_tensor(
            &mut tensors,
            &mut bytes,
            "blk.0.attn_output.weight",
            4,
            4,
            -0.02,
        );
        push_vector(&mut tensors, &mut bytes, "blk.0.ffn_norm.weight", 4, 1.0);
        push_tensor(
            &mut tensors,
            &mut bytes,
            "blk.0.ffn_gate.weight",
            6,
            4,
            0.03,
        );
        push_tensor(
            &mut tensors,
            &mut bytes,
            "blk.0.ffn_up.weight",
            6,
            4,
            -0.025,
        );
        push_tensor(
            &mut tensors,
            &mut bytes,
            "blk.0.ffn_down.weight",
            4,
            6,
            0.018,
        );
        push_vector(&mut tensors, &mut bytes, "output_norm.weight", 4, 1.0);
        if include_output_weight {
            push_tensor(&mut tensors, &mut bytes, "output.weight", 3, 4, 0.022);
        }

        let gguf = det_gguf::Gguf::from_parts(3, metadata, tensors, 0, bytes.len());
        (gguf, bytes)
    }

    fn synthetic_quant_gguf(config: LlamaConfig) -> (det_gguf::Gguf, Vec<u8>) {
        let mut metadata = BTreeMap::new();
        metadata.insert(
            "general.architecture".to_owned(),
            det_gguf::MetadataValue::String("llama".to_owned()),
        );
        metadata.insert(
            "llama.block_count".to_owned(),
            det_gguf::MetadataValue::U32(config.block_count as u32),
        );
        metadata.insert(
            "llama.embedding_length".to_owned(),
            det_gguf::MetadataValue::U32(config.embedding_length as u32),
        );
        metadata.insert(
            "llama.feed_forward_length".to_owned(),
            det_gguf::MetadataValue::U32(config.feed_forward_length as u32),
        );
        metadata.insert(
            "llama.attention.head_count".to_owned(),
            det_gguf::MetadataValue::U32(config.head_count as u32),
        );
        metadata.insert(
            "llama.attention.head_count_kv".to_owned(),
            det_gguf::MetadataValue::U32(config.head_count_kv as u32),
        );
        metadata.insert(
            "llama.attention.layer_norm_rms_epsilon".to_owned(),
            det_gguf::MetadataValue::F32(config.rms_epsilon),
        );
        metadata.insert(
            "llama.rope.dimension_count".to_owned(),
            det_gguf::MetadataValue::U32(config.rope_dimension_count as u32),
        );
        metadata.insert(
            "llama.context_length".to_owned(),
            det_gguf::MetadataValue::U32(config.context_length as u32),
        );
        metadata.insert(
            "llama.vocab_size".to_owned(),
            det_gguf::MetadataValue::U32(3),
        );

        let mut tensors = Vec::new();
        let mut bytes = Vec::new();
        push_q8_tensor(
            &mut tensors,
            &mut bytes,
            "token_embd.weight",
            3,
            32,
            0x3c00,
            2,
        );
        push_vector(&mut tensors, &mut bytes, "blk.0.attn_norm.weight", 32, 1.0);
        push_q8_tensor(
            &mut tensors,
            &mut bytes,
            "blk.0.attn_q.weight",
            32,
            32,
            0x3c00,
            1,
        );
        push_q4_tensor(
            &mut tensors,
            &mut bytes,
            "blk.0.attn_k.weight",
            16,
            32,
            0x3c00,
            0x99,
        );
        push_q8_tensor(
            &mut tensors,
            &mut bytes,
            "blk.0.attn_v.weight",
            16,
            32,
            0x3c00,
            -1,
        );
        push_q4_tensor(
            &mut tensors,
            &mut bytes,
            "blk.0.attn_output.weight",
            32,
            32,
            0x3c00,
            0x99,
        );
        push_vector(&mut tensors, &mut bytes, "blk.0.ffn_norm.weight", 32, 1.0);
        push_q8_tensor(
            &mut tensors,
            &mut bytes,
            "blk.0.ffn_gate.weight",
            32,
            32,
            0x3c00,
            2,
        );
        push_q4_tensor(
            &mut tensors,
            &mut bytes,
            "blk.0.ffn_up.weight",
            32,
            32,
            0x3c00,
            0x99,
        );
        push_q8_tensor(
            &mut tensors,
            &mut bytes,
            "blk.0.ffn_down.weight",
            32,
            32,
            0x3c00,
            -2,
        );
        push_vector(&mut tensors, &mut bytes, "output_norm.weight", 32, 1.0);

        let gguf = det_gguf::Gguf::from_parts(3, metadata, tensors, 0, bytes.len());
        (gguf, bytes)
    }

    fn push_tensor(
        tensors: &mut Vec<det_gguf::TensorInfo>,
        bytes: &mut Vec<u8>,
        name: &str,
        rows: usize,
        cols: usize,
        scale: f32,
    ) {
        let offset = bytes.len() as u64;
        tensors.push(det_gguf::TensorInfo {
            name: name.to_owned(),
            dimensions: vec![cols as u64, rows as u64],
            ty: det_gguf::GgmlType::F32,
            offset,
        });
        for i in 0..rows * cols {
            let value = (((i % 11) as f32) - 5.0) * scale;
            bytes.extend_from_slice(&value.to_le_bytes());
        }
    }

    fn push_vector(
        tensors: &mut Vec<det_gguf::TensorInfo>,
        bytes: &mut Vec<u8>,
        name: &str,
        len: usize,
        value: f32,
    ) {
        let offset = bytes.len() as u64;
        tensors.push(det_gguf::TensorInfo {
            name: name.to_owned(),
            dimensions: vec![len as u64],
            ty: det_gguf::GgmlType::F32,
            offset,
        });
        for _ in 0..len {
            bytes.extend_from_slice(&value.to_le_bytes());
        }
    }

    fn single_f32_tensor_gguf(
        name: &str,
        values: &[f32],
        dimensions: &[u64],
    ) -> (det_gguf::Gguf, Vec<u8>) {
        let mut bytes = Vec::with_capacity(values.len() * 4);
        for &value in values {
            bytes.extend_from_slice(&value.to_le_bytes());
        }
        let tensors = vec![det_gguf::TensorInfo {
            name: name.to_owned(),
            dimensions: dimensions.to_vec(),
            ty: det_gguf::GgmlType::F32,
            offset: 0,
        }];
        let gguf = det_gguf::Gguf::from_parts(3, BTreeMap::new(), tensors, 0, bytes.len());
        (gguf, bytes)
    }

    fn push_f16_tensor(
        tensors: &mut Vec<det_gguf::TensorInfo>,
        bytes: &mut Vec<u8>,
        name: &str,
        rows: usize,
        cols: usize,
    ) {
        const VALUES: [u16; 8] = [
            0x0000, 0x2c00, 0xac00, 0x3000, 0xb000, 0x3400, 0xb400, 0x3800,
        ];
        let offset = bytes.len() as u64;
        tensors.push(det_gguf::TensorInfo {
            name: name.to_owned(),
            dimensions: vec![cols as u64, rows as u64],
            ty: det_gguf::GgmlType::F16,
            offset,
        });
        for i in 0..rows * cols {
            bytes.extend_from_slice(&VALUES[i % VALUES.len()].to_le_bytes());
        }
    }

    fn push_f16_vector(
        tensors: &mut Vec<det_gguf::TensorInfo>,
        bytes: &mut Vec<u8>,
        name: &str,
        len: usize,
        value: u16,
    ) {
        let offset = bytes.len() as u64;
        tensors.push(det_gguf::TensorInfo {
            name: name.to_owned(),
            dimensions: vec![len as u64],
            ty: det_gguf::GgmlType::F16,
            offset,
        });
        for _ in 0..len {
            bytes.extend_from_slice(&value.to_le_bytes());
        }
    }

    fn push_q8_tensor(
        tensors: &mut Vec<det_gguf::TensorInfo>,
        bytes: &mut Vec<u8>,
        name: &str,
        rows: usize,
        cols: usize,
        scale_f16: u16,
        q: i8,
    ) {
        let offset = bytes.len() as u64;
        tensors.push(det_gguf::TensorInfo {
            name: name.to_owned(),
            dimensions: vec![cols as u64, rows as u64],
            ty: det_gguf::GgmlType::Q8_0,
            offset,
        });
        for _ in 0..rows * (cols / BLOCK) {
            bytes.extend_from_slice(&scale_f16.to_le_bytes());
            for _ in 0..BLOCK {
                bytes.push(q as u8);
            }
        }
    }

    fn push_q4_tensor(
        tensors: &mut Vec<det_gguf::TensorInfo>,
        bytes: &mut Vec<u8>,
        name: &str,
        rows: usize,
        cols: usize,
        scale_f16: u16,
        packed: u8,
    ) {
        let offset = bytes.len() as u64;
        tensors.push(det_gguf::TensorInfo {
            name: name.to_owned(),
            dimensions: vec![cols as u64, rows as u64],
            ty: det_gguf::GgmlType::Q4_0,
            offset,
        });
        for _ in 0..rows * (cols / BLOCK) {
            bytes.extend_from_slice(&scale_f16.to_le_bytes());
            for _ in 0..16 {
                bytes.push(packed);
            }
        }
    }

    #[test]
    fn extracts_llama_config_from_gguf_metadata() {
        let mut metadata = BTreeMap::new();
        metadata.insert(
            "general.architecture".to_owned(),
            det_gguf::MetadataValue::String("llama".to_owned()),
        );
        metadata.insert(
            "llama.block_count".to_owned(),
            det_gguf::MetadataValue::U32(2),
        );
        metadata.insert(
            "llama.embedding_length".to_owned(),
            det_gguf::MetadataValue::U32(8),
        );
        metadata.insert(
            "llama.feed_forward_length".to_owned(),
            det_gguf::MetadataValue::U32(16),
        );
        metadata.insert(
            "llama.attention.head_count".to_owned(),
            det_gguf::MetadataValue::U32(2),
        );
        metadata.insert(
            "llama.attention.head_count_kv".to_owned(),
            det_gguf::MetadataValue::U32(1),
        );
        metadata.insert(
            "llama.attention.layer_norm_rms_epsilon".to_owned(),
            det_gguf::MetadataValue::F32(1e-5),
        );
        metadata.insert(
            "llama.context_length".to_owned(),
            det_gguf::MetadataValue::U32(32),
        );
        let gguf = det_gguf::Gguf::from_parts(3, metadata, Vec::new(), 0, 0);
        let cfg = LlamaConfig::from_gguf(&gguf).expect("config");
        assert_eq!(cfg.head_dim().expect("head dim"), 4);
        assert_eq!(cfg.attention_scale.to_bits(), 0.5f32.to_bits());
        assert_eq!(cfg.rope_freq_base.to_bits(), 10_000.0f32.to_bits());
        cfg.validate().expect("valid");
    }

    #[test]
    fn loads_attention_scale_metadata() {
        let mut metadata = minimal_llama_metadata();
        metadata.insert(
            "llama.attention.scale".to_owned(),
            det_gguf::MetadataValue::F32(0.125),
        );
        let gguf = det_gguf::Gguf::from_parts(3, metadata, Vec::new(), 0, 0);
        let cfg = LlamaConfig::from_gguf(&gguf).expect("config");
        assert_eq!(cfg.attention_scale.to_bits(), 0.125f32.to_bits());
        cfg.validate().expect("valid");
    }

    #[test]
    fn rejects_unsupported_attention_head_length_metadata() {
        let mut metadata = minimal_llama_metadata();
        metadata.insert(
            "llama.attention.key_length".to_owned(),
            det_gguf::MetadataValue::U32(4),
        );
        metadata.insert(
            "llama.attention.value_length".to_owned(),
            det_gguf::MetadataValue::U32(4),
        );
        let gguf = det_gguf::Gguf::from_parts(3, metadata, Vec::new(), 0, 0);
        LlamaConfig::from_gguf(&gguf)
            .expect("matching explicit key/value lengths")
            .validate()
            .expect("valid config");

        let mut metadata = minimal_llama_metadata();
        metadata.insert(
            "llama.attention.key_length".to_owned(),
            det_gguf::MetadataValue::U32(8),
        );
        let gguf = det_gguf::Gguf::from_parts(3, metadata, Vec::new(), 0, 0);
        assert_eq!(
            LlamaConfig::from_gguf(&gguf),
            Err(ModelError::UnsupportedAttentionHeadLength)
        );

        let mut metadata = minimal_llama_metadata();
        metadata.insert(
            "llama.attention.value_length".to_owned(),
            det_gguf::MetadataValue::U32(8),
        );
        let gguf = det_gguf::Gguf::from_parts(3, metadata, Vec::new(), 0, 0);
        assert_eq!(
            LlamaConfig::from_gguf(&gguf),
            Err(ModelError::UnsupportedAttentionHeadLength)
        );
    }

    #[test]
    fn rejects_unsupported_decoder_feature_metadata() {
        let mut metadata = minimal_llama_metadata();
        metadata.insert(
            "llama.use_parallel_residual".to_owned(),
            det_gguf::MetadataValue::Bool(false),
        );
        metadata.insert(
            "llama.attention.causal".to_owned(),
            det_gguf::MetadataValue::Bool(true),
        );
        metadata.insert(
            "llama.embedding_length_out".to_owned(),
            det_gguf::MetadataValue::U32(8),
        );
        metadata.insert(
            "llama.attention.sliding_window".to_owned(),
            det_gguf::MetadataValue::U32(0),
        );
        for &key in unsupported_f32_feature_keys() {
            metadata.insert(key.to_owned(), det_gguf::MetadataValue::F32(-0.0));
        }
        let gguf = det_gguf::Gguf::from_parts(3, metadata, Vec::new(), 0, 0);
        LlamaConfig::from_gguf(&gguf)
            .expect("neutral optional feature metadata")
            .validate()
            .expect("valid config");

        let mut metadata = minimal_llama_metadata();
        metadata.insert(
            "llama.use_parallel_residual".to_owned(),
            det_gguf::MetadataValue::Bool(true),
        );
        let gguf = det_gguf::Gguf::from_parts(3, metadata, Vec::new(), 0, 0);
        assert_eq!(
            LlamaConfig::from_gguf(&gguf),
            Err(ModelError::UnsupportedModelFeature)
        );

        let mut metadata = minimal_llama_metadata();
        metadata.insert(
            "llama.attention.causal".to_owned(),
            det_gguf::MetadataValue::Bool(false),
        );
        let gguf = det_gguf::Gguf::from_parts(3, metadata, Vec::new(), 0, 0);
        assert_eq!(
            LlamaConfig::from_gguf(&gguf),
            Err(ModelError::UnsupportedModelFeature)
        );

        let mut metadata = minimal_llama_metadata();
        metadata.insert(
            "llama.embedding_length_out".to_owned(),
            det_gguf::MetadataValue::U32(16),
        );
        let gguf = det_gguf::Gguf::from_parts(3, metadata, Vec::new(), 0, 0);
        assert_eq!(
            LlamaConfig::from_gguf(&gguf),
            Err(ModelError::UnsupportedModelFeature)
        );

        let mut metadata = minimal_llama_metadata();
        metadata.insert(
            "llama.attention.sliding_window".to_owned(),
            det_gguf::MetadataValue::U32(128),
        );
        let gguf = det_gguf::Gguf::from_parts(3, metadata, Vec::new(), 0, 0);
        assert_eq!(
            LlamaConfig::from_gguf(&gguf),
            Err(ModelError::UnsupportedModelFeature)
        );

        for &key in unsupported_f32_feature_keys() {
            let mut metadata = minimal_llama_metadata();
            metadata.insert(key.to_owned(), det_gguf::MetadataValue::F32(1.0));
            let gguf = det_gguf::Gguf::from_parts(3, metadata, Vec::new(), 0, 0);
            assert_eq!(
                LlamaConfig::from_gguf(&gguf),
                Err(ModelError::UnsupportedModelFeature),
                "{key}"
            );
        }

        let mut metadata = minimal_llama_metadata();
        metadata.insert(
            "llama.attention.max_alibi_bias".to_owned(),
            det_gguf::MetadataValue::F64(-0.0),
        );
        LlamaConfig::from_gguf(&det_gguf::Gguf::from_parts(3, metadata, Vec::new(), 0, 0))
            .expect("negative zero f64 remains neutral")
            .validate()
            .expect("valid config");

        let mut metadata = minimal_llama_metadata();
        metadata.insert(
            "llama.attention.max_alibi_bias".to_owned(),
            det_gguf::MetadataValue::F64(f64::from_bits(1)),
        );
        let gguf = det_gguf::Gguf::from_parts(3, metadata, Vec::new(), 0, 0);
        assert_eq!(
            LlamaConfig::from_gguf(&gguf),
            Err(ModelError::UnsupportedModelFeature)
        );
    }

    fn unsupported_f32_feature_keys() -> &'static [&'static str] {
        &[
            "llama.attention.max_alibi_bias",
            "llama.attention.clamp_kqv",
            "llama.attention.value_scale",
            "llama.attn_logit_softcapping",
            "llama.final_logit_softcapping",
            "llama.logit_scale",
            "llama.embedding_scale",
            "llama.residual_scale",
        ]
    }

    #[test]
    fn rejects_unsupported_rope_scaling_metadata() {
        let mut metadata = minimal_llama_metadata();
        metadata.insert(
            "llama.rope.scaling.type".to_owned(),
            det_gguf::MetadataValue::String("linear".to_owned()),
        );
        let gguf = det_gguf::Gguf::from_parts(3, metadata, Vec::new(), 0, 0);
        assert_eq!(
            LlamaConfig::from_gguf(&gguf),
            Err(ModelError::UnsupportedRopeScaling)
        );

        let mut metadata = minimal_llama_metadata();
        metadata.insert(
            "llama.rope.scaling.type".to_owned(),
            det_gguf::MetadataValue::String("none".to_owned()),
        );
        let gguf = det_gguf::Gguf::from_parts(3, metadata, Vec::new(), 0, 0);
        LlamaConfig::from_gguf(&gguf)
            .expect("none scaling is equivalent to ordinary RoPE")
            .validate()
            .expect("valid config");
    }

    fn minimal_llama_metadata() -> BTreeMap<String, det_gguf::MetadataValue> {
        let mut metadata = BTreeMap::new();
        metadata.insert(
            "general.architecture".to_owned(),
            det_gguf::MetadataValue::String("llama".to_owned()),
        );
        metadata.insert(
            "llama.block_count".to_owned(),
            det_gguf::MetadataValue::U32(2),
        );
        metadata.insert(
            "llama.embedding_length".to_owned(),
            det_gguf::MetadataValue::U32(8),
        );
        metadata.insert(
            "llama.feed_forward_length".to_owned(),
            det_gguf::MetadataValue::U32(16),
        );
        metadata.insert(
            "llama.attention.head_count".to_owned(),
            det_gguf::MetadataValue::U32(2),
        );
        metadata.insert(
            "llama.attention.head_count_kv".to_owned(),
            det_gguf::MetadataValue::U32(1),
        );
        metadata.insert(
            "llama.attention.layer_norm_rms_epsilon".to_owned(),
            det_gguf::MetadataValue::F32(1e-5),
        );
        metadata.insert(
            "llama.context_length".to_owned(),
            det_gguf::MetadataValue::U32(32),
        );
        metadata
    }

    #[test]
    fn rejects_config_values_that_can_generate_nan() {
        let valid = LlamaConfig {
            block_count: 1,
            embedding_length: 8,
            feed_forward_length: 16,
            head_count: 2,
            head_count_kv: 1,
            rms_epsilon: 1e-5,
            attention_scale: default_attention_scale(4),
            rope_freq_base: 10_000.0,
            rope_dimension_count: 4,
            rope_pairing: RopePairing::Adjacent,
            context_length: 32,
        };

        let mut cfg = valid;
        cfg.rms_epsilon = 0.0;
        assert_eq!(cfg.validate(), Err(ModelError::Shape));

        cfg = valid;
        cfg.rms_epsilon = -1.0;
        assert_eq!(cfg.validate(), Err(ModelError::Shape));

        cfg = valid;
        cfg.rope_freq_base = 0.0;
        assert_eq!(cfg.validate(), Err(ModelError::Shape));
        assert_eq!(RopeTables::llama(cfg, 1), Err(ModelError::Shape));

        cfg = valid;
        cfg.rope_freq_base = -10_000.0;
        assert_eq!(cfg.validate(), Err(ModelError::Shape));
        assert_eq!(RopeTables::llama(cfg, 1), Err(ModelError::Shape));

        cfg = valid;
        cfg.attention_scale = 0.0;
        assert_eq!(cfg.validate(), Err(ModelError::Shape));

        cfg = valid;
        cfg.attention_scale = f32::NAN;
        assert_eq!(cfg.validate(), Err(ModelError::Shape));

        cfg = valid;
        cfg.rope_dimension_count = 3;
        assert_eq!(cfg.validate(), Err(ModelError::Shape));
        assert_eq!(RopeTables::llama(cfg, 1), Err(ModelError::Shape));

        cfg = valid;
        cfg.rope_dimension_count = 6;
        assert_eq!(cfg.validate(), Err(ModelError::Shape));
        assert_eq!(RopeTables::llama(cfg, 1), Err(ModelError::Shape));
    }

    #[test]
    fn rejects_invalid_numeric_metadata_before_returning_config() {
        for (key, value) in [
            ("llama.attention.layer_norm_rms_epsilon", 0.0),
            ("llama.attention.layer_norm_rms_epsilon", -1.0),
            ("llama.attention.scale", 0.0),
            ("llama.attention.scale", f32::NAN),
            ("llama.rope.freq_base", 0.0),
            ("llama.rope.freq_base", f32::INFINITY),
        ] {
            let mut metadata = minimal_llama_metadata();
            metadata.insert(key.to_owned(), det_gguf::MetadataValue::F32(value));
            let gguf = det_gguf::Gguf::from_parts(3, metadata, Vec::new(), 0, 0);
            assert_eq!(
                LlamaConfig::from_gguf(&gguf),
                Err(ModelError::Shape),
                "{key}"
            );
        }

        let mut metadata = minimal_llama_metadata();
        metadata.insert(
            "llama.rope.dimension_count".to_owned(),
            det_gguf::MetadataValue::U32(3),
        );
        let gguf = det_gguf::Gguf::from_parts(3, metadata, Vec::new(), 0, 0);
        assert_eq!(LlamaConfig::from_gguf(&gguf), Err(ModelError::Shape));
    }

    #[test]
    fn size_calculations_reject_overflow() {
        let valid = LlamaConfig {
            block_count: 1,
            embedding_length: 8,
            feed_forward_length: 16,
            head_count: 2,
            head_count_kv: 1,
            rms_epsilon: 1e-5,
            attention_scale: default_attention_scale(4),
            rope_freq_base: 10_000.0,
            rope_dimension_count: 4,
            rope_pairing: RopePairing::Adjacent,
            context_length: 32,
        };

        assert_eq!(RopeTables::identity(usize::MAX, 4), Err(ModelError::Shape));
        assert_eq!(RopeTables::llama(valid, usize::MAX), Err(ModelError::Shape));
        let mut huge_cache = valid;
        huge_cache.context_length = usize::MAX;
        assert_eq!(KvCache::new(huge_cache), Err(ModelError::Shape));
        assert_eq!(logits_byte_len(3, 4), Ok(48));
        assert_eq!(logits_byte_len(usize::MAX, 2), Err(ModelError::Shape));
        assert_eq!(
            logits_byte_len(usize::MAX / core::mem::size_of::<f32>() + 1, 1),
            Err(ModelError::Shape)
        );
        assert_eq!(
            F32Matrix::new(usize::MAX, 2, Vec::new()),
            Err(ModelError::Shape)
        );
    }
}
