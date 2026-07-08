use std::collections::BTreeMap;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum GgufError {
    BadMagic,
    UnsupportedVersion(u32),
    Truncated,
    UnsupportedValueType(u32),
    InvalidUtf8,
    InvalidTensorOffset,
    InvalidTensorShape,
    TensorNotFound,
    MetadataNotFound,
    DuplicateMetadataKey,
    DuplicateTensorName,
    InvalidBool,
    MetadataTypeMismatch,
    UnsupportedTensorType(u32),
}

#[derive(Debug, Clone, PartialEq)]
pub enum MetadataValue {
    U8(u8),
    I8(i8),
    U16(u16),
    I16(i16),
    U32(u32),
    I32(i32),
    U64(u64),
    I64(i64),
    F32(f32),
    F64(f64),
    Bool(bool),
    String(String),
    ArrayU8(Vec<u8>),
    ArrayI8(Vec<i8>),
    ArrayU16(Vec<u16>),
    ArrayI16(Vec<i16>),
    ArrayU32(Vec<u32>),
    ArrayI32(Vec<i32>),
    ArrayF32(Vec<f32>),
    ArrayBool(Vec<bool>),
    ArrayString(Vec<String>),
    ArrayU64(Vec<u64>),
    ArrayI64(Vec<i64>),
    ArrayF64(Vec<f64>),
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct TensorInfo {
    pub name: String,
    pub dimensions: Vec<u64>,
    pub ty: GgmlType,
    pub offset: u64,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum GgmlType {
    F32,
    F16,
    Q4_0,
    Q8_0,
    Other(u32),
}

impl GgmlType {
    pub fn from_u32(raw: u32) -> Self {
        match raw {
            0 => Self::F32,
            1 => Self::F16,
            2 => Self::Q4_0,
            8 => Self::Q8_0,
            other => Self::Other(other),
        }
    }

    pub fn raw(self) -> u32 {
        match self {
            Self::F32 => 0,
            Self::F16 => 1,
            Self::Q4_0 => 2,
            Self::Q8_0 => 8,
            Self::Other(raw) => raw,
        }
    }

    pub fn type_size(self) -> Option<u64> {
        match self {
            Self::F32 => Some(4),
            Self::F16 => Some(2),
            Self::Q4_0 => Some(18),
            Self::Q8_0 => Some(34),
            Self::Other(_) => None,
        }
    }

    pub fn block_size(self) -> Option<u64> {
        match self {
            Self::F32 | Self::F16 => Some(1),
            Self::Q4_0 | Self::Q8_0 => Some(32),
            Self::Other(_) => None,
        }
    }
}

#[derive(Debug, Clone)]
pub struct Gguf {
    pub version: u32,
    pub metadata: BTreeMap<String, MetadataValue>,
    pub tensors: Vec<TensorInfo>,
    pub data_offset: usize,
    file_len: usize,
}

pub fn parse(bytes: &[u8]) -> Result<Gguf, GgufError> {
    let mut r = Reader { bytes, pos: 0 };
    if r.read_exact(4)? != b"GGUF" {
        return Err(GgufError::BadMagic);
    }
    let version = r.u32()?;
    if version != 2 && version != 3 {
        return Err(GgufError::UnsupportedVersion(version));
    }
    let tensor_count = r.u64()?;
    let metadata_count = r.u64()?;

    let mut metadata = BTreeMap::new();
    for _ in 0..metadata_count {
        let key = r.string()?;
        let value = read_value(&mut r)?;
        if metadata.insert(key, value).is_some() {
            return Err(GgufError::DuplicateMetadataKey);
        }
    }

    r.ensure_count(tensor_count, 24)?;
    let mut tensors = Vec::with_capacity(usize_from_u64(tensor_count)?);
    for _ in 0..tensor_count {
        let name = r.string()?;
        let ndim = r.u32()?;
        let mut dimensions = Vec::with_capacity(usize_from_u32(ndim)?);
        for _ in 0..ndim {
            dimensions.push(r.u64()?);
        }
        let ty = GgmlType::from_u32(r.u32()?);
        let offset = r.u64()?;
        if tensors
            .iter()
            .any(|tensor: &TensorInfo| tensor.name == name)
        {
            return Err(GgufError::DuplicateTensorName);
        }
        tensors.push(TensorInfo {
            name,
            dimensions,
            ty,
            offset,
        });
    }

    let data_offset = align_up(r.pos, 32).ok_or(GgufError::Truncated)?;
    if data_offset > bytes.len() {
        return Err(GgufError::Truncated);
    }
    validate_tensor_data_layout(data_offset, bytes.len(), &tensors)?;
    Ok(Gguf {
        version,
        metadata,
        tensors,
        data_offset,
        file_len: bytes.len(),
    })
}

fn validate_tensor_data_layout(
    data_offset: usize,
    file_len: usize,
    tensors: &[TensorInfo],
) -> Result<(), GgufError> {
    let mut ranges = Vec::with_capacity(tensors.len());
    for tensor in tensors {
        let len = tensor.encoded_len()?;
        let start = (data_offset as u64)
            .checked_add(tensor.offset)
            .ok_or(GgufError::InvalidTensorOffset)?;
        let end = start
            .checked_add(len)
            .ok_or(GgufError::InvalidTensorOffset)?;
        let start = usize::try_from(start).map_err(|_| GgufError::InvalidTensorOffset)?;
        let end = usize::try_from(end).map_err(|_| GgufError::InvalidTensorOffset)?;
        if end > file_len {
            return Err(GgufError::Truncated);
        }
        ranges.push(start..end);
    }

    ranges.sort_by_key(|range| range.start);
    for pair in ranges.windows(2) {
        if pair[0].end > pair[1].start {
            return Err(GgufError::InvalidTensorOffset);
        }
    }
    Ok(())
}

impl Gguf {
    pub fn from_parts(
        version: u32,
        metadata: BTreeMap<String, MetadataValue>,
        tensors: Vec<TensorInfo>,
        data_offset: usize,
        file_len: usize,
    ) -> Self {
        Self {
            version,
            metadata,
            tensors,
            data_offset,
            file_len,
        }
    }

    pub fn metadata_value(&self, key: &str) -> Result<&MetadataValue, GgufError> {
        self.metadata.get(key).ok_or(GgufError::MetadataNotFound)
    }

    pub fn metadata_str(&self, key: &str) -> Result<&str, GgufError> {
        match self.metadata_value(key)? {
            MetadataValue::String(s) => Ok(s),
            _ => Err(GgufError::MetadataTypeMismatch),
        }
    }

    pub fn metadata_u32(&self, key: &str) -> Result<u32, GgufError> {
        match self.metadata_value(key)? {
            MetadataValue::U32(v) => Ok(*v),
            MetadataValue::U64(v) => u32::try_from(*v).map_err(|_| GgufError::MetadataTypeMismatch),
            _ => Err(GgufError::MetadataTypeMismatch),
        }
    }

    pub fn metadata_f32(&self, key: &str) -> Result<f32, GgufError> {
        match self.metadata_value(key)? {
            MetadataValue::F32(v) => Ok(*v),
            MetadataValue::F64(v) => Ok(*v as f32),
            _ => Err(GgufError::MetadataTypeMismatch),
        }
    }

    pub fn tensor(&self, name: &str) -> Result<&TensorInfo, GgufError> {
        let mut matches = self.tensors.iter().filter(|t| t.name == name);
        let tensor = matches.next().ok_or(GgufError::TensorNotFound)?;
        if matches.next().is_some() {
            return Err(GgufError::DuplicateTensorName);
        }
        Ok(tensor)
    }

    pub fn tensor_data_range(
        &self,
        tensor: &TensorInfo,
    ) -> Result<core::ops::Range<usize>, GgufError> {
        let len = tensor.encoded_len()?;
        let start = (self.data_offset as u64)
            .checked_add(tensor.offset)
            .ok_or(GgufError::InvalidTensorOffset)?;
        let end = start
            .checked_add(len)
            .ok_or(GgufError::InvalidTensorOffset)?;
        let start = usize::try_from(start).map_err(|_| GgufError::InvalidTensorOffset)?;
        let end = usize::try_from(end).map_err(|_| GgufError::InvalidTensorOffset)?;
        if end > self.file_len {
            return Err(GgufError::Truncated);
        }
        Ok(start..end)
    }

    pub fn tensor_data<'a>(&self, bytes: &'a [u8], name: &str) -> Result<&'a [u8], GgufError> {
        let tensor = self.tensor(name)?;
        let range = self.tensor_data_range(tensor)?;
        bytes.get(range).ok_or(GgufError::Truncated)
    }
}

impl TensorInfo {
    pub fn element_count(&self) -> Result<u64, GgufError> {
        if self.dimensions.is_empty() {
            return Err(GgufError::InvalidTensorShape);
        }
        let mut n = 1u64;
        for &d in &self.dimensions {
            if d == 0 {
                return Err(GgufError::InvalidTensorShape);
            }
            n = n.checked_mul(d).ok_or(GgufError::InvalidTensorShape)?;
        }
        Ok(n)
    }

    pub fn encoded_len(&self) -> Result<u64, GgufError> {
        let n = self.element_count()?;
        let type_size = self
            .ty
            .type_size()
            .ok_or(GgufError::UnsupportedTensorType(self.ty.raw()))?;
        let block_size = self
            .ty
            .block_size()
            .ok_or(GgufError::UnsupportedTensorType(self.ty.raw()))?;
        if n % block_size != 0 {
            return Err(GgufError::InvalidTensorShape);
        }
        (n / block_size)
            .checked_mul(type_size)
            .ok_or(GgufError::InvalidTensorShape)
    }
}

fn read_value(r: &mut Reader<'_>) -> Result<MetadataValue, GgufError> {
    let ty = r.u32()?;
    match ty {
        0 => Ok(MetadataValue::U8(r.u8()?)),
        1 => Ok(MetadataValue::I8(r.u8()? as i8)),
        2 => Ok(MetadataValue::U16(r.u16()?)),
        3 => Ok(MetadataValue::I16(r.u16()? as i16)),
        4 => Ok(MetadataValue::U32(r.u32()?)),
        5 => Ok(MetadataValue::I32(r.i32()?)),
        6 => Ok(MetadataValue::F32(r.f32()?)),
        7 => Ok(MetadataValue::Bool(r.bool()?)),
        8 => Ok(MetadataValue::String(r.string()?)),
        9 => {
            let elem_ty = r.u32()?;
            let n = r.u64()?;
            let min_elem_bytes = match elem_ty {
                0 | 1 | 7 => 1,
                2 | 3 => 2,
                4..=6 => 4,
                8 | 10..=12 => 8,
                other => return Err(GgufError::UnsupportedValueType(other)),
            };
            r.ensure_count(n, min_elem_bytes)?;
            let capacity = usize_from_u64(n)?;
            match elem_ty {
                0 => {
                    let mut out = Vec::with_capacity(capacity);
                    for _ in 0..n {
                        out.push(r.u8()?);
                    }
                    Ok(MetadataValue::ArrayU8(out))
                }
                1 => {
                    let mut out = Vec::with_capacity(capacity);
                    for _ in 0..n {
                        out.push(r.u8()? as i8);
                    }
                    Ok(MetadataValue::ArrayI8(out))
                }
                2 => {
                    let mut out = Vec::with_capacity(capacity);
                    for _ in 0..n {
                        out.push(r.u16()?);
                    }
                    Ok(MetadataValue::ArrayU16(out))
                }
                3 => {
                    let mut out = Vec::with_capacity(capacity);
                    for _ in 0..n {
                        out.push(r.u16()? as i16);
                    }
                    Ok(MetadataValue::ArrayI16(out))
                }
                4 => {
                    let mut out = Vec::with_capacity(capacity);
                    for _ in 0..n {
                        out.push(r.u32()?);
                    }
                    Ok(MetadataValue::ArrayU32(out))
                }
                5 => {
                    let mut out = Vec::with_capacity(capacity);
                    for _ in 0..n {
                        out.push(r.i32()?);
                    }
                    Ok(MetadataValue::ArrayI32(out))
                }
                6 => {
                    let mut out = Vec::with_capacity(capacity);
                    for _ in 0..n {
                        out.push(r.f32()?);
                    }
                    Ok(MetadataValue::ArrayF32(out))
                }
                7 => {
                    let mut out = Vec::with_capacity(capacity);
                    for _ in 0..n {
                        out.push(r.bool()?);
                    }
                    Ok(MetadataValue::ArrayBool(out))
                }
                8 => {
                    let mut out = Vec::with_capacity(capacity);
                    for _ in 0..n {
                        out.push(r.string()?);
                    }
                    Ok(MetadataValue::ArrayString(out))
                }
                10 => {
                    let mut out = Vec::with_capacity(capacity);
                    for _ in 0..n {
                        out.push(r.u64()?);
                    }
                    Ok(MetadataValue::ArrayU64(out))
                }
                11 => {
                    let mut out = Vec::with_capacity(capacity);
                    for _ in 0..n {
                        out.push(r.u64()? as i64);
                    }
                    Ok(MetadataValue::ArrayI64(out))
                }
                12 => {
                    let mut out = Vec::with_capacity(capacity);
                    for _ in 0..n {
                        out.push(r.f64()?);
                    }
                    Ok(MetadataValue::ArrayF64(out))
                }
                other => Err(GgufError::UnsupportedValueType(other)),
            }
        }
        10 => Ok(MetadataValue::U64(r.u64()?)),
        11 => Ok(MetadataValue::I64(r.u64()? as i64)),
        12 => Ok(MetadataValue::F64(r.f64()?)),
        other => Err(GgufError::UnsupportedValueType(other)),
    }
}

fn align_up(x: usize, align: usize) -> Option<usize> {
    Some(x.checked_add(align.checked_sub(1)?)? & !(align - 1))
}

fn usize_from_u64(value: u64) -> Result<usize, GgufError> {
    usize::try_from(value).map_err(|_| GgufError::Truncated)
}

fn usize_from_u32(value: u32) -> Result<usize, GgufError> {
    usize::try_from(value).map_err(|_| GgufError::Truncated)
}

struct Reader<'a> {
    bytes: &'a [u8],
    pos: usize,
}

impl<'a> Reader<'a> {
    fn read_exact(&mut self, n: usize) -> Result<&'a [u8], GgufError> {
        let end = self.pos.checked_add(n).ok_or(GgufError::Truncated)?;
        if end > self.bytes.len() {
            return Err(GgufError::Truncated);
        }
        let out = &self.bytes[self.pos..end];
        self.pos = end;
        Ok(out)
    }

    fn ensure_count(&self, n: u64, min_item_bytes: usize) -> Result<(), GgufError> {
        let n = usize_from_u64(n)?;
        let min_len = n.checked_mul(min_item_bytes).ok_or(GgufError::Truncated)?;
        let end = self.pos.checked_add(min_len).ok_or(GgufError::Truncated)?;
        if end > self.bytes.len() {
            return Err(GgufError::Truncated);
        }
        Ok(())
    }

    fn u8(&mut self) -> Result<u8, GgufError> {
        Ok(self.read_exact(1)?[0])
    }

    fn u32(&mut self) -> Result<u32, GgufError> {
        let b = self.read_exact(4)?;
        Ok(u32::from_le_bytes([b[0], b[1], b[2], b[3]]))
    }

    fn u16(&mut self) -> Result<u16, GgufError> {
        let b = self.read_exact(2)?;
        Ok(u16::from_le_bytes([b[0], b[1]]))
    }

    fn i32(&mut self) -> Result<i32, GgufError> {
        Ok(self.u32()? as i32)
    }

    fn u64(&mut self) -> Result<u64, GgufError> {
        let b = self.read_exact(8)?;
        Ok(u64::from_le_bytes([
            b[0], b[1], b[2], b[3], b[4], b[5], b[6], b[7],
        ]))
    }

    fn f32(&mut self) -> Result<f32, GgufError> {
        Ok(f32::from_bits(self.u32()?))
    }

    fn f64(&mut self) -> Result<f64, GgufError> {
        Ok(f64::from_bits(self.u64()?))
    }

    fn bool(&mut self) -> Result<bool, GgufError> {
        match self.u8()? {
            0 => Ok(false),
            1 => Ok(true),
            _ => Err(GgufError::InvalidBool),
        }
    }

    fn string(&mut self) -> Result<String, GgufError> {
        let n = usize_from_u64(self.u64()?)?;
        let b = self.read_exact(n)?;
        std::str::from_utf8(b)
            .map_err(|_| GgufError::InvalidUtf8)
            .map(ToOwned::to_owned)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn rejects_bad_magic() {
        assert!(matches!(parse(b"nope"), Err(GgufError::BadMagic)));
    }

    #[test]
    fn rejects_missing_aligned_tensor_data_region() {
        let mut bytes = Vec::new();
        bytes.extend_from_slice(b"GGUF");
        bytes.extend_from_slice(&3u32.to_le_bytes());
        bytes.extend_from_slice(&0u64.to_le_bytes());
        bytes.extend_from_slice(&0u64.to_le_bytes());

        assert!(matches!(parse(&bytes), Err(GgufError::Truncated)));

        while bytes.len() % 32 != 0 {
            bytes.push(0);
        }
        parse(&bytes).expect("aligned header without tensors is valid");
    }

    #[test]
    fn rejects_impossible_counts_before_allocating() {
        let mut tensor_count = Vec::new();
        tensor_count.extend_from_slice(b"GGUF");
        tensor_count.extend_from_slice(&3u32.to_le_bytes());
        tensor_count.extend_from_slice(&u64::MAX.to_le_bytes());
        tensor_count.extend_from_slice(&0u64.to_le_bytes());
        assert!(matches!(parse(&tensor_count), Err(GgufError::Truncated)));

        let mut array_len = Vec::new();
        array_len.extend_from_slice(b"GGUF");
        array_len.extend_from_slice(&3u32.to_le_bytes());
        array_len.extend_from_slice(&0u64.to_le_bytes());
        array_len.extend_from_slice(&1u64.to_le_bytes());
        push_string(&mut array_len, "huge");
        array_len.extend_from_slice(&9u32.to_le_bytes());
        array_len.extend_from_slice(&0u32.to_le_bytes());
        array_len.extend_from_slice(&u64::MAX.to_le_bytes());
        assert!(matches!(parse(&array_len), Err(GgufError::Truncated)));
    }

    #[test]
    fn rejects_duplicate_metadata_keys_and_tensor_names() {
        let mut metadata = Vec::new();
        metadata.extend_from_slice(b"GGUF");
        metadata.extend_from_slice(&3u32.to_le_bytes());
        metadata.extend_from_slice(&0u64.to_le_bytes());
        metadata.extend_from_slice(&2u64.to_le_bytes());
        push_u32_metadata(&mut metadata, "dup", 1);
        push_u32_metadata(&mut metadata, "dup", 2);
        assert!(matches!(
            parse(&metadata),
            Err(GgufError::DuplicateMetadataKey)
        ));

        let mut tensors = Vec::new();
        tensors.extend_from_slice(b"GGUF");
        tensors.extend_from_slice(&3u32.to_le_bytes());
        tensors.extend_from_slice(&2u64.to_le_bytes());
        tensors.extend_from_slice(&0u64.to_le_bytes());
        push_tensor_header(&mut tensors, "dup");
        push_tensor_header(&mut tensors, "dup");
        assert!(matches!(
            parse(&tensors),
            Err(GgufError::DuplicateTensorName)
        ));
    }

    #[test]
    fn tensor_lookup_rejects_duplicate_names_from_public_parts() {
        let tensors = vec![
            TensorInfo {
                name: "dup".to_owned(),
                dimensions: vec![1],
                ty: GgmlType::F32,
                offset: 0,
            },
            TensorInfo {
                name: "dup".to_owned(),
                dimensions: vec![1],
                ty: GgmlType::F32,
                offset: 4,
            },
        ];
        let bytes = vec![0; 8];
        let gguf = Gguf::from_parts(3, BTreeMap::new(), tensors, 0, bytes.len());

        assert_eq!(gguf.tensor("dup"), Err(GgufError::DuplicateTensorName));
        assert_eq!(
            gguf.tensor_data(&bytes, "dup"),
            Err(GgufError::DuplicateTensorName)
        );
    }

    #[test]
    fn rejects_overlapping_or_out_of_bounds_tensor_data() {
        let mut overlapping = empty_header_with_counts(2, 0);
        push_tensor_header_with_offset(&mut overlapping, "a", 0);
        push_tensor_header_with_offset(&mut overlapping, "b", 2);
        pad_to_data(&mut overlapping);
        overlapping.extend_from_slice(&[0; 6]);
        assert!(matches!(
            parse(&overlapping),
            Err(GgufError::InvalidTensorOffset)
        ));

        let mut out_of_bounds = empty_header_with_counts(1, 0);
        push_tensor_header_with_offset(&mut out_of_bounds, "a", 4);
        pad_to_data(&mut out_of_bounds);
        out_of_bounds.extend_from_slice(&[0; 4]);
        assert!(matches!(parse(&out_of_bounds), Err(GgufError::Truncated)));
    }

    #[test]
    fn rejects_noncanonical_bool_metadata_values() {
        let mut scalar = empty_header_with_counts(0, 1);
        push_string(&mut scalar, "flag");
        scalar.extend_from_slice(&7u32.to_le_bytes());
        scalar.push(2);
        assert!(matches!(parse(&scalar), Err(GgufError::InvalidBool)));

        let mut array = empty_header_with_counts(0, 1);
        push_array_header(&mut array, "flags", 7, 3);
        array.extend_from_slice(&[0, 1, 2]);
        assert!(matches!(parse(&array), Err(GgufError::InvalidBool)));
    }

    #[test]
    fn computes_tensor_encoded_lengths() {
        let f32_tensor = TensorInfo {
            name: "a".to_owned(),
            dimensions: vec![4, 8],
            ty: GgmlType::F32,
            offset: 0,
        };
        assert_eq!(f32_tensor.encoded_len(), Ok(128));

        let q8 = TensorInfo {
            name: "b".to_owned(),
            dimensions: vec![32, 3],
            ty: GgmlType::Q8_0,
            offset: 0,
        };
        assert_eq!(q8.encoded_len(), Ok(102));

        let bad = TensorInfo {
            name: "c".to_owned(),
            dimensions: vec![31],
            ty: GgmlType::Q4_0,
            offset: 0,
        };
        assert_eq!(bad.encoded_len(), Err(GgufError::InvalidTensorShape));

        let huge = TensorInfo {
            name: "huge".to_owned(),
            dimensions: vec![u64::MAX],
            ty: GgmlType::F32,
            offset: 0,
        };
        assert_eq!(huge.encoded_len(), Err(GgufError::InvalidTensorShape));
    }

    #[test]
    fn parses_all_supported_metadata_array_types() {
        let mut bytes = Vec::new();
        bytes.extend_from_slice(b"GGUF");
        bytes.extend_from_slice(&3u32.to_le_bytes());
        bytes.extend_from_slice(&0u64.to_le_bytes());
        bytes.extend_from_slice(&12u64.to_le_bytes());

        push_array_header(&mut bytes, "u8s", 0, 2);
        bytes.extend_from_slice(&[1, 2]);
        push_array_header(&mut bytes, "i8s", 1, 2);
        bytes.extend_from_slice(&[255, 2]);
        push_array_header(&mut bytes, "u16s", 2, 2);
        bytes.extend_from_slice(&1u16.to_le_bytes());
        bytes.extend_from_slice(&2u16.to_le_bytes());
        push_array_header(&mut bytes, "i16s", 3, 2);
        bytes.extend_from_slice(&(-1i16).to_le_bytes());
        bytes.extend_from_slice(&2i16.to_le_bytes());
        push_array_header(&mut bytes, "u32s", 4, 2);
        bytes.extend_from_slice(&1u32.to_le_bytes());
        bytes.extend_from_slice(&2u32.to_le_bytes());
        push_array_header(&mut bytes, "i32s", 5, 2);
        bytes.extend_from_slice(&(-1i32).to_le_bytes());
        bytes.extend_from_slice(&2i32.to_le_bytes());
        push_array_header(&mut bytes, "f32s", 6, 2);
        bytes.extend_from_slice(&1.5f32.to_bits().to_le_bytes());
        bytes.extend_from_slice(&(-2.25f32).to_bits().to_le_bytes());
        push_array_header(&mut bytes, "bools", 7, 2);
        bytes.extend_from_slice(&[0, 1]);
        push_array_header(&mut bytes, "strings", 8, 2);
        push_string(&mut bytes, "a");
        push_string(&mut bytes, "bc");
        push_array_header(&mut bytes, "u64s", 10, 2);
        bytes.extend_from_slice(&1u64.to_le_bytes());
        bytes.extend_from_slice(&2u64.to_le_bytes());
        push_array_header(&mut bytes, "i64s", 11, 2);
        bytes.extend_from_slice(&(-1i64).to_le_bytes());
        bytes.extend_from_slice(&2i64.to_le_bytes());
        push_array_header(&mut bytes, "f64s", 12, 2);
        bytes.extend_from_slice(&1.5f64.to_bits().to_le_bytes());
        bytes.extend_from_slice(&(-2.25f64).to_bits().to_le_bytes());

        while bytes.len() % 32 != 0 {
            bytes.push(0);
        }

        let gguf = parse(&bytes).expect("parse");
        assert_eq!(
            gguf.metadata_value("u8s"),
            Ok(&MetadataValue::ArrayU8(vec![1, 2]))
        );
        assert_eq!(
            gguf.metadata_value("i8s"),
            Ok(&MetadataValue::ArrayI8(vec![-1, 2]))
        );
        assert_eq!(
            gguf.metadata_value("u16s"),
            Ok(&MetadataValue::ArrayU16(vec![1, 2]))
        );
        assert_eq!(
            gguf.metadata_value("i16s"),
            Ok(&MetadataValue::ArrayI16(vec![-1, 2]))
        );
        assert_eq!(
            gguf.metadata_value("u32s"),
            Ok(&MetadataValue::ArrayU32(vec![1, 2]))
        );
        assert_eq!(
            gguf.metadata_value("i32s"),
            Ok(&MetadataValue::ArrayI32(vec![-1, 2]))
        );
        assert_eq!(
            gguf.metadata_value("f32s"),
            Ok(&MetadataValue::ArrayF32(vec![1.5, -2.25]))
        );
        assert_eq!(
            gguf.metadata_value("bools"),
            Ok(&MetadataValue::ArrayBool(vec![false, true]))
        );
        assert_eq!(
            gguf.metadata_value("strings"),
            Ok(&MetadataValue::ArrayString(vec![
                "a".to_owned(),
                "bc".to_owned()
            ]))
        );
        assert_eq!(
            gguf.metadata_value("u64s"),
            Ok(&MetadataValue::ArrayU64(vec![1, 2]))
        );
        assert_eq!(
            gguf.metadata_value("i64s"),
            Ok(&MetadataValue::ArrayI64(vec![-1, 2]))
        );
        assert_eq!(
            gguf.metadata_value("f64s"),
            Ok(&MetadataValue::ArrayF64(vec![1.5, -2.25]))
        );
    }

    fn empty_header_with_counts(tensor_count: u64, metadata_count: u64) -> Vec<u8> {
        let mut out = Vec::new();
        out.extend_from_slice(b"GGUF");
        out.extend_from_slice(&3u32.to_le_bytes());
        out.extend_from_slice(&tensor_count.to_le_bytes());
        out.extend_from_slice(&metadata_count.to_le_bytes());
        out
    }

    fn pad_to_data(out: &mut Vec<u8>) {
        while out.len() % 32 != 0 {
            out.push(0);
        }
    }

    fn push_string(out: &mut Vec<u8>, s: &str) {
        out.extend_from_slice(&(s.len() as u64).to_le_bytes());
        out.extend_from_slice(s.as_bytes());
    }

    fn push_u32_metadata(out: &mut Vec<u8>, key: &str, value: u32) {
        push_string(out, key);
        out.extend_from_slice(&4u32.to_le_bytes());
        out.extend_from_slice(&value.to_le_bytes());
    }

    fn push_tensor_header(out: &mut Vec<u8>, name: &str) {
        push_tensor_header_with_offset(out, name, 0);
    }

    fn push_tensor_header_with_offset(out: &mut Vec<u8>, name: &str, offset: u64) {
        push_string(out, name);
        out.extend_from_slice(&1u32.to_le_bytes());
        out.extend_from_slice(&1u64.to_le_bytes());
        out.extend_from_slice(&0u32.to_le_bytes());
        out.extend_from_slice(&offset.to_le_bytes());
    }

    fn push_array_header(out: &mut Vec<u8>, key: &str, elem_ty: u32, len: u64) {
        push_string(out, key);
        out.extend_from_slice(&9u32.to_le_bytes());
        out.extend_from_slice(&elem_ty.to_le_bytes());
        out.extend_from_slice(&len.to_le_bytes());
    }
}
