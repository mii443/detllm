pub const MAGIC: [u8; 4] = *b"DTLZ";
pub const VERSION: u16 = 1;
pub const FLAGS: u16 = 0;
pub const HEADER_LEN: usize = 4 + 2 + 2 + 32 + 4 + 4 + 8;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct DtlzHeader {
    pub flags: u16,
    pub model_sha256: [u8; 32],
    pub n_ctx: u32,
    pub overlap: u32,
    pub orig_len: u64,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum FileError {
    TooShort,
    BadMagic,
    UnsupportedVersion(u16),
    UnsupportedFlags(u16),
    InvalidContext,
    InvalidOverlap,
}

impl DtlzHeader {
    pub fn encode(self) -> [u8; HEADER_LEN] {
        let mut out = [0u8; HEADER_LEN];
        out[0..4].copy_from_slice(&MAGIC);
        out[4..6].copy_from_slice(&VERSION.to_le_bytes());
        out[6..8].copy_from_slice(&self.flags.to_le_bytes());
        out[8..40].copy_from_slice(&self.model_sha256);
        out[40..44].copy_from_slice(&self.n_ctx.to_le_bytes());
        out[44..48].copy_from_slice(&self.overlap.to_le_bytes());
        out[48..56].copy_from_slice(&self.orig_len.to_le_bytes());
        out
    }

    pub fn encode_checked(self) -> Result<[u8; HEADER_LEN], FileError> {
        self.validate()?;
        Ok(self.encode())
    }

    pub fn validate(self) -> Result<(), FileError> {
        if self.flags != FLAGS {
            return Err(FileError::UnsupportedFlags(self.flags));
        }
        if self.n_ctx == 0 {
            return Err(FileError::InvalidContext);
        }
        if self.overlap >= self.n_ctx {
            return Err(FileError::InvalidOverlap);
        }
        Ok(())
    }

    pub fn decode(bytes: &[u8]) -> Result<Self, FileError> {
        if bytes.len() < HEADER_LEN {
            return Err(FileError::TooShort);
        }
        if bytes[0..4] != MAGIC {
            return Err(FileError::BadMagic);
        }
        let version = u16::from_le_bytes([bytes[4], bytes[5]]);
        if version != VERSION {
            return Err(FileError::UnsupportedVersion(version));
        }
        let flags = u16::from_le_bytes([bytes[6], bytes[7]]);
        if flags != FLAGS {
            return Err(FileError::UnsupportedFlags(flags));
        }
        let mut model_sha256 = [0u8; 32];
        model_sha256.copy_from_slice(&bytes[8..40]);
        let n_ctx = u32::from_le_bytes([bytes[40], bytes[41], bytes[42], bytes[43]]);
        let overlap = u32::from_le_bytes([bytes[44], bytes[45], bytes[46], bytes[47]]);
        let orig_len = u64::from_le_bytes([
            bytes[48], bytes[49], bytes[50], bytes[51], bytes[52], bytes[53], bytes[54], bytes[55],
        ]);
        let header = Self {
            flags,
            model_sha256,
            n_ctx,
            overlap,
            orig_len,
        };
        header.validate()?;
        Ok(header)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn header_round_trips() {
        let h = DtlzHeader {
            flags: FLAGS,
            model_sha256: [7; 32],
            n_ctx: 2048,
            overlap: 512,
            orig_len: 123456789,
        };
        assert_eq!(h.validate(), Ok(()));
        assert_eq!(h.encode_checked(), Ok(h.encode()));
        assert_eq!(DtlzHeader::decode(&h.encode()), Ok(h));
    }

    #[test]
    fn rejects_unsupported_flags() {
        let h = DtlzHeader {
            flags: 1,
            model_sha256: [7; 32],
            n_ctx: 2048,
            overlap: 512,
            orig_len: 123456789,
        };
        assert_eq!(h.validate(), Err(FileError::UnsupportedFlags(1)));
        assert_eq!(h.encode_checked(), Err(FileError::UnsupportedFlags(1)));
        assert_eq!(
            DtlzHeader::decode(&h.encode()),
            Err(FileError::UnsupportedFlags(1))
        );
    }

    #[test]
    fn rejects_malformed_header_envelope() {
        assert_eq!(DtlzHeader::decode(&[]), Err(FileError::TooShort));
        assert_eq!(
            DtlzHeader::decode(&[0; HEADER_LEN - 1]),
            Err(FileError::TooShort)
        );

        let h = DtlzHeader {
            flags: FLAGS,
            model_sha256: [7; 32],
            n_ctx: 2048,
            overlap: 512,
            orig_len: 123456789,
        };
        let mut bytes = h.encode();
        bytes[0..4].copy_from_slice(b"NOPE");
        assert_eq!(DtlzHeader::decode(&bytes), Err(FileError::BadMagic));

        let mut bytes = h.encode();
        bytes[4..6].copy_from_slice(&(VERSION + 1).to_le_bytes());
        assert_eq!(
            DtlzHeader::decode(&bytes),
            Err(FileError::UnsupportedVersion(VERSION + 1))
        );
    }

    #[test]
    fn rejects_invalid_window_fields() {
        let h = DtlzHeader {
            flags: FLAGS,
            model_sha256: [7; 32],
            n_ctx: 0,
            overlap: 0,
            orig_len: 123456789,
        };
        assert_eq!(h.validate(), Err(FileError::InvalidContext));
        assert_eq!(h.encode_checked(), Err(FileError::InvalidContext));
        assert_eq!(
            DtlzHeader::decode(&h.encode()),
            Err(FileError::InvalidContext)
        );

        let h = DtlzHeader {
            flags: FLAGS,
            model_sha256: [7; 32],
            n_ctx: 8,
            overlap: 8,
            orig_len: 123456789,
        };
        assert_eq!(h.validate(), Err(FileError::InvalidOverlap));
        assert_eq!(h.encode_checked(), Err(FileError::InvalidOverlap));
        assert_eq!(
            DtlzHeader::decode(&h.encode()),
            Err(FileError::InvalidOverlap)
        );
    }
}
