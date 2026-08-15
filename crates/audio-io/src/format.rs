use crate::error::{AudioError, Result};

/// Pipeline-internal audio format. We always operate on mono f32 @ 48 kHz.
/// Resampling/channel-mixing happens in the WASAPI wrappers when the device's
/// mix format differs.
#[derive(Debug, Clone, Copy)]
pub struct StreamFormat {
    pub sample_rate: u32,
    pub channels: u16,
}

impl StreamFormat {
    pub const PIPELINE: Self = Self {
        sample_rate: 48_000,
        channels: 1,
    };
}

/// The fields of a device's WASAPI mix format that the audio loops care
/// about, snapshotted out of the (packed, possibly extensible) `WAVEFORMATEX`
/// the engine hands back from `GetMixFormat`.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct MixFormat {
    pub sample_rate: u32,
    pub channels: u16,
    pub bits_per_sample: u16,
    /// Bytes per frame (all channels). For 32-bit float this is `channels * 4`.
    pub block_align: u16,
    /// True when the sample type is IEEE float — either `WAVE_FORMAT_IEEE_FLOAT`
    /// or `WAVE_FORMAT_EXTENSIBLE` carrying the IEEE-float subformat GUID.
    pub is_float: bool,
}

impl MixFormat {
    /// Both audio loops reinterpret the engine's raw byte buffer as `[f32]`.
    /// That is only sound when the device really is mixing 32-bit float: on a
    /// 16-bit device we would read (capture) or write (render) twice the bytes
    /// the engine allocated, walking off the end of the shared buffer. The
    /// mix format is *usually* float32 in shared mode, but nothing in WASAPI
    /// guarantees it, so refuse the device instead of trusting it.
    pub fn validate(&self) -> Result<()> {
        if self.channels == 0 {
            return Err(AudioError::UnsupportedFormat(
                "device reports 0 channels".into(),
            ));
        }
        if !self.is_float || self.bits_per_sample != 32 {
            return Err(AudioError::UnsupportedFormat(format!(
                "device mixes {} {}-bit samples; RoomMute needs 32-bit IEEE float",
                if self.is_float {
                    "float"
                } else {
                    "integer PCM"
                },
                self.bits_per_sample
            )));
        }
        let expected = u32::from(self.channels) * 4;
        if u32::from(self.block_align) != expected {
            return Err(AudioError::UnsupportedFormat(format!(
                "nBlockAlign is {} but {} channels of 32-bit float need {expected}",
                self.block_align, self.channels
            )));
        }
        Ok(())
    }
}

/// The `WAVEFORMATEX` the audio engine allocated for us, freed on drop.
///
/// `GetMixFormat` hands over a `CoTaskMem` allocation that the caller owns.
/// Freeing it by hand meant a `CoTaskMemFree` on the validation-failure path
/// and another after `Initialize`, in each of two files — four chances to
/// leak, and each new early return added a fifth.
///
/// The original pointer has to be handed back to `Initialize` unchanged: on
/// most devices this is really a `WAVEFORMATEXTENSIBLE`, and copying it into
/// an 18-byte `WAVEFORMATEX` truncates the extensible tail, which the engine
/// rejects with `E_INVALIDARG`. So the guard lends the pointer out rather than
/// owning a decoded copy.
#[cfg(windows)]
pub(crate) struct EngineMixFormat(*mut windows::Win32::Media::Audio::WAVEFORMATEX);

#[cfg(windows)]
impl EngineMixFormat {
    /// # Safety
    /// `ptr` must be the `CoTaskMem` allocation returned by `GetMixFormat`,
    /// and must not be freed by anyone else.
    pub unsafe fn from_engine(ptr: *mut windows::Win32::Media::Audio::WAVEFORMATEX) -> Self {
        Self(ptr)
    }

    pub fn decode(&self) -> MixFormat {
        // Safe by this type's invariant: we own a valid engine allocation.
        unsafe { read_mix_format(self.0) }
    }

    /// The untouched pointer, for handing back to `Initialize`.
    pub fn as_ptr(&self) -> *const windows::Win32::Media::Audio::WAVEFORMATEX {
        self.0
    }
}

#[cfg(windows)]
impl Drop for EngineMixFormat {
    fn drop(&mut self) {
        unsafe { windows::Win32::System::Com::CoTaskMemFree(Some(self.0 as _)) }
    }
}

/// Read the mix format out of a `GetMixFormat` pointer.
///
/// # Safety
/// `ptr` must be a valid `WAVEFORMATEX` from the audio engine. When
/// `wFormatTag` is `WAVE_FORMAT_EXTENSIBLE` the engine guarantees the
/// extensible tail is present, but we check `cbSize` before reading it
/// anyway rather than take that on faith.
#[cfg(windows)]
unsafe fn read_mix_format(ptr: *const windows::Win32::Media::Audio::WAVEFORMATEX) -> MixFormat {
    use windows::Win32::Media::Audio::WAVEFORMATEXTENSIBLE;
    use windows::Win32::Media::KernelStreaming::WAVE_FORMAT_EXTENSIBLE;
    use windows::Win32::Media::Multimedia::{
        KSDATAFORMAT_SUBTYPE_IEEE_FLOAT, WAVE_FORMAT_IEEE_FLOAT,
    };

    let format_tag = (*ptr).wFormatTag as u32;
    let cb_size = (*ptr).cbSize;

    let is_float = if format_tag == WAVE_FORMAT_IEEE_FLOAT {
        true
    } else if format_tag == WAVE_FORMAT_EXTENSIBLE && cb_size >= 22 {
        // Packed struct: read the GUID out unaligned rather than borrowing it.
        let ext = ptr as *const WAVEFORMATEXTENSIBLE;
        std::ptr::addr_of!((*ext).SubFormat).read_unaligned() == KSDATAFORMAT_SUBTYPE_IEEE_FLOAT
    } else {
        false
    };

    MixFormat {
        sample_rate: (*ptr).nSamplesPerSec,
        channels: (*ptr).nChannels,
        bits_per_sample: (*ptr).wBitsPerSample,
        block_align: (*ptr).nBlockAlign,
        is_float,
    }
}

#[cfg(test)]
mod tests {
    use super::MixFormat;

    fn float32(channels: u16) -> MixFormat {
        MixFormat {
            sample_rate: 48_000,
            channels,
            bits_per_sample: 32,
            block_align: channels * 4,
            is_float: true,
        }
    }

    #[test]
    fn accepts_float32_mix_formats() {
        assert!(float32(1).validate().is_ok());
        assert!(float32(2).validate().is_ok());
        assert!(float32(8).validate().is_ok());
    }

    #[test]
    fn rejects_integer_pcm() {
        // The dangerous case: 16-bit PCM would make us read/write 2x the
        // bytes the engine allocated.
        let f = MixFormat {
            bits_per_sample: 16,
            block_align: 4,
            is_float: false,
            ..float32(2)
        };
        assert!(f.validate().is_err());
    }

    #[test]
    fn rejects_float64_and_mismatched_block_align() {
        let wide = MixFormat {
            bits_per_sample: 64,
            block_align: 16,
            ..float32(2)
        };
        assert!(wide.validate().is_err());

        let lying = MixFormat {
            block_align: 6,
            ..float32(2)
        };
        assert!(lying.validate().is_err());
    }

    #[test]
    fn rejects_zero_channels() {
        let f = MixFormat {
            channels: 0,
            block_align: 0,
            ..float32(1)
        };
        assert!(f.validate().is_err());
    }

    /// A rejection the user cannot act on is barely better than a crash.
    #[test]
    fn a_refusal_says_what_the_device_actually_does() {
        let pcm = MixFormat {
            bits_per_sample: 16,
            block_align: 4,
            is_float: false,
            ..float32(2)
        };
        let msg = pcm.validate().unwrap_err().to_string();
        assert!(msg.contains("16"), "name the bit depth found: {msg}");
        assert!(msg.contains("integer PCM"), "name the sample type: {msg}");

        let lying = MixFormat {
            block_align: 6,
            ..float32(2)
        };
        let msg = lying.validate().unwrap_err().to_string();
        assert!(
            msg.contains('6') && msg.contains('8'),
            "give both the claimed and the required stride: {msg}"
        );
    }

    /// Decoding the engine's header, which is where the pointer arithmetic
    /// lives. The structs are ordinary `#[repr(C)]` data, so they can be built
    /// by hand — no audio engine and no device required.
    #[cfg(windows)]
    mod headers {
        use windows::core::GUID;
        use windows::Win32::Media::Audio::{WAVEFORMATEX, WAVEFORMATEXTENSIBLE};
        use windows::Win32::Media::KernelStreaming::WAVE_FORMAT_EXTENSIBLE;
        use windows::Win32::Media::Multimedia::{
            KSDATAFORMAT_SUBTYPE_IEEE_FLOAT, WAVE_FORMAT_IEEE_FLOAT,
        };

        use crate::format::read_mix_format;

        const PCM_TAG: u16 = 1;
        /// KSDATAFORMAT_SUBTYPE_PCM, spelled out because the `windows` crate
        /// does not re-export it beside the float one.
        const SUBTYPE_PCM: GUID = GUID::from_u128(0x0000_0001_0000_0010_8000_00aa_0038_9b71);

        fn header(tag: u16, bits: u16, channels: u16, cb_size: u16) -> WAVEFORMATEX {
            let block = channels * (bits / 8);
            WAVEFORMATEX {
                wFormatTag: tag,
                nChannels: channels,
                nSamplesPerSec: 48_000,
                nAvgBytesPerSec: 48_000 * u32::from(block),
                nBlockAlign: block,
                wBitsPerSample: bits,
                cbSize: cb_size,
            }
        }

        fn extensible(subformat: windows::core::GUID, cb_size: u16) -> WAVEFORMATEXTENSIBLE {
            WAVEFORMATEXTENSIBLE {
                Format: header(WAVE_FORMAT_EXTENSIBLE as u16, 32, 2, cb_size),
                SubFormat: subformat,
                ..Default::default()
            }
        }

        #[test]
        fn a_plain_float_header_decodes_field_for_field() {
            let h = header(WAVE_FORMAT_IEEE_FLOAT as u16, 32, 2, 0);
            let got = unsafe { read_mix_format(&h) };
            assert!(got.is_float);
            assert_eq!(got.sample_rate, 48_000);
            assert_eq!(got.channels, 2);
            assert_eq!(got.bits_per_sample, 32);
            assert_eq!(got.block_align, 8);
            assert!(got.validate().is_ok(), "this is the format we want");
        }

        #[test]
        fn integer_pcm_is_not_mistaken_for_float() {
            let h = header(PCM_TAG, 16, 2, 0);
            let got = unsafe { read_mix_format(&h) };
            assert!(!got.is_float, "16-bit PCM must not decode as float");
            assert!(
                got.validate().is_err(),
                "reinterpreting this buffer as f32 would read twice the bytes the \
                 engine allocated"
            );
        }

        #[test]
        fn an_extensible_header_is_read_through_to_its_subformat() {
            let float = extensible(KSDATAFORMAT_SUBTYPE_IEEE_FLOAT, 22);
            assert!(
                unsafe { read_mix_format(&float.Format) }.is_float,
                "the float subformat GUID is how real shared-mode devices say float"
            );

            let pcm = extensible(SUBTYPE_PCM, 22);
            assert!(
                !unsafe { read_mix_format(&pcm.Format) }.is_float,
                "an extensible PCM device is still PCM"
            );
        }

        /// `cbSize` is the header's own statement about how many extra bytes
        /// follow it. Below 22 there is no subformat GUID to read, and reading
        /// one anyway walks off the end of the engine's allocation. The tag
        /// alone must not be enough to go looking.
        #[test]
        fn an_extensible_header_without_a_declared_tail_is_not_read_further() {
            let lying = extensible(KSDATAFORMAT_SUBTYPE_IEEE_FLOAT, 0);
            assert!(
                !unsafe { read_mix_format(&lying.Format) }.is_float,
                "cbSize said there is no subformat, so none may be read — even \
                 though the bytes happen to be there in this test"
            );
        }
    }
}
