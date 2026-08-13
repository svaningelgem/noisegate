//! WASAPI shared low-latency capture.
//!
//! Uses `IAudioClient3::InitializeSharedAudioStream` so we don't take
//! exclusive ownership of the mic (other apps can still record, the system
//! mixer still works). The capture loop is event-driven: we wait on the
//! buffer-ready event the audio engine signals every period.
//!
//! Output of this module is always **mono f32 @ 48 kHz, 480-sample frames**.
//! If the device's mix format differs we down-mix and (linear) resample
//! inline. Linear resampling is fine for a near-rate match (most modern
//! mics already negotiate 48 kHz); for 44.1 → 48 we sound a small quality
//! cost for a much lighter dependency than rubato. Swap to `rubato` if
//! quality matters.

use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::Arc;

use windows::core::PCWSTR;
use windows::Win32::Devices::FunctionDiscovery::PKEY_Device_FriendlyName;
use windows::Win32::Foundation::WAIT_OBJECT_0;
use windows::Win32::Media::Audio::*;
use windows::Win32::System::Com::*;
use windows::Win32::System::Threading::{CreateEventW, WaitForSingleObject};

use crate::devices::{Device, DeviceDirection, DeviceList};
use crate::error::{AudioError, Result};
use crate::{FRAME_SAMPLES, SAMPLE_RATE};

#[allow(non_upper_case_globals)]
const CLSID_MMDeviceEnumerator: windows::core::GUID =
    windows::core::GUID::from_u128(0xBCDE0395_E52F_467C_8E3D_C4579291692E);

/// Frame produced by the capture loop: always 480 mono f32 samples.
pub type Frame = [f32; FRAME_SAMPLES];

/// Callback the capture thread invokes every 10 ms with a fresh frame.
/// Must be cheap and non-blocking — push to a ring buffer and return.
pub trait FrameSink: Send {
    fn on_frame(&mut self, frame: &Frame);
    /// Called when the audio engine reports glitches (data discontinuity,
    /// timestamp error, silent fill). Useful for logging xruns.
    fn on_glitch(&mut self, _flags: u32) {}
}

pub struct WasapiCapture {
    stop: Arc<AtomicBool>,
    thread: Option<std::thread::JoinHandle<()>>,
}

impl WasapiCapture {
    /// Open the given capture device and start delivering 480-sample frames
    /// to `sink`. The capture runs on a dedicated MMCSS "Pro Audio" thread.
    pub fn start(device_id: &str, mut sink: Box<dyn FrameSink>) -> Result<Self> {
        let stop = Arc::new(AtomicBool::new(false));
        let stop_thread = stop.clone();
        let device_id = device_id.to_string();

        let (ready_tx, ready_rx) = std::sync::mpsc::channel::<Result<()>>();

        let thread = std::thread::Builder::new()
            .name("noisegate-capture".into())
            .spawn(move || {
                let res = capture_loop(&device_id, &mut *sink, &stop_thread, &ready_tx);
                if let Err(e) = res {
                    tracing::error!(error = %e, "capture loop exited with error");
                    // ready_tx may already have been signaled; ignore.
                    let _ = ready_tx.send(Err(e));
                }
            })
            .map_err(|e| AudioError::Other(anyhow::anyhow!("spawn capture thread: {e}")))?;

        // Wait for the thread to finish init so callers learn about open/format
        // failures synchronously.
        ready_rx.recv().map_err(|_| AudioError::ThreadDied)??;

        Ok(Self {
            stop,
            thread: Some(thread),
        })
    }
}

impl Drop for WasapiCapture {
    fn drop(&mut self) {
        self.stop.store(true, Ordering::Release);
        if let Some(t) = self.thread.take() {
            let _ = t.join();
        }
    }
}

fn capture_loop(
    device_id: &str,
    sink: &mut dyn FrameSink,
    stop: &AtomicBool,
    ready_tx: &std::sync::mpsc::Sender<Result<()>>,
) -> Result<()> {
    unsafe {
        // COM init for this thread. STA isn't required for WASAPI; MTA is
        // simpler and matches the audio engine's threading model.
        let _ = CoInitializeEx(None, COINIT_MULTITHREADED);

        let enumerator: IMMDeviceEnumerator =
            CoCreateInstance(&CLSID_MMDeviceEnumerator, None, CLSCTX_ALL)
                .map_err(|e| AudioError::wasapi("CoCreateInstance(MMDeviceEnumerator)", e))?;

        let device = find_device(&enumerator, device_id, eCapture)?;

        let client: IAudioClient3 = device
            .Activate(CLSCTX_ALL, None)
            .map_err(|e| AudioError::wasapi("IMMDevice::Activate", e))?;

        // GetMixFormat returns a CoTaskMem-allocated WAVEFORMATEX. On most
        // devices this is actually a WAVEFORMATEXTENSIBLE (40 bytes, with
        // cbSize > 0). We MUST pass the original pointer back to Initialize
        // — copying it into a 18-byte WAVEFORMATEX truncates the extensible
        // header and the engine rejects it with E_INVALIDARG.
        let mix_ptr = client
            .GetMixFormat()
            .map_err(|e| AudioError::wasapi("GetMixFormat", e))?;

        // Snapshot the fields we need into locals (Copy), so we can drop
        // the pointer after Initialize and not worry about packed-struct
        // unaligned-reference issues elsewhere in the loop.
        let mix = crate::format::read_mix_format(mix_ptr);
        let device_rate = mix.sample_rate;
        let device_channels = mix.channels as usize;
        let needs_convert = !(device_rate == SAMPLE_RATE && device_channels == 1);

        // Always log the chosen format — useful for diagnosing "init OK but
        // no data" issues which are usually privacy permissions or wrong
        // device picks.
        tracing::info!(
            device_rate,
            device_channels,
            device_bits = mix.bits_per_sample,
            device_is_float = mix.is_float,
            needs_convert,
            "capture device format negotiated"
        );

        // The loop below reinterprets the engine's byte buffer as `[f32]`, so
        // bail out before Initialize if this device doesn't actually mix
        // 32-bit float. Free the format first — we own that allocation.
        if let Err(e) = mix.validate() {
            windows::Win32::System::Com::CoTaskMemFree(Some(mix_ptr as _));
            return Err(e);
        }

        // Legacy Initialize — accepts the device's native (possibly
        // extensible) mix format reliably. Buffer duration 0 = engine
        // default (~30 ms), fine for voice.
        let init_res = client.Initialize(
            AUDCLNT_SHAREMODE_SHARED,
            AUDCLNT_STREAMFLAGS_EVENTCALLBACK,
            0,       // hnsBufferDuration
            0,       // hnsPeriodicity (must be 0 in shared mode)
            mix_ptr, // pass the original pointer through
            None,
        );
        // Always free the format pointer, regardless of success/failure.
        windows::Win32::System::Com::CoTaskMemFree(Some(mix_ptr as _));
        init_res.map_err(|e| AudioError::wasapi("IAudioClient::Initialize", e))?;

        let event = CreateEventW(None, false, false, PCWSTR::null())
            .map_err(|e| AudioError::wasapi("CreateEventW", e))?;
        client
            .SetEventHandle(event)
            .map_err(|e| AudioError::wasapi("SetEventHandle", e))?;

        let cap_client: IAudioCaptureClient = client
            .GetService()
            .map_err(|e| AudioError::wasapi("GetService(IAudioCaptureClient)", e))?;

        // MMCSS: ask the scheduler to treat this as Pro Audio. Without this,
        // we'll get random scheduling delays under load and audible glitches.
        let _mmcss = crate::mmcss::ProAudio::set_for_current_thread();

        client
            .Start()
            .map_err(|e| AudioError::wasapi("IAudioClient::Start", e))?;

        // Init succeeded — unblock the caller.
        let _ = ready_tx.send(Ok(()));

        let mut accumulator = FrameAccumulator::new();
        let mut converter = if needs_convert {
            Some(InlineConverter::new(
                device_rate,
                device_channels,
                SAMPLE_RATE,
            ))
        } else {
            None
        };

        let start_time = std::time::Instant::now();
        // Reused when the engine hands us a buffer flagged silent; see below.
        let mut silence: Vec<f32> = Vec::new();
        let mut got_first_buffer = false;
        let mut last_silence_warn = std::time::Instant::now();
        let mut wait_count = 0u64;
        let mut timeout_count = 0u64;

        while !stop.load(Ordering::Acquire) {
            let wait = WaitForSingleObject(event, 200 /* ms */);
            wait_count += 1;
            if wait != WAIT_OBJECT_0 {
                timeout_count += 1;
                tracing::trace!(
                    wait_result = wait.0,
                    waits = wait_count,
                    timeouts = timeout_count,
                    "capture event timeout"
                );
                // Timeout. If we've never seen a buffer and >2s have passed,
                // that's almost certainly a Microphone-Privacy block in
                // Windows Settings. Tell the user once every ~5s.
                if !got_first_buffer
                    && start_time.elapsed() > std::time::Duration::from_secs(2)
                    && last_silence_warn.elapsed() > std::time::Duration::from_secs(5)
                {
                    tracing::error!(
                        "no audio from device after {:?}. \
                         Most common cause: Windows Settings → Privacy & Security → \
                         Microphone is OFF for this app. Also check that \
                         'Let desktop apps access your microphone' is ON, \
                         and that the mic isn't hardware-muted.",
                        start_time.elapsed()
                    );
                    last_silence_warn = std::time::Instant::now();
                }
                continue;
            }
            tracing::trace!(waits = wait_count, "capture event signaled");

            // Drain everything the engine has for us this tick.
            loop {
                let mut buffer_ptr: *mut u8 = std::ptr::null_mut();
                let mut frames_avail: u32 = 0;
                let mut flags: u32 = 0;
                let r = cap_client.GetBuffer(
                    &mut buffer_ptr,
                    &mut frames_avail,
                    &mut flags,
                    None,
                    None,
                );
                if let Err(e) = r {
                    // AUDCLNT_S_BUFFER_EMPTY is informational, not an error.
                    if e.code() == windows::Win32::Media::Audio::AUDCLNT_S_BUFFER_EMPTY {
                        tracing::trace!("GetBuffer: AUDCLNT_S_BUFFER_EMPTY");
                        break;
                    }
                    return Err(AudioError::wasapi("GetBuffer", e));
                }
                tracing::trace!(frames_avail, flags, "GetBuffer returned");
                if frames_avail == 0 {
                    let _ = cap_client.ReleaseBuffer(0);
                    break;
                }

                if !got_first_buffer {
                    tracing::info!(
                        frames = frames_avail,
                        elapsed_ms = start_time.elapsed().as_millis() as u64,
                        "first capture buffer received — audio is flowing"
                    );
                    got_first_buffer = true;
                }

                if flags
                    & (AUDCLNT_BUFFERFLAGS_DATA_DISCONTINUITY.0
                        | AUDCLNT_BUFFERFLAGS_TIMESTAMP_ERROR.0
                        | AUDCLNT_BUFFERFLAGS_SILENT.0) as u32
                    != 0
                {
                    sink.on_glitch(flags);
                }

                // The buffer matches the device's mix format, which we
                // validated as 32-bit float above, so `sample_count` f32s is
                // exactly what the engine allocated.
                let sample_count = frames_avail as usize * device_channels;
                let raw: &[f32] = if flags & AUDCLNT_BUFFERFLAGS_SILENT.0 as u32 != 0 {
                    // When the engine flags a buffer silent the contents are
                    // undefined — it just didn't bother zeroing the mapping.
                    // Feeding it downstream would push whatever happens to be
                    // in that memory out to the render endpoint, so substitute
                    // real silence.
                    silence.resize(sample_count, 0.0);
                    &silence[..sample_count]
                } else {
                    std::slice::from_raw_parts(buffer_ptr as *const f32, sample_count)
                };

                let mono_48k: &[f32] = match converter.as_mut() {
                    None => raw,
                    Some(c) => c.process(raw, frames_avail as usize),
                };

                accumulator.feed(mono_48k, |frame| sink.on_frame(frame));

                cap_client
                    .ReleaseBuffer(frames_avail)
                    .map_err(|e| AudioError::wasapi("ReleaseBuffer", e))?;
            }
        }

        let _ = client.Stop();
        Ok(())
    }
}

/// Public for `devices::DeviceList::enumerate()`.
pub(crate) fn enumerate_all() -> Result<DeviceList> {
    unsafe {
        let _ = CoInitializeEx(None, COINIT_MULTITHREADED);
        let enumerator: IMMDeviceEnumerator =
            CoCreateInstance(&CLSID_MMDeviceEnumerator, None, CLSCTX_ALL)
                .map_err(|e| AudioError::wasapi("CoCreateInstance(MMDeviceEnumerator)", e))?;

        Ok(DeviceList {
            capture: enumerate_direction(&enumerator, eCapture)?,
            render: enumerate_direction(&enumerator, eRender)?,
        })
    }
}

unsafe fn enumerate_direction(
    enumerator: &IMMDeviceEnumerator,
    flow: EDataFlow,
) -> Result<Vec<Device>> {
    let coll = enumerator
        .EnumAudioEndpoints(flow, DEVICE_STATE_ACTIVE)
        .map_err(|e| AudioError::wasapi("EnumAudioEndpoints", e))?;

    let default_id = enumerator
        .GetDefaultAudioEndpoint(flow, eCommunications)
        .ok()
        .and_then(|d| d.GetId().ok())
        .map(|p| p.to_string().unwrap_or_default())
        .unwrap_or_default();

    let count = coll
        .GetCount()
        .map_err(|e| AudioError::wasapi("GetCount", e))?;
    let mut out = Vec::with_capacity(count as usize);
    for i in 0..count {
        let dev = coll.Item(i).map_err(|e| AudioError::wasapi("Item", e))?;
        let id = dev
            .GetId()
            .map_err(|e| AudioError::wasapi("GetId", e))?
            .to_string()
            .unwrap_or_default();
        let friendly_name = read_friendly_name(&dev).unwrap_or_else(|_| id.clone());
        out.push(Device {
            is_default: id == default_id,
            id,
            friendly_name,
            direction: match flow {
                EDataFlow(0) => DeviceDirection::Render,
                _ => DeviceDirection::Capture,
            },
        });
    }
    Ok(out)
}

unsafe fn read_friendly_name(dev: &IMMDevice) -> Result<String> {
    let store = dev
        .OpenPropertyStore(STGM_READ)
        .map_err(|e| AudioError::wasapi("OpenPropertyStore", e))?;
    // windows 0.58: GetValue returns the PROPVARIANT directly.
    let prop = store
        .GetValue(&PKEY_Device_FriendlyName)
        .map_err(|e| AudioError::wasapi("GetValue(FriendlyName)", e))?;
    // PROPVARIANT for a string is VT_LPWSTR; Display impl decodes it.
    Ok(prop.to_string())
}

unsafe fn find_device(
    enumerator: &IMMDeviceEnumerator,
    id: &str,
    flow: EDataFlow,
) -> Result<IMMDevice> {
    if id.is_empty() || id == "default" {
        return enumerator
            .GetDefaultAudioEndpoint(flow, eCommunications)
            .map_err(|e| AudioError::wasapi("GetDefaultAudioEndpoint", e));
    }
    let wide: Vec<u16> = id.encode_utf16().chain(std::iter::once(0)).collect();
    enumerator
        .GetDevice(PCWSTR::from_raw(wide.as_ptr()))
        .map_err(|e| AudioError::wasapi("GetDevice", e))
}

/// Accumulates an arbitrary-length mono f32 stream into fixed 480-sample
/// frames. Holds at most one partial frame across calls.
pub(crate) struct FrameAccumulator {
    buf: Vec<f32>,
}

impl FrameAccumulator {
    pub fn new() -> Self {
        Self {
            buf: Vec::with_capacity(FRAME_SAMPLES * 2),
        }
    }

    pub fn feed(&mut self, samples: &[f32], mut emit: impl FnMut(&Frame)) {
        let mut i = 0;
        while i < samples.len() {
            let need = FRAME_SAMPLES - self.buf.len();
            let take = need.min(samples.len() - i);
            self.buf.extend_from_slice(&samples[i..i + take]);
            i += take;
            if self.buf.len() == FRAME_SAMPLES {
                let mut frame = [0f32; FRAME_SAMPLES];
                frame.copy_from_slice(&self.buf);
                self.buf.clear();
                emit(&frame);
            }
        }
    }
}

/// Cheap inline downmix + linear resampler. Good enough for voice; replace
/// with `rubato` if you ever want to ship music-grade quality.
pub(crate) struct InlineConverter {
    src_rate: u32,
    src_channels: usize,
    dst_rate: u32,
    last_sample: f32,
    /// Fractional position into the source stream; advanced by src/dst per
    /// output sample.
    phase: f64,
    out: Vec<f32>,
}

impl InlineConverter {
    pub fn new(src_rate: u32, src_channels: usize, dst_rate: u32) -> Self {
        Self {
            src_rate,
            src_channels,
            dst_rate,
            last_sample: 0.0,
            phase: 0.0,
            out: Vec::with_capacity(2048),
        }
    }

    pub fn process(&mut self, interleaved: &[f32], frames: usize) -> &[f32] {
        // Step 1: downmix to mono in a scratch buffer of length `frames`.
        let mut mono = Vec::with_capacity(frames);
        if self.src_channels == 1 {
            mono.extend_from_slice(&interleaved[..frames]);
        } else {
            for f in 0..frames {
                let base = f * self.src_channels;
                let mut acc = 0.0f32;
                for c in 0..self.src_channels {
                    acc += interleaved[base + c];
                }
                mono.push(acc / self.src_channels as f32);
            }
        }

        // Step 2: linear-resample mono → dst_rate.
        self.out.clear();
        if self.src_rate == self.dst_rate {
            self.out.extend_from_slice(&mono);
            self.last_sample = *mono.last().unwrap_or(&self.last_sample);
            return &self.out;
        }

        let ratio = self.src_rate as f64 / self.dst_rate as f64;
        let total_src = mono.len() as f64;
        while self.phase < total_src {
            let idx = self.phase as usize;
            let frac = self.phase - idx as f64;
            let a = if idx == 0 {
                self.last_sample
            } else {
                mono[idx - 1]
            };
            let b = mono.get(idx).copied().unwrap_or(self.last_sample);
            self.out
                .push((a as f64 + (b as f64 - a as f64) * frac) as f32);
            self.phase += ratio;
        }
        self.phase -= total_src;
        if let Some(&s) = mono.last() {
            self.last_sample = s;
        }
        &self.out
    }
}

#[cfg(test)]
mod accumulator_tests {
    use super::*;

    /// Collect whatever the accumulator emits, so a test can look at the
    /// frames rather than at a callback.
    fn feed(acc: &mut FrameAccumulator, samples: &[f32]) -> Vec<Frame> {
        let mut out = Vec::new();
        acc.feed(samples, |f| out.push(*f));
        out
    }

    #[test]
    fn a_full_frame_comes_straight_back_out() {
        let mut acc = FrameAccumulator::new();
        let input: Vec<f32> = (0..FRAME_SAMPLES).map(|i| i as f32).collect();

        let frames = feed(&mut acc, &input);

        assert_eq!(frames.len(), 1);
        assert_eq!(frames[0].as_slice(), input.as_slice());
    }

    /// The engine hands us whatever it has — usually not a multiple of 480.
    /// A partial frame has to be held and completed by the next buffer, in
    /// order and with nothing dropped or repeated.
    #[test]
    fn a_partial_frame_is_carried_across_calls() {
        let mut acc = FrameAccumulator::new();
        let input: Vec<f32> = (0..FRAME_SAMPLES).map(|i| i as f32).collect();

        assert!(feed(&mut acc, &input[..100]).is_empty(), "not a frame yet");
        assert!(feed(&mut acc, &input[100..300]).is_empty());
        let frames = feed(&mut acc, &input[300..]);

        assert_eq!(frames.len(), 1, "the third chunk completes the frame");
        assert_eq!(
            frames[0].as_slice(),
            input.as_slice(),
            "samples were reordered or lost across the joins"
        );
    }

    /// A big buffer must produce every whole frame in it and keep the rest.
    #[test]
    fn a_long_buffer_yields_every_whole_frame_and_holds_the_remainder() {
        let mut acc = FrameAccumulator::new();
        let total = FRAME_SAMPLES * 3 + 40;
        let input: Vec<f32> = (0..total).map(|i| i as f32).collect();

        let frames = feed(&mut acc, &input);
        assert_eq!(frames.len(), 3);
        for (n, frame) in frames.iter().enumerate() {
            let start = n * FRAME_SAMPLES;
            assert_eq!(
                frame.as_slice(),
                &input[start..start + FRAME_SAMPLES],
                "frame {n} has the wrong samples"
            );
        }

        // The 40 left over emerge once the next buffer tops them up.
        let more: Vec<f32> = vec![-1.0; FRAME_SAMPLES - 40];
        let frames = feed(&mut acc, &more);
        assert_eq!(frames.len(), 1);
        assert_eq!(&frames[0][..40], &input[total - 40..]);
        assert_eq!(frames[0][40], -1.0);
    }

    #[test]
    fn an_empty_buffer_emits_nothing() {
        let mut acc = FrameAccumulator::new();
        assert!(feed(&mut acc, &[]).is_empty());
    }

    /// Nothing may be emitted that was not fed in: over a long run the frame
    /// count has to track the sample count exactly.
    #[test]
    fn no_samples_are_invented_or_lost_over_a_long_run() {
        let mut acc = FrameAccumulator::new();
        let mut emitted = 0usize;
        let mut fed = 0usize;
        // Chunk sizes that never line up with 480.
        for chunk in [7usize, 113, 480, 1021, 33, 962] {
            let block = vec![0.5f32; chunk];
            emitted += feed(&mut acc, &block).len();
            fed += chunk;
        }
        assert_eq!(emitted, fed / FRAME_SAMPLES);
    }
}

#[cfg(test)]
mod converter_tests {
    use super::*;

    /// Peak amplitude, for checking a conversion didn't swallow the signal.
    fn peak(x: &[f32]) -> f32 {
        x.iter().fold(0.0f32, |a, s| a.max(s.abs()))
    }

    #[test]
    fn a_matching_mono_device_passes_through_untouched() {
        let mut c = InlineConverter::new(SAMPLE_RATE, 1, SAMPLE_RATE);
        let input: Vec<f32> = (0..480).map(|i| (i as f32 / 480.0) - 0.5).collect();
        assert_eq!(c.process(&input, 480), input.as_slice());
    }

    /// Stereo mics are the common case, and a downmix that took only the left
    /// channel would sound fine right up until someone's mic is on the right.
    #[test]
    fn stereo_is_averaged_rather_than_half_discarded() {
        let mut c = InlineConverter::new(SAMPLE_RATE, 2, SAMPLE_RATE);
        // Interleaved L,R: left silent, right at full scale.
        let input = [0.0, 1.0, 0.0, 1.0, 0.0, 1.0];
        assert_eq!(c.process(&input, 3), &[0.5, 0.5, 0.5]);

        let mut c = InlineConverter::new(SAMPLE_RATE, 2, SAMPLE_RATE);
        let input = [0.4, 0.6, -0.2, 0.2];
        let out = c.process(&input, 2);
        assert!(
            (out[0] - 0.5).abs() < 1e-6 && out[1].abs() < 1e-6,
            "{out:?}"
        );
    }

    #[test]
    fn a_four_channel_device_is_averaged_too() {
        let mut c = InlineConverter::new(SAMPLE_RATE, 4, SAMPLE_RATE);
        let input = [1.0, 1.0, 1.0, 1.0, 0.0, 0.0, 0.0, 2.0];
        assert_eq!(c.process(&input, 2), &[1.0, 0.5]);
    }

    /// 44.1 kHz mics are still everywhere. The output must come out at 48 k,
    /// which means slightly *more* samples than went in.
    #[test]
    fn forty_four_one_is_resampled_up_to_forty_eight() {
        let mut c = InlineConverter::new(44_100, 1, 48_000);
        let input = vec![0.25f32; 441];
        let out = c.process(&input, 441);

        let expected = (441.0f64 * 48_000.0 / 44_100.0).round() as usize;
        assert!(
            out.len().abs_diff(expected) <= 1,
            "got {} samples, expected about {expected}",
            out.len()
        );
        // A constant in must stay that constant, not be attenuated by the
        // interpolation.
        assert!((peak(out) - 0.25).abs() < 1e-3, "peak was {}", peak(out));
    }

    /// The one that matters: the fractional read position carries across
    /// calls. If it resets each time, the stream gains or loses a fraction of
    /// a sample every 10 ms, which is an audible click every few seconds.
    /// The one that matters: the fractional read position carries across
    /// calls. If it resets each time, the stream gains a fraction of a sample
    /// every buffer, which is an audible click every few seconds.
    ///
    /// The chunk size is deliberately 400 rather than 441. 441 samples at
    /// 44.1 kHz is exactly 480 at 48 kHz, so a converter that threw its phase
    /// away every call would still emit exactly 480 and look correct. 400 is
    /// not commensurate, so only genuine continuity gives the right total.
    #[test]
    fn the_resampler_does_not_drift_across_calls() {
        const CHUNK: usize = 400;
        const CALLS: usize = 100;

        let mut c = InlineConverter::new(44_100, 1, 48_000);
        let chunk = vec![0.1f32; CHUNK];

        let mut produced = 0usize;
        for _ in 0..CALLS {
            produced += c.process(&chunk, CHUNK).len();
        }

        let expected = (CHUNK as f64 * CALLS as f64 * 48_000.0 / 44_100.0).round() as usize;
        assert!(
            produced.abs_diff(expected) <= 2,
            "produced {produced} samples, expected about {expected} — the phase is drifting \
             (a converter that resets phase every call would produce {})",
            CALLS * (CHUNK as f64 / (44_100.0 / 48_000.0)).ceil() as usize
        );
    }

    #[test]
    fn downmix_and_resample_compose() {
        let mut c = InlineConverter::new(44_100, 2, 48_000);
        let input: Vec<f32> = std::iter::repeat_n([0.0f32, 0.8], 441).flatten().collect();
        let out = c.process(&input, 441);
        assert!(out.len() >= 470, "expected upsampling, got {}", out.len());
        assert!((peak(out) - 0.4).abs() < 1e-3, "peak was {}", peak(out));
    }

    /// A device running faster than 48 k gets decimated, and must not alias
    /// its way into a longer buffer.
    #[test]
    fn a_ninety_six_kilohertz_device_is_halved() {
        let mut c = InlineConverter::new(96_000, 1, 48_000);
        let input: Vec<f32> = (0..960).map(|i| (i % 2) as f32).collect();
        let out = c.process(&input, 960);
        assert!(
            out.len().abs_diff(480) <= 1,
            "expected about 480 samples, got {}",
            out.len()
        );
    }
}
