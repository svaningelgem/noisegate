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
use windows::Win32::System::Threading::WaitForSingleObject;

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
            .name("roommute-capture".into())
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
    // COM init for this thread. STA isn't required for WASAPI; MTA is simpler
    // and matches the audio engine's threading model.
    let _ = unsafe { CoInitializeEx(None, COINIT_MULTITHREADED) };

    let enumerator: IMMDeviceEnumerator =
        unsafe { CoCreateInstance(&CLSID_MMDeviceEnumerator, None, CLSCTX_ALL) }
            .map_err(|e| AudioError::wasapi("CoCreateInstance(MMDeviceEnumerator)", e))?;

    let device = find_device(&enumerator, device_id, eCapture)?;

    let client: IAudioClient3 = unsafe { device.Activate(CLSCTX_ALL, None) }
        .map_err(|e| AudioError::wasapi("IMMDevice::Activate", e))?;

    // Scoped so the engine's format allocation is freed as soon as Initialize
    // has consumed it, rather than lingering for the life of the stream.
    let (device_rate, device_channels, needs_convert) = {
        // GetMixFormat returns a CoTaskMem allocation we own. On most devices
        // it is really a WAVEFORMATEXTENSIBLE (40 bytes, cbSize > 0), and the
        // original pointer MUST go back to Initialize unchanged — copying it
        // into an 18-byte WAVEFORMATEX truncates the extensible header and the
        // engine rejects it with E_INVALIDARG.
        let mix_ptr =
            unsafe { client.GetMixFormat() }.map_err(|e| AudioError::wasapi("GetMixFormat", e))?;
        let format = unsafe { crate::format::EngineMixFormat::from_engine(mix_ptr) };
        let mix = format.decode();
        let needs_convert = !(mix.sample_rate == SAMPLE_RATE && mix.channels == 1);

        // Always log the chosen format — useful for diagnosing "init OK but no
        // data", which is usually privacy permissions or a wrong device pick.
        tracing::info!(
            device_rate = mix.sample_rate,
            device_channels = mix.channels,
            device_bits = mix.bits_per_sample,
            device_is_float = mix.is_float,
            needs_convert,
            "capture device format negotiated"
        );

        // The pump reinterprets the engine's byte buffer as `[f32]`, so bail
        // out before Initialize if this device doesn't actually mix 32-bit
        // float. The allocation is freed by the guard either way.
        mix.validate()?;

        // Legacy Initialize — accepts the device's native (possibly
        // extensible) mix format reliably. Buffer duration 0 = engine default
        // (~30 ms), fine for voice.
        unsafe {
            client.Initialize(
                AUDCLNT_SHAREMODE_SHARED,
                AUDCLNT_STREAMFLAGS_EVENTCALLBACK,
                0, // hnsBufferDuration
                0, // hnsPeriodicity (must be 0 in shared mode)
                format.as_ptr(),
                None,
            )
        }
        .map_err(|e| AudioError::wasapi("IAudioClient::Initialize", e))?;

        (mix.sample_rate, mix.channels as usize, needs_convert)
    };

    // Closes itself when this function returns, however it returns.
    let event = crate::event::Event::new()?;
    unsafe { client.SetEventHandle(event.handle()) }
        .map_err(|e| AudioError::wasapi("SetEventHandle", e))?;

    let cap_client: IAudioCaptureClient = unsafe { client.GetService() }
        .map_err(|e| AudioError::wasapi("GetService(IAudioCaptureClient)", e))?;

    // MMCSS: ask the scheduler to treat this as Pro Audio. Without it we get
    // random scheduling delays under load and audible glitches.
    let _mmcss = crate::mmcss::ProAudio::set_for_current_thread();

    unsafe { client.Start() }.map_err(|e| AudioError::wasapi("IAudioClient::Start", e))?;

    // Init succeeded — unblock the caller.
    let _ = ready_tx.send(Ok(()));

    let mut engine = WasapiEngine {
        client: cap_client,
        event: event.handle(),
        channels: device_channels,
    };
    let result = pump(
        &mut engine,
        device_channels,
        needs_convert.then_some(device_rate),
        sink,
        stop,
    );

    let _ = unsafe { client.Stop() };
    result
}

/// The audio engine as the capture pump uses it.
///
/// This exists so the pump can be driven by a script. Every failure this
/// module has actually met in the wild is a sequence of these calls: a device
/// invalidated mid-stream, a buffer flagged silent, an engine that signals and
/// then has nothing, an event that never fires at all. None of them can be
/// reproduced on demand against real hardware, and all of them can be written
/// down here — so when the next one turns up it can be encoded rather than
/// described in a comment.
pub(crate) trait CaptureEngine {
    /// Wait for the engine's buffer-ready event. `false` on timeout.
    fn wait(&mut self, timeout: std::time::Duration) -> bool;

    /// Hand the next available buffer to `consume` as (interleaved samples,
    /// flags), then release it back to the engine. `false` means the engine
    /// has nothing more for this tick.
    ///
    /// Pairing GetBuffer with ReleaseBuffer is the implementation's job: doing
    /// it here rather than in the pump means an early return in the caller
    /// cannot leak a buffer and wedge the stream.
    fn next_buffer(&mut self, consume: &mut dyn FnMut(&[f32], u32)) -> Result<bool>;
}

/// How long to block on the buffer-ready event before looking at `stop`.
const WAIT_TIMEOUT: std::time::Duration = std::time::Duration::from_millis(200);

/// Drain the engine into 480-sample mono frames until asked to stop.
///
/// `resample_from` is the device's rate when it needs converting, `None` when
/// the device already produces mono 48 kHz.
pub(crate) fn pump(
    engine: &mut dyn CaptureEngine,
    channels: usize,
    resample_from: Option<u32>,
    sink: &mut dyn FrameSink,
    stop: &AtomicBool,
) -> Result<()> {
    let mut accumulator = FrameAccumulator::new();
    let mut converter = resample_from.map(|rate| InlineConverter::new(rate, channels, SAMPLE_RATE));

    let start_time = std::time::Instant::now();
    // Reused when the engine hands us a buffer flagged silent.
    let mut silence: Vec<f32> = Vec::new();
    let mut got_first_buffer = false;
    let mut last_silence_warn = std::time::Instant::now();

    while !stop.load(Ordering::Acquire) {
        if !engine.wait(WAIT_TIMEOUT) {
            // If we have never seen a buffer and a couple of seconds have
            // passed, this is almost certainly a Microphone-Privacy block in
            // Windows Settings rather than a quiet room. Say so every ~5s.
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

        // Drain everything the engine has for us this tick.
        while engine.next_buffer(&mut |raw, flags| {
            if !got_first_buffer {
                tracing::info!(
                    samples = raw.len(),
                    elapsed_ms = start_time.elapsed().as_millis() as u64,
                    "first capture buffer received — audio is flowing"
                );
                got_first_buffer = true;
            }

            let status = BufferStatus::from_flags(flags);
            if status.glitch {
                sink.on_glitch(flags);
            }

            let samples: &[f32] = if status.silent {
                // Undefined contents; substitute real silence rather than
                // forwarding whatever was in that mapping.
                silence.clear();
                silence.resize(raw.len(), 0.0);
                &silence
            } else {
                raw
            };

            let frames = samples.len() / channels.max(1);
            let mono_48k: &[f32] = match converter.as_mut() {
                None => samples,
                Some(c) => c.process(samples, frames),
            };
            accumulator.feed(mono_48k, |frame| sink.on_frame(frame));
        })? {}
    }
    Ok(())
}

/// The real engine: an `IAudioCaptureClient` and the event it signals.
struct WasapiEngine {
    client: IAudioCaptureClient,
    event: windows::Win32::Foundation::HANDLE,
    channels: usize,
}

impl CaptureEngine for WasapiEngine {
    fn wait(&mut self, timeout: std::time::Duration) -> bool {
        unsafe { WaitForSingleObject(self.event, timeout.as_millis() as u32) == WAIT_OBJECT_0 }
    }

    fn next_buffer(&mut self, consume: &mut dyn FnMut(&[f32], u32)) -> Result<bool> {
        unsafe {
            let mut buffer_ptr: *mut u8 = std::ptr::null_mut();
            let mut frames_avail: u32 = 0;
            let mut flags: u32 = 0;
            if let Err(e) =
                self.client
                    .GetBuffer(&mut buffer_ptr, &mut frames_avail, &mut flags, None, None)
            {
                // AUDCLNT_S_BUFFER_EMPTY is informational, not an error.
                if e.code() == windows::Win32::Media::Audio::AUDCLNT_S_BUFFER_EMPTY {
                    return Ok(false);
                }
                return Err(AudioError::wasapi("GetBuffer", e));
            }
            if frames_avail == 0 {
                let _ = self.client.ReleaseBuffer(0);
                return Ok(false);
            }

            // The buffer matches the device's mix format, which was validated
            // as 32-bit float before Initialize, so this many f32s is exactly
            // what the engine allocated.
            let samples = std::slice::from_raw_parts(
                buffer_ptr as *const f32,
                frames_avail as usize * self.channels,
            );
            consume(samples, flags);

            self.client
                .ReleaseBuffer(frames_avail)
                .map_err(|e| AudioError::wasapi("ReleaseBuffer", e))?;
            Ok(true)
        }
    }
}

/// Public for `devices::DeviceList::enumerate()`.
pub(crate) fn enumerate_all() -> Result<DeviceList> {
    let _ = unsafe { CoInitializeEx(None, COINIT_MULTITHREADED) };
    let enumerator: IMMDeviceEnumerator =
        unsafe { CoCreateInstance(&CLSID_MMDeviceEnumerator, None, CLSCTX_ALL) }
            .map_err(|e| AudioError::wasapi("CoCreateInstance(MMDeviceEnumerator)", e))?;

    Ok(DeviceList {
        capture: enumerate_direction(&enumerator, eCapture)?,
        render: enumerate_direction(&enumerator, eRender)?,
    })
}

/// These three take live COM objects and impose no extra contract on the
/// caller, so they are safe functions that use `unsafe` internally rather than
/// `unsafe fn`s that push the obligation outward. Only the FFI calls are
/// marked, which is what makes the marks worth reading.
fn enumerate_direction(enumerator: &IMMDeviceEnumerator, flow: EDataFlow) -> Result<Vec<Device>> {
    let coll = unsafe { enumerator.EnumAudioEndpoints(flow, DEVICE_STATE_ACTIVE) }
        .map_err(|e| AudioError::wasapi("EnumAudioEndpoints", e))?;

    let default_id = unsafe { enumerator.GetDefaultAudioEndpoint(flow, eCommunications) }
        .ok()
        .and_then(|d| unsafe { d.GetId() }.ok())
        .map(|p| unsafe { p.to_string() }.unwrap_or_default())
        .unwrap_or_default();

    let count = unsafe { coll.GetCount() }.map_err(|e| AudioError::wasapi("GetCount", e))?;
    let mut out = Vec::with_capacity(count as usize);
    for i in 0..count {
        let dev = unsafe { coll.Item(i) }.map_err(|e| AudioError::wasapi("Item", e))?;
        let id = unsafe {
            let raw = dev.GetId().map_err(|e| AudioError::wasapi("GetId", e))?;
            raw.to_string().unwrap_or_default()
        };
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

fn read_friendly_name(dev: &IMMDevice) -> Result<String> {
    let store = unsafe { dev.OpenPropertyStore(STGM_READ) }
        .map_err(|e| AudioError::wasapi("OpenPropertyStore", e))?;
    // windows 0.58: GetValue returns the PROPVARIANT directly.
    let prop = unsafe { store.GetValue(&PKEY_Device_FriendlyName) }
        .map_err(|e| AudioError::wasapi("GetValue(FriendlyName)", e))?;
    // PROPVARIANT for a string is VT_LPWSTR; Display impl decodes it.
    Ok(prop.to_string())
}

fn find_device(enumerator: &IMMDeviceEnumerator, id: &str, flow: EDataFlow) -> Result<IMMDevice> {
    if id.is_empty() || id == "default" {
        return unsafe { enumerator.GetDefaultAudioEndpoint(flow, eCommunications) }
            .map_err(|e| AudioError::wasapi("GetDefaultAudioEndpoint", e));
    }
    let wide: Vec<u16> = id.encode_utf16().chain(std::iter::once(0)).collect();
    unsafe { enumerator.GetDevice(PCWSTR::from_raw(wide.as_ptr())) }
        .map_err(|e| AudioError::wasapi("GetDevice", e))
}

/// What the engine's per-buffer flags mean for us.
#[derive(Debug, PartialEq, Eq)]
pub(crate) struct BufferStatus {
    /// The contents are undefined — the engine did not bother zeroing the
    /// mapping. Passing them downstream would push whatever happened to be in
    /// that memory out to the render endpoint, so they must be replaced with
    /// real silence rather than trusted.
    pub silent: bool,
    /// Worth telling the sink about: the stream is not intact here.
    pub glitch: bool,
}

impl BufferStatus {
    pub fn from_flags(flags: u32) -> Self {
        let has = |bit: u32| flags & bit != 0;
        let silent = has(AUDCLNT_BUFFERFLAGS_SILENT.0 as u32);
        Self {
            silent,
            glitch: silent
                || has(AUDCLNT_BUFFERFLAGS_DATA_DISCONTINUITY.0 as u32)
                || has(AUDCLNT_BUFFERFLAGS_TIMESTAMP_ERROR.0 as u32),
        }
    }
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
mod pump_tests {
    use super::*;

    /// One thing the engine does when the pump asks. A test is a list of
    /// these, which is the point of the whole abstraction: a bug report that
    /// says "my mic dies when I unplug the dock" becomes `Invalidated` in a
    /// sequence rather than a hardware reproduction someone has to own.
    enum Tick {
        /// The event fired and these buffers are available, each with flags.
        Buffers(Vec<(Vec<f32>, u32)>),
        /// The event did not fire within the timeout.
        Timeout,
        /// The engine signalled but has nothing — a real and common case.
        SignalledButEmpty,
        /// The endpoint went away underneath us.
        Invalidated,
    }

    struct ScriptedEngine {
        ticks: std::collections::VecDeque<Tick>,
        /// Set once the script runs out, so the pump is asked to stop.
        exhausted: Arc<AtomicBool>,
        released: usize,
    }

    impl ScriptedEngine {
        fn new(ticks: Vec<Tick>, stop: Arc<AtomicBool>) -> Self {
            Self {
                ticks: ticks.into(),
                exhausted: stop,
                released: 0,
            }
        }
    }

    impl CaptureEngine for ScriptedEngine {
        fn wait(&mut self, _timeout: std::time::Duration) -> bool {
            match self.ticks.front() {
                None => {
                    // Script over: let the pump's loop condition end it.
                    self.exhausted.store(true, Ordering::Release);
                    false
                }
                Some(Tick::Timeout) => {
                    self.ticks.pop_front();
                    false
                }
                Some(_) => true,
            }
        }

        fn next_buffer(&mut self, consume: &mut dyn FnMut(&[f32], u32)) -> Result<bool> {
            match self.ticks.front_mut() {
                Some(Tick::Buffers(queue)) if !queue.is_empty() => {
                    let (samples, flags) = queue.remove(0);
                    consume(&samples, flags);
                    self.released += 1;
                    Ok(true)
                }
                Some(Tick::Buffers(_)) | Some(Tick::SignalledButEmpty) => {
                    self.ticks.pop_front();
                    Ok(false)
                }
                Some(Tick::Invalidated) => Err(AudioError::DeviceInvalidated {
                    context: "GetBuffer",
                }),
                _ => Ok(false),
            }
        }
    }

    #[derive(Default)]
    struct Recorder {
        frames: Vec<Frame>,
        glitches: Vec<u32>,
    }

    impl FrameSink for Recorder {
        fn on_frame(&mut self, frame: &Frame) {
            self.frames.push(*frame);
        }
        fn on_glitch(&mut self, flags: u32) {
            self.glitches.push(flags);
        }
    }

    /// Run a script through the pump and report what the sink saw.
    fn run(
        ticks: Vec<Tick>,
        channels: usize,
        resample_from: Option<u32>,
    ) -> (Recorder, Result<()>) {
        let stop = Arc::new(AtomicBool::new(false));
        let mut engine = ScriptedEngine::new(ticks, stop.clone());
        let mut sink = Recorder::default();
        let result = pump(&mut engine, channels, resample_from, &mut sink, &stop);
        (sink, result)
    }

    fn buffer(samples: usize, value: f32) -> (Vec<f32>, u32) {
        (vec![value; samples], 0)
    }

    #[test]
    fn audio_flows_from_the_engine_into_whole_frames() {
        let (sink, result) = run(
            vec![Tick::Buffers(vec![
                buffer(FRAME_SAMPLES, 0.5),
                buffer(FRAME_SAMPLES, 0.25),
            ])],
            1,
            None,
        );
        assert!(result.is_ok());
        assert_eq!(sink.frames.len(), 2);
        assert!(sink.frames[0].iter().all(|&s| s == 0.5));
        assert!(sink.frames[1].iter().all(|&s| s == 0.25));
    }

    /// The engine hands over whatever it has, which is rarely a multiple of
    /// 480. Frames must still come out whole and in order.
    #[test]
    fn ragged_buffers_are_reassembled_into_frames() {
        let (sink, _) = run(
            vec![
                Tick::Buffers(vec![buffer(100, 0.1), buffer(200, 0.1)]),
                Tick::Buffers(vec![buffer(180, 0.1), buffer(700, 0.2)]),
            ],
            1,
            None,
        );
        // 1180 samples in = 2 whole frames, 220 held back.
        assert_eq!(sink.frames.len(), 2);
    }

    /// The regression this abstraction was built for. A silent-flagged buffer
    /// holds undefined memory; forwarding it would push whatever was in that
    /// mapping out to the render endpoint — someone else's audio, or worse.
    #[test]
    fn a_silent_flagged_buffer_never_reaches_the_sink() {
        const JUNK: f32 = 0.987;
        let silent_flag = AUDCLNT_BUFFERFLAGS_SILENT.0 as u32;

        let (sink, _) = run(
            vec![Tick::Buffers(vec![(
                vec![JUNK; FRAME_SAMPLES],
                silent_flag,
            )])],
            1,
            None,
        );

        assert_eq!(sink.frames.len(), 1);
        assert!(
            sink.frames[0].iter().all(|&s| s == 0.0),
            "undefined buffer contents were forwarded as audio: {}",
            sink.frames[0][0]
        );
        assert_eq!(sink.glitches, vec![silent_flag], "and it is reported");
    }

    /// A dropout is worth reporting, but the samples are real. Blanking them
    /// would turn every glitch into a hole in the audio.
    #[test]
    fn a_discontinuity_is_reported_but_the_audio_is_kept() {
        let flag = AUDCLNT_BUFFERFLAGS_DATA_DISCONTINUITY.0 as u32;
        let (sink, _) = run(
            vec![Tick::Buffers(vec![(vec![0.75; FRAME_SAMPLES], flag)])],
            1,
            None,
        );
        assert_eq!(sink.glitches, vec![flag]);
        assert!(
            sink.frames[0].iter().all(|&s| s == 0.75),
            "real audio was discarded along with the glitch report"
        );
    }

    /// An engine that signals and then has nothing is normal, not an error.
    #[test]
    fn signalling_with_nothing_available_is_not_a_failure() {
        let (sink, result) = run(
            vec![
                Tick::SignalledButEmpty,
                Tick::Buffers(vec![buffer(FRAME_SAMPLES, 0.3)]),
            ],
            1,
            None,
        );
        assert!(result.is_ok());
        assert_eq!(sink.frames.len(), 1, "the next tick must still be served");
    }

    /// Timeouts are how a blocked microphone presents: the event simply never
    /// fires. The pump must keep waiting rather than treating it as an error.
    #[test]
    fn timeouts_do_not_end_the_stream() {
        let (sink, result) = run(
            vec![
                Tick::Timeout,
                Tick::Timeout,
                Tick::Buffers(vec![buffer(FRAME_SAMPLES, 0.6)]),
            ],
            1,
            None,
        );
        assert!(result.is_ok());
        assert_eq!(sink.frames.len(), 1, "audio after the timeouts was lost");
    }

    /// The failure that started all of this: a device that disappears reports
    /// through the error channel, and must surface as the recoverable variant
    /// so the watchdog reopens it instead of giving up.
    #[test]
    fn a_device_that_disappears_surfaces_as_recoverable() {
        let (_, result) = run(
            vec![
                Tick::Buffers(vec![buffer(FRAME_SAMPLES, 0.1)]),
                Tick::Invalidated,
            ],
            1,
            None,
        );
        let err = result.expect_err("the pump must not swallow this");
        assert!(
            err.is_recoverable(),
            "{err} should be retried with a freshly resolved device"
        );
    }

    /// A stereo device at 44.1 kHz — the two conversions the pump wires up —
    /// still produces mono 48 kHz frames.
    #[test]
    fn a_stereo_forty_four_one_device_still_yields_mono_forty_eight() {
        // 441 stereo frames = 10 ms, which is 480 samples at 48 kHz.
        let interleaved: Vec<f32> = std::iter::repeat_n([0.0f32, 1.0], 441).flatten().collect();
        let (sink, _) = run(
            vec![
                Tick::Buffers(vec![(interleaved.clone(), 0)]),
                Tick::Buffers(vec![(interleaved, 0)]),
            ],
            2,
            Some(44_100),
        );
        assert_eq!(sink.frames.len(), 2, "expected 10 ms per buffer");
        // Downmixed: the average of 0.0 and 1.0. The resampler interpolates
        // against the previous source sample, which at start-up is silence,
        // so output begins with a short ramp-in — one output per source
        // sample not yet seen, here ceil(48000/44100) = 2. About 40 µs.
        const RAMP_IN: usize = 2;
        assert_eq!(sink.frames[0][0], 0.0, "expected to start from silence");
        assert!(
            sink.frames[0][..RAMP_IN].iter().all(|&s| s < 0.5),
            "the ramp should climb toward the signal: {:?}",
            &sink.frames[0][..RAMP_IN]
        );
        assert!(
            sink.frames[0][RAMP_IN..]
                .iter()
                .all(|&s| (s - 0.5).abs() < 1e-3),
            "should have settled by sample {RAMP_IN}: got {}",
            sink.frames[0][RAMP_IN]
        );
        assert!(
            sink.frames[1].iter().all(|&s| (s - 0.5).abs() < 1e-3),
            "steady state should be clean: {}",
            sink.frames[1][0]
        );
    }

    /// Stopping must be honoured promptly even mid-stream.
    #[test]
    fn the_stop_flag_ends_the_pump() {
        let stop = Arc::new(AtomicBool::new(true));
        let mut engine = ScriptedEngine::new(
            vec![Tick::Buffers(vec![buffer(FRAME_SAMPLES, 0.5)])],
            stop.clone(),
        );
        let mut sink = Recorder::default();
        pump(&mut engine, 1, None, &mut sink, &stop).unwrap();
        assert!(
            sink.frames.is_empty(),
            "already stopped; nothing should run"
        );
    }
}

#[cfg(test)]
mod buffer_status_tests {
    use super::*;

    #[test]
    fn a_clean_buffer_is_neither_silent_nor_a_glitch() {
        let s = BufferStatus::from_flags(0);
        assert!(!s.silent && !s.glitch);
    }

    /// The one that matters for what reaches the render endpoint: a
    /// silent-flagged buffer holds undefined memory, so it must be recognised
    /// and replaced rather than passed on.
    #[test]
    fn a_silent_flagged_buffer_is_recognised() {
        let s = BufferStatus::from_flags(AUDCLNT_BUFFERFLAGS_SILENT.0 as u32);
        assert!(s.silent, "undefined memory would be forwarded as audio");
        assert!(s.glitch, "and it is worth reporting");
    }

    /// A dropout is a glitch, but the samples we were handed are real — they
    /// must not be thrown away and replaced with silence.
    #[test]
    fn a_discontinuity_is_reported_without_discarding_the_audio() {
        for (name, flag) in [
            (
                "DATA_DISCONTINUITY",
                AUDCLNT_BUFFERFLAGS_DATA_DISCONTINUITY.0 as u32,
            ),
            (
                "TIMESTAMP_ERROR",
                AUDCLNT_BUFFERFLAGS_TIMESTAMP_ERROR.0 as u32,
            ),
        ] {
            let s = BufferStatus::from_flags(flag);
            assert!(s.glitch, "{name} should be reported");
            assert!(!s.silent, "{name} does not mean the data is undefined");
        }
    }

    #[test]
    fn flags_we_do_not_know_about_are_ignored() {
        let s = BufferStatus::from_flags(0x8000_0000);
        assert!(!s.silent && !s.glitch);
    }

    #[test]
    fn several_flags_at_once_are_all_honoured() {
        let s = BufferStatus::from_flags(
            AUDCLNT_BUFFERFLAGS_SILENT.0 as u32 | AUDCLNT_BUFFERFLAGS_DATA_DISCONTINUITY.0 as u32,
        );
        assert!(s.silent && s.glitch);
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
