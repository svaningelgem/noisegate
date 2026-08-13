//! WASAPI shared low-latency render to the VB-Cable Input endpoint.
//!
//! Pulls 480-sample mono f32 frames from a `FrameSource` and writes them
//! to the chosen render device. If the device's mix format isn't mono
//! 48 kHz f32 we up-mix and (linear) resample inline, mirroring the
//! capture path.

use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::Arc;

use windows::core::PCWSTR;
use windows::Win32::Foundation::WAIT_OBJECT_0;
use windows::Win32::Media::Audio::*;
use windows::Win32::System::Com::*;
use windows::Win32::System::Threading::WaitForSingleObject;

use crate::error::{AudioError, Result};
use crate::wasapi_capture::Frame;
use crate::{FRAME_SAMPLES, SAMPLE_RATE};

#[allow(non_upper_case_globals)]
const CLSID_MMDeviceEnumerator: windows::core::GUID =
    windows::core::GUID::from_u128(0xBCDE0395_E52F_467C_8E3D_C4579291692E);

/// Source of frames for the render thread. Must be lock-free / wait-free
/// since it's polled from the audio engine's tick. Returning `None`
/// renders silence for one period.
pub trait FrameSource: Send {
    fn next_frame(&mut self) -> Option<Frame>;
    fn on_underrun(&mut self) {}
}

pub struct WasapiRender {
    stop: Arc<AtomicBool>,
    thread: Option<std::thread::JoinHandle<()>>,
}

impl WasapiRender {
    pub fn start(device_id: &str, mut source: Box<dyn FrameSource>) -> Result<Self> {
        let stop = Arc::new(AtomicBool::new(false));
        let stop_thread = stop.clone();
        let device_id = device_id.to_string();
        let (ready_tx, ready_rx) = std::sync::mpsc::channel::<Result<()>>();

        let thread = std::thread::Builder::new()
            .name("noisegate-render".into())
            .spawn(move || {
                if let Err(e) = render_loop(&device_id, &mut *source, &stop_thread, &ready_tx) {
                    tracing::error!(error = %e, "render loop exited with error");
                    let _ = ready_tx.send(Err(e));
                }
            })
            .map_err(|e| AudioError::Other(anyhow::anyhow!("spawn render thread: {e}")))?;

        ready_rx.recv().map_err(|_| AudioError::ThreadDied)??;
        Ok(Self {
            stop,
            thread: Some(thread),
        })
    }
}

impl Drop for WasapiRender {
    fn drop(&mut self) {
        self.stop.store(true, Ordering::Release);
        if let Some(t) = self.thread.take() {
            let _ = t.join();
        }
    }
}

fn render_loop(
    device_id: &str,
    source: &mut dyn FrameSource,
    stop: &AtomicBool,
    ready_tx: &std::sync::mpsc::Sender<Result<()>>,
) -> Result<()> {
    unsafe {
        let _ = CoInitializeEx(None, COINIT_MULTITHREADED);

        let enumerator: IMMDeviceEnumerator =
            CoCreateInstance(&CLSID_MMDeviceEnumerator, None, CLSCTX_ALL)
                .map_err(|e| AudioError::wasapi("CoCreateInstance(MMDeviceEnumerator)", e))?;

        let device = if device_id.is_empty() {
            enumerator
                .GetDefaultAudioEndpoint(eRender, eCommunications)
                .map_err(|e| AudioError::wasapi("GetDefaultAudioEndpoint", e))?
        } else {
            let wide: Vec<u16> = device_id.encode_utf16().chain(std::iter::once(0)).collect();
            enumerator
                .GetDevice(PCWSTR::from_raw(wide.as_ptr()))
                .map_err(|e| AudioError::wasapi("GetDevice", e))?
        };

        let client: IAudioClient3 = device
            .Activate(CLSCTX_ALL, None)
            .map_err(|e| AudioError::wasapi("IMMDevice::Activate", e))?;

        // GetMixFormat may return WAVEFORMATEXTENSIBLE; pass the original
        // pointer through to Initialize unchanged. Snapshot the fields we
        // need into Copy locals before freeing.
        let mix_ptr = client
            .GetMixFormat()
            .map_err(|e| AudioError::wasapi("GetMixFormat", e))?;
        let mix = crate::format::read_mix_format(mix_ptr);
        let device_rate = mix.sample_rate;
        let device_channels = mix.channels as usize;
        let device_block_align = mix.block_align as u32;

        tracing::info!(
            device_rate,
            device_channels,
            device_bits = mix.bits_per_sample,
            device_is_float = mix.is_float,
            "render device format negotiated"
        );

        // UpConverter writes `frames * channels` f32s into a buffer the engine
        // sized at `frames * nBlockAlign` bytes. Those only agree when the
        // device mixes 32-bit float — on a 16-bit device we would write twice
        // the buffer. Refuse the device rather than overrun it.
        if let Err(e) = mix.validate() {
            windows::Win32::System::Com::CoTaskMemFree(Some(mix_ptr as _));
            return Err(e);
        }

        let init_res = client.Initialize(
            AUDCLNT_SHAREMODE_SHARED,
            AUDCLNT_STREAMFLAGS_EVENTCALLBACK,
            0,
            0,
            mix_ptr,
            None,
        );
        windows::Win32::System::Com::CoTaskMemFree(Some(mix_ptr as _));
        init_res.map_err(|e| AudioError::wasapi("IAudioClient::Initialize", e))?;

        // Closes itself when this function returns, however it returns.
        let event = crate::event::Event::new()?;
        client
            .SetEventHandle(event.handle())
            .map_err(|e| AudioError::wasapi("SetEventHandle", e))?;

        let render_client: IAudioRenderClient = client
            .GetService()
            .map_err(|e| AudioError::wasapi("GetService(IAudioRenderClient)", e))?;

        let buffer_frames = client
            .GetBufferSize()
            .map_err(|e| AudioError::wasapi("GetBufferSize", e))?;

        let _mmcss = crate::mmcss::ProAudio::set_for_current_thread();

        // Pre-fill with silence so the engine doesn't underrun on the first tick.
        let prefill = render_client
            .GetBuffer(buffer_frames)
            .map_err(|e| AudioError::wasapi("GetBuffer(prefill)", e))?;
        std::ptr::write_bytes(prefill, 0, (buffer_frames * device_block_align) as usize);
        render_client
            .ReleaseBuffer(buffer_frames, AUDCLNT_BUFFERFLAGS_SILENT.0 as u32)
            .map_err(|e| AudioError::wasapi("ReleaseBuffer(prefill)", e))?;

        client.Start().map_err(|e| AudioError::wasapi("Start", e))?;
        let _ = ready_tx.send(Ok(()));

        let mut engine = WasapiRenderEngine {
            client: client.clone(),
            render_client,
            event: event.handle(),
            buffer_frames,
        };
        let result = pump(&mut engine, device_rate, device_channels, source, stop);

        let _ = client.Stop();
        result
    }
}

/// The audio engine as the render pump uses it.
///
/// The mirror of [`crate::wasapi_capture::CaptureEngine`], and there for the
/// same reason: an engine that stops accepting frames, one that goes away
/// mid-stream, and a source that cannot keep up are all sequences that can be
/// written down here rather than reproduced on hardware.
pub(crate) trait RenderEngine {
    /// Wait for the engine's buffer-ready event. `false` on timeout.
    fn wait(&mut self, timeout: std::time::Duration) -> bool;

    /// How many frames the engine can take right now.
    fn writable_frames(&mut self) -> Result<u32>;

    /// Hand `fill` a buffer for `frames` frames, then release it back.
    ///
    /// # Safety
    /// `fill` receives a pointer valid for `frames * channels` f32s.
    unsafe fn with_buffer(
        &mut self,
        frames: u32,
        fill: &mut dyn FnMut(*mut f32) -> usize,
    ) -> Result<()>;
}

const WAIT_TIMEOUT: std::time::Duration = std::time::Duration::from_millis(200);

/// Feed the engine from `source` until asked to stop.
pub(crate) fn pump(
    engine: &mut dyn RenderEngine,
    device_rate: u32,
    device_channels: usize,
    source: &mut dyn FrameSource,
    stop: &AtomicBool,
) -> Result<()> {
    let mut upconverter = UpConverter::new(SAMPLE_RATE, device_rate, device_channels);
    let mut pending: Vec<f32> = Vec::with_capacity(FRAME_SAMPLES * 2);

    while !stop.load(Ordering::Acquire) {
        if !engine.wait(WAIT_TIMEOUT) {
            continue;
        }

        let frames_writable = engine.writable_frames()?;
        if frames_writable == 0 {
            continue;
        }

        // Pull mono frames until we have enough pre-converted samples to
        // fill `frames_writable` frames at the device rate.
        let needed_src = source_samples_needed(frames_writable, device_rate);
        while pending.len() < needed_src {
            match source.next_frame() {
                Some(f) => pending.extend_from_slice(&f),
                None => {
                    source.on_underrun();
                    // Pad with silence so we don't busy-loop or stall.
                    pending.extend(std::iter::repeat_n(0.0, FRAME_SAMPLES));
                }
            }
        }

        let available = needed_src.min(pending.len());
        let mut consumed = 0usize;
        unsafe {
            engine.with_buffer(frames_writable, &mut |buf| {
                consumed =
                    upconverter.write_into(&pending[..available], buf, frames_writable as usize);
                consumed
            })?;
        }
        // Drop the source samples we used.
        pending.drain(..consumed);
    }
    Ok(())
}

struct WasapiRenderEngine {
    client: IAudioClient3,
    render_client: IAudioRenderClient,
    event: windows::Win32::Foundation::HANDLE,
    buffer_frames: u32,
}

impl RenderEngine for WasapiRenderEngine {
    fn wait(&mut self, timeout: std::time::Duration) -> bool {
        unsafe { WaitForSingleObject(self.event, timeout.as_millis() as u32) == WAIT_OBJECT_0 }
    }

    fn writable_frames(&mut self) -> Result<u32> {
        let padding = unsafe { self.client.GetCurrentPadding() }
            .map_err(|e| AudioError::wasapi("GetCurrentPadding", e))?;
        Ok(self.buffer_frames.saturating_sub(padding))
    }

    unsafe fn with_buffer(
        &mut self,
        frames: u32,
        fill: &mut dyn FnMut(*mut f32) -> usize,
    ) -> Result<()> {
        let buf = self
            .render_client
            .GetBuffer(frames)
            .map_err(|e| AudioError::wasapi("GetBuffer", e))?;
        fill(buf as *mut f32);
        self.render_client
            .ReleaseBuffer(frames, 0)
            .map_err(|e| AudioError::wasapi("ReleaseBuffer", e))
    }
}

/// How many mono 48 kHz samples are needed to fill `frames` frames at the
/// device's rate.
///
/// Rounds up deliberately. Coming up short leaves the tail of the engine's
/// buffer unwritten, which the device plays as whatever was there before;
/// one spare sample stays in `pending` and is used by the next buffer.
fn source_samples_needed(frames: u32, device_rate: u32) -> usize {
    (frames as u64 * SAMPLE_RATE as u64).div_ceil(device_rate as u64) as usize
}

/// Mono 48 kHz → multi-channel device-rate f32 interleaved.
struct UpConverter {
    src_rate: u32,
    dst_rate: u32,
    dst_channels: usize,
    phase: f64,
    last: f32,
}

impl UpConverter {
    fn new(src_rate: u32, dst_rate: u32, dst_channels: usize) -> Self {
        Self {
            src_rate,
            dst_rate,
            dst_channels,
            phase: 0.0,
            last: 0.0,
        }
    }

    /// Linearly resample mono `src` (48 kHz) into `frames` device-rate
    /// frames written to `dst` (interleaved, dst_channels). Returns the
    /// number of source samples consumed so the caller can advance.
    unsafe fn write_into(&mut self, src: &[f32], dst: *mut f32, frames: usize) -> usize {
        if src.is_empty() {
            std::ptr::write_bytes(dst, 0, frames * self.dst_channels);
            return 0;
        }
        let ratio = self.src_rate as f64 / self.dst_rate as f64;
        for f in 0..frames {
            let pos = self.phase + f as f64 * ratio;
            let idx = pos as usize;
            let frac = pos - idx as f64;
            let a = if idx == 0 {
                self.last
            } else {
                src.get(idx - 1).copied().unwrap_or(self.last)
            };
            let b = src.get(idx).copied().unwrap_or(self.last);
            let s = (a as f64 + (b as f64 - a as f64) * frac) as f32;
            for c in 0..self.dst_channels {
                *dst.add(f * self.dst_channels + c) = s;
            }
        }
        // Advance phase past the samples we used; keep the fractional part
        // so we don't drift across calls.
        let advance = frames as f64 * ratio + self.phase;
        let consumed = advance as usize;
        self.phase = advance - consumed as f64;
        if consumed > 0 {
            self.last = src[(consumed - 1).min(src.len() - 1)];
        }
        consumed.min(src.len())
    }
}

#[cfg(test)]
mod pump_tests {
    use super::*;

    enum Tick {
        /// The engine can take this many frames.
        Writable(u32),
        /// The event did not fire.
        Timeout,
        /// The engine is full — nothing to do this tick.
        Full,
        /// The endpoint went away underneath us.
        Invalidated,
    }

    struct ScriptedEngine {
        ticks: std::collections::VecDeque<Tick>,
        exhausted: Arc<AtomicBool>,
        channels: usize,
        /// Everything written, in order, so a test can look at the audio.
        written: Vec<f32>,
        buffers_taken: usize,
    }

    impl RenderEngine for ScriptedEngine {
        fn wait(&mut self, _timeout: std::time::Duration) -> bool {
            match self.ticks.front() {
                None => {
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

        fn writable_frames(&mut self) -> Result<u32> {
            match self.ticks.pop_front() {
                Some(Tick::Writable(n)) => Ok(n),
                Some(Tick::Full) => Ok(0),
                Some(Tick::Invalidated) => Err(AudioError::DeviceInvalidated {
                    context: "GetCurrentPadding",
                }),
                _ => Ok(0),
            }
        }

        unsafe fn with_buffer(
            &mut self,
            frames: u32,
            fill: &mut dyn FnMut(*mut f32) -> usize,
        ) -> Result<()> {
            // NaN-filled so an unwritten tail is visible rather than plausible.
            let mut buf = vec![f32::NAN; frames as usize * self.channels];
            fill(buf.as_mut_ptr());
            assert!(
                !buf.iter().any(|s| s.is_nan()),
                "the engine buffer was left partly unwritten — the device would \
                 play whatever was in that memory"
            );
            self.written.extend_from_slice(&buf);
            self.buffers_taken += 1;
            Ok(())
        }
    }

    /// A source that yields a fixed number of frames and then runs dry.
    struct Source {
        remaining: usize,
        value: f32,
        underruns: usize,
    }

    impl FrameSource for Source {
        fn next_frame(&mut self) -> Option<Frame> {
            if self.remaining == 0 {
                return None;
            }
            self.remaining -= 1;
            Some([self.value; FRAME_SAMPLES])
        }
        fn on_underrun(&mut self) {
            self.underruns += 1;
        }
    }

    fn run(
        ticks: Vec<Tick>,
        device_rate: u32,
        channels: usize,
        frames_available: usize,
    ) -> (ScriptedEngine, Source, Result<()>) {
        let stop = Arc::new(AtomicBool::new(false));
        let mut engine = ScriptedEngine {
            ticks: ticks.into(),
            exhausted: stop.clone(),
            channels,
            written: Vec::new(),
            buffers_taken: 0,
        };
        let mut source = Source {
            remaining: frames_available,
            value: 0.5,
            underruns: 0,
        };
        let result = pump(&mut engine, device_rate, channels, &mut source, &stop);
        (engine, source, result)
    }

    #[test]
    fn audio_reaches_the_engine() {
        let (engine, _, result) = run(vec![Tick::Writable(480)], SAMPLE_RATE, 2, 4);
        assert!(result.is_ok());
        assert_eq!(engine.buffers_taken, 1);
        assert_eq!(engine.written.len(), 480 * 2, "480 frames x 2 channels");
    }

    /// The failure that matters here: if the pump consumed fewer source
    /// samples than it played, `pending` would grow every tick until the
    /// process ran out of memory — and the audio would drift further behind
    /// real time the whole while.
    #[test]
    fn the_pending_queue_does_not_grow_without_bound() {
        // 200 ticks of 441 device frames at 44.1 kHz, with a source that
        // always has audio: sustained operation, not a burst.
        let ticks: Vec<Tick> = (0..200).map(|_| Tick::Writable(441)).collect();
        let (engine, source, result) = run(ticks, 44_100, 2, 10_000);
        assert!(result.is_ok());
        assert_eq!(engine.buffers_taken, 200);
        assert_eq!(source.underruns, 0, "the source never ran dry");

        // 200 buffers x 10 ms = 2 s of audio, so about 2 s of source consumed.
        // Frames pulled = 10_000 - remaining.
        let pulled = 10_000 - source.remaining;
        let expected = 200; // 200 x 10 ms, one 480-sample frame each
        assert!(
            pulled.abs_diff(expected) <= 2,
            "pulled {pulled} frames to play {expected} — the queue is drifting"
        );
    }

    /// A source that runs dry must produce silence rather than stalling the
    /// device or busy-looping.
    #[test]
    fn an_empty_source_renders_silence_and_says_so() {
        let (engine, source, _) = run(vec![Tick::Writable(480)], SAMPLE_RATE, 1, 0);
        assert!(source.underruns > 0, "the underrun must be reported");
        assert!(
            engine.written.iter().all(|&s| s == 0.0),
            "expected silence, got {}",
            engine.written[0]
        );
    }

    /// A full engine is the normal case when the buffer is still draining.
    #[test]
    fn a_full_engine_is_skipped_without_taking_a_buffer() {
        let (engine, source, result) = run(
            vec![Tick::Full, Tick::Full, Tick::Writable(480)],
            SAMPLE_RATE,
            1,
            4,
        );
        assert!(result.is_ok());
        assert_eq!(engine.buffers_taken, 1, "took a buffer it could not fill");
        assert_eq!(source.underruns, 0);
    }

    #[test]
    fn timeouts_do_not_end_the_stream() {
        let (engine, _, result) = run(
            vec![Tick::Timeout, Tick::Timeout, Tick::Writable(480)],
            SAMPLE_RATE,
            1,
            4,
        );
        assert!(result.is_ok());
        assert_eq!(engine.buffers_taken, 1);
    }

    #[test]
    fn a_device_that_disappears_surfaces_as_recoverable() {
        let (_, _, result) = run(
            vec![Tick::Writable(480), Tick::Invalidated],
            SAMPLE_RATE,
            1,
            8,
        );
        let err = result.expect_err("must not be swallowed");
        assert!(err.is_recoverable(), "{err} should be retried");
    }

    /// Every frame of the engine's buffer has to be written, whatever the
    /// rate conversion — the NaN check in the fake enforces it.
    #[test]
    fn the_whole_buffer_is_written_at_every_rate() {
        for (rate, frames) in [(44_100u32, 441u32), (48_000, 480), (96_000, 960)] {
            let (engine, _, result) = run(vec![Tick::Writable(frames)], rate, 2, 8);
            assert!(result.is_ok(), "{rate} Hz failed");
            assert_eq!(engine.written.len(), frames as usize * 2, "{rate} Hz");
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// `write_into` fills a raw engine buffer. Wrap it so tests can look at
    /// what landed there: returns (interleaved output, source samples used).
    fn render(uc: &mut UpConverter, src: &[f32], frames: usize) -> (Vec<f32>, usize) {
        let mut buf = vec![f32::NAN; frames * uc.dst_channels];
        let consumed = unsafe { uc.write_into(src, buf.as_mut_ptr(), frames) };
        assert!(
            !buf.iter().any(|s| s.is_nan()),
            "left part of the engine buffer uninitialised — the engine would \
             play whatever was in that memory"
        );
        (buf, consumed)
    }

    /// Every channel of the device gets the same mono sample. Writing only
    /// channel 0 would come out of one side of a headset.
    #[test]
    fn mono_is_written_to_every_channel() {
        let mut uc = UpConverter::new(SAMPLE_RATE, SAMPLE_RATE, 2);
        let src = [0.5f32; 8];
        let (out, _) = render(&mut uc, &src, 4);

        assert_eq!(out.len(), 8, "4 frames x 2 channels");
        for pair in out.chunks(2) {
            assert_eq!(pair[0], pair[1], "channels disagree: {pair:?}");
        }
    }

    #[test]
    fn a_surround_device_gets_all_six_channels_filled() {
        let mut uc = UpConverter::new(SAMPLE_RATE, SAMPLE_RATE, 6);
        let (out, _) = render(&mut uc, &[0.25f32; 4], 2);
        assert_eq!(out.len(), 12);
        for frame in out.chunks(6) {
            assert!(
                frame.iter().all(|&s| s == frame[0]),
                "channels within a frame disagree: {frame:?}"
            );
        }
    }

    /// Nothing to play must be silence, not the previous contents of the
    /// engine's buffer.
    #[test]
    fn an_empty_source_writes_real_silence() {
        let mut uc = UpConverter::new(SAMPLE_RATE, SAMPLE_RATE, 2);
        let (out, consumed) = render(&mut uc, &[], 4);
        assert_eq!(consumed, 0);
        assert!(out.iter().all(|&s| s == 0.0), "{out:?}");
    }

    /// At a matching rate the converter is a one-sample delay, and it must
    /// consume exactly what it emits or the pending buffer grows without
    /// bound.
    #[test]
    fn a_matching_rate_consumes_exactly_what_it_plays() {
        let mut uc = UpConverter::new(SAMPLE_RATE, SAMPLE_RATE, 1);
        let src: Vec<f32> = (1..=8).map(|i| i as f32).collect();
        let (out, consumed) = render(&mut uc, &src, 8);

        assert_eq!(consumed, 8, "consumed {consumed} to play 8 frames");
        // One sample of delay: the first output is the previous call's tail
        // (silence at start-up), then the input follows.
        assert_eq!(out[0], 0.0);
        assert_eq!(&out[1..], &src[..7]);
    }

    /// The device rate is whatever the mix format says; 44.1 kHz endpoints are
    /// common. Playing 44.1 k frames needs fewer 48 k samples than frames.
    #[test]
    fn a_slower_device_consumes_more_source_than_it_plays() {
        let mut uc = UpConverter::new(SAMPLE_RATE, 44_100, 2);
        let src = vec![0.3f32; 1000];
        let (_, consumed) = render(&mut uc, &src, 441);

        let expected = (441.0f64 * 48_000.0 / 44_100.0).round() as usize;
        assert!(
            consumed.abs_diff(expected) <= 1,
            "consumed {consumed} for 441 device frames, expected about {expected}"
        );
    }

    /// The render counterpart of the capture drift test: phase has to carry
    /// across calls, or the pending queue slowly fills or starves.
    #[test]
    fn consumption_stays_in_step_over_many_buffers() {
        const FRAMES: usize = 400;
        const CALLS: usize = 100;

        let mut uc = UpConverter::new(SAMPLE_RATE, 44_100, 2);
        let src = vec![0.2f32; 2048];

        let mut consumed = 0usize;
        for _ in 0..CALLS {
            consumed += render(&mut uc, &src, FRAMES).1;
        }

        let expected = (FRAMES as f64 * CALLS as f64 * 48_000.0 / 44_100.0).round() as usize;
        assert!(
            consumed.abs_diff(expected) <= 2,
            "consumed {consumed} over {CALLS} buffers, expected about {expected} — \
             the pending queue would drift by {} samples a second",
            (consumed as i64 - expected as i64).abs() * 48_000 / (FRAMES * CALLS) as i64
        );
    }

    #[test]
    fn a_matching_device_needs_one_source_sample_per_frame() {
        assert_eq!(source_samples_needed(480, SAMPLE_RATE), 480);
        assert_eq!(source_samples_needed(0, SAMPLE_RATE), 0);
    }

    #[test]
    fn a_slower_device_needs_more_source_than_frames() {
        // 441 frames at 44.1 kHz is 10 ms, which is 480 samples at 48 kHz.
        assert_eq!(source_samples_needed(441, 44_100), 480);
        // 96 kHz is the other way round: two device frames per source sample.
        assert_eq!(source_samples_needed(960, 96_000), 480);
    }

    /// Rounding down would leave the tail of the engine's buffer unwritten,
    /// and the device plays whatever was there before.
    #[test]
    fn the_sample_count_always_rounds_up() {
        // 1 frame at 44.1 kHz is 1.088 samples at 48 kHz.
        assert_eq!(source_samples_needed(1, 44_100), 2);
        assert_eq!(source_samples_needed(100, 44_100), 109); // 108.84
    }

    /// A whole engine buffer at once must not overflow the intermediate
    /// arithmetic — the calculation is done in u64 for exactly this reason.
    #[test]
    fn a_huge_buffer_does_not_overflow() {
        let needed = source_samples_needed(u32::MAX, 44_100);
        assert!(
            needed > u32::MAX as usize,
            "u32 arithmetic would have wrapped"
        );
    }

    /// A constant signal must come out at the same level, whatever the rate.
    #[test]
    fn resampling_does_not_change_the_level() {
        for rate in [44_100, 48_000, 96_000] {
            let mut uc = UpConverter::new(SAMPLE_RATE, rate, 2);
            let src = vec![0.75f32; 2048];
            // Second call, so the interpolation is past its start-up sample.
            render(&mut uc, &src, 200);
            let (out, _) = render(&mut uc, &src, 200);
            let peak = out.iter().fold(0.0f32, |a, s| a.max(s.abs()));
            assert!((peak - 0.75).abs() < 1e-3, "{rate} Hz gave peak {peak}");
        }
    }
}
