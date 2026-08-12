//! Pipeline glue: capture → ring A → DSP thread → ring B → render.
//!
//! Two SPSC ring buffers connect the three audio threads. We size the rings
//! at 8 frames (~80 ms) — enough headroom to absorb a scheduler hiccup,
//! small enough that we don't hide actual problems.

use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::Arc;

use anyhow::{Context, Result};
use ringbuf::traits::{Consumer, Producer, Split};
use ringbuf::HeapRb;
use tracing::{info, warn};

use audio_io::{devices::DeviceList, wasapi_capture::Frame};
use dsp::{DenoiserHost, Stats};

use crate::config::Config;
use crate::parking_lot_compat::RwLock;

const RING_FRAMES: usize = 8;

pub struct Pipeline {
    /// Held to keep the audio threads alive. Dropping these stops them.
    #[allow(dead_code)]
    capture: audio_io::WasapiCapture,
    #[allow(dead_code)]
    render: audio_io::WasapiRender,
    #[allow(dead_code)]
    dsp_thread: Option<std::thread::JoinHandle<()>>,

    bypass: Arc<AtomicBool>,
    stats: Arc<Stats>,
    denoiser_name: &'static str,
    /// Used to ask the DSP thread to exit cleanly when we drop.
    shutdown: Arc<AtomicBool>,
}

impl Pipeline {
    pub fn start(cfg: Arc<RwLock<Config>>) -> Result<Self> {
        let snapshot = cfg.read().unwrap().clone();

        // Resolve devices.
        let devices = DeviceList::enumerate().context("enumerating audio devices")?;
        // Config names devices; ids are an internal detail resolved here,
        // fresh on every start and every restart.
        let input = devices
            .resolve_capture(&snapshot.input_device)
            .ok_or_else(|| {
                if snapshot.input_device.is_empty() {
                    anyhow::anyhow!("no microphone is available")
                } else {
                    anyhow::anyhow!(
                        "microphone {:?} is not available (unplugged?). \
                         Pick one from the tray menu",
                        snapshot.input_device
                    )
                }
            })?
            .clone();

        // Installing a virtual cable usually makes it the default *capture*
        // device too. Left alone, "follow the Windows default" then means
        // recording from the same cable we render into — the cable feeding
        // itself, with no real microphone anywhere in the loop.
        if let Some(product) = input.virtual_cable_output() {
            anyhow::bail!(
                "the selected microphone is {product}'s own output, which is where NoiseGate \
                 sends audio — routing it back in would loop. Pick a real microphone from the \
                 tray menu"
            );
        }

        // Auto-detection failing must not fall back to the default render
        // device: that would play the microphone out of whatever speakers,
        // Bluetooth headset or meeting-room HDMI display happens to be
        // default. Fail closed and let the user fix the routing.
        let output = devices
            .resolve_render(&snapshot.output_device)
            .with_context(|| {
                format!(
                    "no virtual audio cable is installed (looked for {}). Cleaned audio needs \
                     one so other apps can hear it as a microphone. Install VB-Cable, or name \
                     one in output_device",
                    audio_io::devices::known_cable_products().join(", ")
                )
            })?
            .clone();
        let (input_id, output_id) = (input.id.clone(), output.id.clone());
        info!(mic = %input.friendly_name, to = %output.friendly_name, "using devices");

        // Build the rings.
        let (prod_a, mut cons_a) = HeapRb::<Frame>::new(RING_FRAMES).split();
        let (mut prod_b, cons_b) = HeapRb::<Frame>::new(RING_FRAMES).split();

        // DSP setup.
        let model_path = snapshot.active_model();
        let denoiser = dsp::build_denoiser(model_path.as_deref(), snapshot.attenuation_db)
            .context("loading denoiser")?;
        let denoiser_name = denoiser.name();
        let (mut host, bypass, stats) = DenoiserHost::new(denoiser);
        bypass.store(!snapshot.enabled, Ordering::Relaxed);

        let shutdown = Arc::new(AtomicBool::new(false));
        let shutdown_dsp = shutdown.clone();
        let stats_for_thread = stats.clone();

        // DSP thread: pulls from ring A, processes, pushes to ring B.
        let dsp_thread = std::thread::Builder::new()
            .name("noisegate-dsp".into())
            .spawn(move || {
                #[cfg(windows)]
                let _mmcss = audio_io::mmcss_pro_audio_for_current_thread();
                let _ = stats_for_thread; // keep alive via host
                // An empty ring is not news: this thread polls every 2 ms
                // while frames arrive every 10 ms, so it finds nothing most
                // of the time even when everything is healthy. Only a long
                // gap means anything, and that is what gets reported.
                let mut last_frame_at = std::time::Instant::now();
                let mut reported_gap = false;
                while !shutdown_dsp.load(Ordering::Acquire) {
                    let mut frame = match cons_a.try_pop() {
                        Some(f) => f,
                        None => {
                            if !reported_gap
                                && last_frame_at.elapsed() > std::time::Duration::from_secs(2)
                            {
                                warn!(
                                    gap_ms = last_frame_at.elapsed().as_millis() as u64,
                                    "no audio from the microphone — it may have been unplugged, \
                                     muted, or blocked in Windows privacy settings"
                                );
                                reported_gap = true;
                            }
                            std::thread::sleep(std::time::Duration::from_millis(2));
                            continue;
                        }
                    };
                    if reported_gap {
                        info!("microphone audio resumed");
                        reported_gap = false;
                    }
                    last_frame_at = std::time::Instant::now();
                    if let Err(e) = host.process(&mut frame) {
                        warn!(error = %e, "denoiser error; passing frame through");
                    }
                    if prod_b.try_push(frame).is_err() {
                        // Render is behind — drop. Audible as a click; better
                        // than blocking the DSP thread.
                        warn!("ring B full; dropping a frame");
                    }
                }
            })
            .context("spawn dsp thread")?;

        // Capture sink: pushes into ring A.
        struct Sink<P: Producer<Item = Frame> + Send> {
            prod: P,
        }
        impl<P: Producer<Item = Frame> + Send> audio_io::wasapi_capture::FrameSink for Sink<P> {
            fn on_frame(&mut self, frame: &Frame) {
                if self.prod.try_push(*frame).is_err() {
                    // DSP behind — overwrite oldest by popping (we don't
                    // have direct access here; the simplest cheap option is
                    // to just drop. Audible as a tiny click; far better
                    // than blocking the audio engine.)
                    tracing::warn!("ring A full; dropping captured frame");
                }
            }
            fn on_glitch(&mut self, flags: u32) {
                tracing::warn!(flags, "capture glitch reported by audio engine");
            }
        }

        let capture = audio_io::WasapiCapture::start(
            &input_id,
            Box::new(Sink { prod: prod_a }),
        )
        .map_err(|e| anyhow::anyhow!(e))?;

        // Render source: pulls from ring B.
        struct Source<C: Consumer<Item = Frame> + Send> {
            cons: C,
            last_underrun_log: std::time::Instant,
        }
        impl<C: Consumer<Item = Frame> + Send> audio_io::wasapi_render::FrameSource for Source<C> {
            fn next_frame(&mut self) -> Option<Frame> {
                self.cons.try_pop()
            }
            fn on_underrun(&mut self) {
                if self.last_underrun_log.elapsed() > std::time::Duration::from_secs(5) {
                    tracing::warn!("render underrun (cleaned audio not arriving from DSP)");
                    self.last_underrun_log = std::time::Instant::now();
                }
            }
        }

        let render = audio_io::WasapiRender::start(
            &output_id,
            Box::new(Source {
                cons: cons_b,
                last_underrun_log: std::time::Instant::now()
                    - std::time::Duration::from_secs(60),
            }),
        )
        .map_err(|e| anyhow::anyhow!(e))?;

        Ok(Self {
            capture,
            render,
            dsp_thread: Some(dsp_thread),
            bypass,
            stats,
            denoiser_name,
            shutdown,
        })
    }

    pub fn set_enabled(&self, enabled: bool) {
        self.bypass.store(!enabled, Ordering::Relaxed);
    }

    pub fn is_enabled(&self) -> bool {
        !self.bypass.load(Ordering::Relaxed)
    }

    pub fn stats(&self) -> &Stats {
        &self.stats
    }

    pub fn denoiser_name(&self) -> &'static str {
        self.denoiser_name
    }

    /// Frames processed so far. The tray watches this: a counter that stops
    /// advancing means the capture stream died, which WASAPI reports by
    /// simply never signalling its event again. Silence still produces
    /// frames, so this distinguishes "quiet room" from "dead device".
    pub fn frames_processed(&self) -> u64 {
        self.stats.frames.load(Ordering::Relaxed)
    }
}

impl Drop for Pipeline {
    fn drop(&mut self) {
        self.shutdown.store(true, Ordering::Release);
        if let Some(t) = self.dsp_thread.take() {
            let _ = t.join();
        }
    }
}
