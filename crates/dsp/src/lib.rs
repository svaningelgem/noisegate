//! DSP layer for RoomMute.
//!
//! Defines a [`Denoiser`] trait so the pipeline doesn't care which model is
//! running underneath. Three backends are compiled in, always: DeepFilterNet3
//! through tract, an ONNX Runtime loader for single-file streaming exports,
//! and RNNoise as the fallback that needs no model at all.

// `deny` rather than `forbid`, so that exactly one exception can exist and be
// argued for in place: the `Send` impl in `tract.rs`. Everything else in this
// crate is still rejected at compile time, and any new `unsafe` needs an
// explicit `#[allow]` that shows up in review.
#![deny(unsafe_code)]

use std::sync::atomic::{AtomicBool, AtomicU64, Ordering};
use std::sync::Arc;

pub const FRAME_SAMPLES: usize = 480;

#[derive(Debug, thiserror::Error)]
pub enum DspError {
    #[error("model load failed: {0}")]
    Load(String),
    #[error("inference failed: {0}")]
    Inference(String),
    #[error(transparent)]
    Other(#[from] anyhow::Error),
}

pub type Result<T> = std::result::Result<T, DspError>;

/// Generic denoiser interface. Implementations operate strictly on
/// 480-sample mono f32 frames @ 48 kHz, in-place, single-threaded.
pub trait Denoiser: Send {
    fn process_frame(&mut self, frame: &mut [f32; FRAME_SAMPLES]) -> Result<()>;
    fn name(&self) -> &'static str;
}

/// Bypass mode + latency stats wrapper. The pipeline thread holds this and
/// flips the bypass flag from the UI thread.
pub struct DenoiserHost {
    inner: Box<dyn Denoiser>,
    bypass: Arc<AtomicBool>,
    stats: Arc<Stats>,
}

#[derive(Default)]
pub struct Stats {
    /// Total frames processed (incl. bypassed).
    pub frames: AtomicU64,
    /// Cumulative DSP time in nanoseconds (excludes bypassed frames).
    pub dsp_ns: AtomicU64,
    /// Peak per-frame time in ns (sticky high-water mark, useful for
    /// surfacing glitch headroom in the tray tooltip).
    pub peak_frame_ns: AtomicU64,
}

impl DenoiserHost {
    pub fn new(inner: Box<dyn Denoiser>) -> (Self, Arc<AtomicBool>, Arc<Stats>) {
        let bypass = Arc::new(AtomicBool::new(false));
        let stats = Arc::new(Stats::default());
        let host = Self {
            inner,
            bypass: bypass.clone(),
            stats: stats.clone(),
        };
        (host, bypass, stats)
    }

    pub fn process(&mut self, frame: &mut [f32; FRAME_SAMPLES]) -> Result<()> {
        self.stats.frames.fetch_add(1, Ordering::Relaxed);
        if self.bypass.load(Ordering::Relaxed) {
            return Ok(());
        }
        let start = std::time::Instant::now();
        let r = self.inner.process_frame(frame);
        let elapsed = start.elapsed().as_nanos() as u64;
        self.stats.dsp_ns.fetch_add(elapsed, Ordering::Relaxed);
        // Atomic max via compare-exchange loop.
        let mut peak = self.stats.peak_frame_ns.load(Ordering::Relaxed);
        while elapsed > peak {
            match self.stats.peak_frame_ns.compare_exchange_weak(
                peak,
                elapsed,
                Ordering::Relaxed,
                Ordering::Relaxed,
            ) {
                Ok(_) => break,
                Err(actual) => peak = actual,
            }
        }
        r
    }

    pub fn name(&self) -> &'static str {
        self.inner.name()
    }
}

pub mod dfn_frontend;

mod onnx;
mod rnnoise;
mod tract;

pub use onnx::OnnxDenoiser;
pub use rnnoise::RnNoise;

/// Build a denoiser: the model at `model_path` if there is one, else the
/// built-in RNNoise.
///
/// Every backend is compiled in, always. They were once cargo features, which
/// only ever produced binaries that looked fine and quietly could not load a
/// model — `cargo build --release` left out DeepFilterNet3 because the feature
/// defaults lived in this crate and not in the app that depends on it.
pub fn build_denoiser(
    model_path: Option<&std::path::Path>,
    attenuation_db: f32,
) -> Result<Box<dyn Denoiser>> {
    if let Some(path) = model_path {
        // DeepFilterNet3's three-graph export, which we build ourselves and
        // which streams correctly because tract threads the model's state
        // across frames. Preferred over everything below.
        //
        // A directory is the normal case — that is how the installer lays the
        // model down, and it has no extension to match on. A `.tar.gz` is the
        // same model in the form upstream distributes it.
        if path.is_dir() || path.extension().is_some_and(|e| e == "gz") {
            return Ok(Box::new(tract::TractDenoiser::load(path, attenuation_db)?));
        }
        let mut d = OnnxDenoiser::load(path)?;
        // The config caps how much the model may suppress. RNNoise has no
        // equivalent knob, so this only applies to the ONNX path.
        d.set_attenuation_db(attenuation_db);
        return Ok(Box::new(d));
    }
    Ok(Box::new(RnNoise::new()?))
}

#[cfg(test)]
mod dispatch_tests {
    use std::path::PathBuf;

    use super::build_denoiser;

    fn shipped_model() -> PathBuf {
        PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../../models/dfn3")
    }

    /// The installer lays the model down as loose files in a directory, which
    /// has no extension to dispatch on. Routing that to the ONNX loader would
    /// fail at load, or worse, succeed against some other file and quietly run
    /// the wrong backend.
    #[test]
    fn a_model_directory_is_routed_to_tract() {
        let d = build_denoiser(Some(&shipped_model()), 0.0).expect("a model directory must load");
        assert_eq!(d.name(), "DeepFilterNet3 (tract)");
    }
}
