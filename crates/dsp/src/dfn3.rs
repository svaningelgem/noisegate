//! DeepFilterNet3 from upstream's own three-graph export.
//!
//! Upstream publishes DFN3 as `enc` / `erb_dec` / `df_dec` rather than one
//! graph, because the STFT front-end is meant to live in the host. That makes
//! this module the counterpart of [`crate::dfn_frontend`]: the front-end
//! computes the features, these graphs produce a mask and a set of filter
//! coefficients, and the work of *applying* them is here.
//!
//! The alternative — [`crate::onnx`] — runs a single-file streaming graph that
//! does all of this internally. That file is a third party's repackaging and
//! cannot be regenerated; this path can, from the published checkpoint, with
//! `scripts/export_dfn3.py`. See `docs/model-pipeline.md`.
//!
//! Per 480-sample hop:
//!
//! 1. `analysis` -> 481-bin spectrum, pushed onto two rolling histories
//! 2. `feat_erb` / `feat_spec` -> `enc` -> `erb_dec` (gains) and `df_dec` (taps)
//! 3. expand the 32 ERB gains back over the 481 bins and multiply
//! 4. replace the lowest 96 bins with a complex FIR across `DF_ORDER` frames
//!    of *noisy* history
//! 5. `synthesis` -> 480 samples
//!
//! Output lags input by `DF_LOOKAHEAD` hops; the model estimates frame *t*
//! having seen two frames beyond it.

use std::path::Path;

use ort::session::Session;
use ort::value::TensorRef;
use realfft::num_complex::Complex32;

use crate::dfn_frontend::{erb_widths, DfFrontend, FREQ_BINS, NB_DF, NB_ERB};
use crate::{Denoiser, DspError, Result, FRAME_SAMPLES};

/// Filter taps per frequency bin, and how far ahead the model looks. Both are
/// fixed by the exported graphs (`df_order` / `df_lookahead` in config.ini).
const DF_ORDER: usize = 5;
const DF_LOOKAHEAD: usize = 2;
/// Enough history to serve the deep filter for a frame we emit `DF_LOOKAHEAD`
/// hops late.
const HISTORY: usize = DF_ORDER + DF_LOOKAHEAD;

pub struct Dfn3Denoiser {
    enc: Session,
    erb_dec: Session,
    df_dec: Session,
    front: DfFrontend,

    /// Spectra we will emit, mask and filter applied in place.
    enhanced: Vec<[Complex32; FREQ_BINS]>,
    /// The same spectra untouched. The deep filter reads *noisy* history: it
    /// is estimating the clean signal from the observations, so feeding it
    /// already-masked frames would apply the gains twice.
    noisy: Vec<[Complex32; FREQ_BINS]>,

    erb_bands: Vec<usize>,
    feat_erb: [f32; NB_ERB],
    feat_spec: [Complex32; NB_DF],
    /// Frames seen so far, to suppress output until the history is primed.
    primed: usize,
}

impl Dfn3Denoiser {
    /// Load the three graphs from a directory produced by
    /// `scripts/export_dfn3.py`.
    pub fn load(dir: impl AsRef<Path>) -> Result<Self> {
        let dir = dir.as_ref();
        crate::onnx::pin_dylib_path();

        let open = |name: &str| -> Result<Session> {
            let path = dir.join(name);
            Session::builder()
                .map_err(|e| DspError::Load(format!("ort builder: {e}")))?
                .with_intra_threads(1)
                .map_err(|e| DspError::Load(format!("with_intra_threads: {e}")))?
                .commit_from_file(&path)
                .map_err(|e| DspError::Load(format!("loading {}: {e}", path.display())))
        };

        Ok(Self {
            enc: open("enc.onnx")?,
            erb_dec: open("erb_dec.onnx")?,
            df_dec: open("df_dec.onnx")?,
            front: DfFrontend::new(),
            enhanced: vec![[Complex32::new(0.0, 0.0); FREQ_BINS]; HISTORY],
            noisy: vec![[Complex32::new(0.0, 0.0); FREQ_BINS]; HISTORY],
            erb_bands: erb_widths(),
            feat_erb: [0.0; NB_ERB],
            feat_spec: [Complex32::new(0.0, 0.0); NB_DF],
            primed: 0,
        })
    }

    /// Expand the 32 ERB gains back over the 481 bins and multiply. Mirrors
    /// libDF's `apply_interp_band_gain`: one gain per band, applied flat
    /// across the band's width.
    fn apply_erb_gains(&self, spec: &mut [Complex32; FREQ_BINS], gains: &[f32]) {
        let mut bin = 0;
        for (&width, &gain) in self.erb_bands.iter().zip(gains) {
            for slot in spec.iter_mut().skip(bin).take(width) {
                *slot *= gain;
            }
            bin += width;
        }
    }

    /// Replace the lowest `NB_DF` bins with a complex FIR across the noisy
    /// history. Tap 0 multiplies the *oldest* frame — libDF zips its rolling
    /// buffer front-to-back against the tap axis, and reversing that measurably
    /// degrades the result (0.982 -> 0.960 correlation against the reference).
    fn apply_deep_filter(&self, out: &mut [Complex32; FREQ_BINS], coefs: &[f32]) {
        for (bin, slot) in out.iter_mut().take(NB_DF).enumerate() {
            let mut acc = Complex32::new(0.0, 0.0);
            for tap in 0..DF_ORDER {
                // coefs is [NB_DF, DF_ORDER, 2] flattened: re, im per tap.
                let base = (bin * DF_ORDER + tap) * 2;
                let c = Complex32::new(coefs[base], coefs[base + 1]);
                // Oldest first, but over the newest DF_ORDER slots: the frame
                // being emitted sits at HISTORY-1-DF_LOOKAHEAD, and the taps
                // span from DF_LOOKAHEAD before it to DF_LOOKAHEAD after — the
                // look-ahead the model was given. Starting at slot 0 instead
                // shifts the whole filter two frames into the past.
                acc += self.noisy[HISTORY - DF_ORDER + tap][bin] * c;
            }
            *slot = acc;
        }
        // DC has no phase in a real signal, and the inverse FFT insists on it.
        // The filter is complex and happily puts something in the imaginary
        // part of bin 0; numpy's irfft discards that silently, realfft panics.
        // Discarding it explicitly matches the reference implementation.
        out[0].im = 0.0;
    }
}

impl Denoiser for Dfn3Denoiser {
    fn process_frame(&mut self, frame: &mut [f32; FRAME_SAMPLES]) -> Result<()> {
        // Slide both histories along and analyse the new hop into the newest
        // slot.
        self.enhanced.rotate_left(1);
        self.noisy.rotate_left(1);
        let newest = HISTORY - 1;
        let mut spec = [Complex32::new(0.0, 0.0); FREQ_BINS];
        self.front.analysis(frame, &mut spec);
        self.enhanced[newest] = spec;
        self.noisy[newest] = spec;

        self.front.feat_erb(&spec, &mut self.feat_erb);
        self.front.feat_spec(&spec, &mut self.feat_spec);

        // enc expects [1, 1, 1, NB_ERB] and [1, 2, 1, NB_DF] — one time step.
        let erb_in = TensorRef::from_array_view(([1_i64, 1, 1, NB_ERB as i64], &self.feat_erb[..]))
            .map_err(|e| DspError::Inference(format!("feat_erb: {e}")))?;

        let mut spec_planes = [0f32; 2 * NB_DF];
        for (i, c) in self.feat_spec.iter().enumerate() {
            spec_planes[i] = c.re;
            spec_planes[NB_DF + i] = c.im;
        }
        let spec_in = TensorRef::from_array_view(([1_i64, 2, 1, NB_DF as i64], &spec_planes[..]))
            .map_err(|e| DspError::Inference(format!("feat_spec: {e}")))?;

        // Each run's `SessionOutputs` borrows its session mutably for as long
        // as it lives, so copy what we need out and let it go before touching
        // `self` again.
        let (e0, e1, e2, e3, emb, c0) = {
            let enc_out = self
                .enc
                .run(ort::inputs!["feat_erb" => erb_in, "feat_spec" => spec_in])
                .map_err(|e| DspError::Inference(format!("enc: {e}")))?;
            let take = |name: &str| -> Result<Vec<f32>> {
                let (_, v) = enc_out[name]
                    .try_extract_tensor::<f32>()
                    .map_err(|e| DspError::Inference(format!("enc output {name}: {e}")))?;
                Ok(v.to_vec())
            };
            (
                take("e0")?,
                take("e1")?,
                take("e2")?,
                take("e3")?,
                take("emb")?,
                take("c0")?,
            )
        };

        // The encoder's skip connections keep 64 channels; only the last
        // dimension differs per stage, so derive it from the length.
        let dim = |v: &[f32]| [1_i64, 64, 1, (v.len() / 64) as i64];
        let gains = {
            let erb_m = self
                .erb_dec
                .run(ort::inputs![
                    "emb" => TensorRef::from_array_view(([1_i64, 1, emb.len() as i64], &emb[..]))
                        .map_err(|e| DspError::Inference(format!("emb: {e}")))?,
                    "e3" => TensorRef::from_array_view((dim(&e3), &e3[..]))
                        .map_err(|e| DspError::Inference(format!("e3: {e}")))?,
                    "e2" => TensorRef::from_array_view((dim(&e2), &e2[..]))
                        .map_err(|e| DspError::Inference(format!("e2: {e}")))?,
                    "e1" => TensorRef::from_array_view((dim(&e1), &e1[..]))
                        .map_err(|e| DspError::Inference(format!("e1: {e}")))?,
                    "e0" => TensorRef::from_array_view((dim(&e0), &e0[..]))
                        .map_err(|e| DspError::Inference(format!("e0: {e}")))?,
                ])
                .map_err(|e| DspError::Inference(format!("erb_dec: {e}")))?;
            let (_, g) = erb_m[0]
                .try_extract_tensor::<f32>()
                .map_err(|e| DspError::Inference(format!("erb_dec mask: {e}")))?;
            g.to_vec()
        };

        let coefs = {
            let df_out = self
                .df_dec
                .run(ort::inputs![
                    "emb" => TensorRef::from_array_view(([1_i64, 1, emb.len() as i64], &emb[..]))
                        .map_err(|e| DspError::Inference(format!("emb: {e}")))?,
                    "c0" => TensorRef::from_array_view(([1_i64, 64, 1, NB_DF as i64], &c0[..]))
                        .map_err(|e| DspError::Inference(format!("c0: {e}")))?,
                ])
                .map_err(|e| DspError::Inference(format!("df_dec: {e}")))?;
            let (_, c) = df_out[0]
                .try_extract_tensor::<f32>()
                .map_err(|e| DspError::Inference(format!("df_dec coefs: {e}")))?;
            c.to_vec()
        };

        // The frame we emit is the one the model has now seen past.
        let emit = HISTORY - 1 - DF_LOOKAHEAD;
        let mut target = self.enhanced[emit];
        self.apply_erb_gains(&mut target, &gains);
        if std::env::var_os("NOISEGATE_NO_DF").is_none() {
            self.apply_deep_filter(&mut target, &coefs);
        }

        self.front.synthesis(&mut target, frame);

        // Until the history is full the emitted frame is partly zeros, which
        // would be an audible burst of nothing at start-up. Stay silent for
        // those first few hops instead.
        self.primed += 1;
        if self.primed <= DF_LOOKAHEAD {
            frame.fill(0.0);
        }
        Ok(())
    }

    fn name(&self) -> &'static str {
        "DeepFilterNet3 (3-graph)"
    }
}
