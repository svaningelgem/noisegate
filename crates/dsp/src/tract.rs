//! DeepFilterNet3 through upstream's own tract runner.
//!
//! This is the backend to prefer. It loads a `*_onnx.tar.gz` produced by
//! `scripts/export_dfn3.py` from DeepFilterNet's published checkpoint, so
//! every artefact in the chain is one we can rebuild. The alternative,
//! [`crate::onnx`], runs a single-file graph that a third party repackaged and
//! that nobody can regenerate.
//!
//! ## Why not just run the exported graphs under ONNX Runtime
//!
//! Because they carry no hidden state. DeepFilterNet3 is full of GRUs and
//! temporal convolutions, and upstream's export exposes none of that state as
//! graph inputs — it expects the host to feed whole sequences, or to use a
//! runtime that can stream them.
//!
//! Running them a frame at a time under ORT therefore resets the model's
//! entire memory every 10 ms. It still produces plausible audio, which is what
//! makes it dangerous: measured against `samples/kid_raw.wav`, whole-sequence
//! inference leaves the near voice at −0.1 dB while frame-at-a-time strips
//! −6.1 dB from it, with an identical front-end and identical post-processing.
//!
//! tract's `pulse` support solves exactly this: it rewrites the graph into a
//! streaming form and carries the state across calls. Upstream wrote that
//! integration, ships it under MIT/Apache-2.0, and uses it for their own
//! real-time tooling, so we use it rather than reimplement it.

use std::path::Path;

// The package is `deep_filter`; its library is named `df`.
use df::tract::{DfParams, DfTract, RuntimeParams};
use ndarray::{Array2, ArrayView2};

use crate::{Denoiser, DspError, Result, FRAME_SAMPLES};

/// Both stages, on every frame, always.
///
/// Upstream's runner varies the treatment per 10 ms frame from the model's own
/// SNR estimate: below `min_db_thresh` it zeroes the output, above
/// `max_db_erb_thresh` it passes audio through untouched, between them it runs
/// one stage or both. When the estimate sits near a boundary — which it does
/// constantly on speech with background chatter — neighbouring frames get
/// wholly different processing, and the seams are audible as cracking. On a
/// 60 s sample it also produced 17 seconds of *exact* digital silence.
///
/// Pushing the thresholds out of reach keeps one treatment in force. That is
/// not a tuning compromise: it makes the output bit-for-bit indistinguishable
/// from the reference DeepFilterNet3 model (correlation 1.0000), because that
/// single-graph export has no equivalent switching — it simply always applies
/// both stages. See docs/model-pipeline.md.
const NEVER: f32 = -100.0;
const ALWAYS: f32 = 100.0;

pub struct TractDenoiser {
    inner: DfTract,
    /// `process` wants 2-D `[channels, samples]` views, and we are always mono.
    scratch_in: Array2<f32>,
    scratch_out: Array2<f32>,
}

// SAFETY: `DfTract` holds `Rc`s internally (tract's `TValue` is an
// `Rc<Tensor>`), so it is not `Send` by derivation. Sending it is nonetheless
// sound here because ownership is *exclusive and moved exactly once*:
// `DfTract::new` allocates every one of those `Rc`s fresh, none of them is
// cloned out of the struct, and `TractDenoiser` is handed to the DSP thread
// and never touched again from anywhere else. `Rc` is only unsound across
// threads when refcounts are updated concurrently, which requires two live
// handles — there is only ever one.
//
// The pipeline builds the denoiser on the calling thread and moves it into the
// audio thread; if that ever changes to sharing rather than moving, this
// becomes wrong and must be revisited.
#[allow(unsafe_code)]
unsafe impl Send for TractDenoiser {}

impl TractDenoiser {
    /// Load from a `DeepFilterNet3_onnx.tar.gz` as produced by
    /// `scripts/export_dfn3.py`.
    pub fn load(tar_gz: impl AsRef<Path>, attenuation_db: f32) -> Result<Self> {
        let path = tar_gz.as_ref();
        let params = DfParams::new(path.to_path_buf())
            .map_err(|e| DspError::Load(format!("loading {}: {e}", path.display())))?;

        // 0 dB would mean "suppress nothing", which is not what the config
        // means by 0 — there it means "no limit". Upstream spells that as a
        // very large value.
        let atten = if attenuation_db <= 0.0 {
            100.0
        } else {
            attenuation_db
        };

        let runtime = RuntimeParams::default_with_ch(1)
            .with_atten_lim(atten)
            .with_thresholds(NEVER, ALWAYS, ALWAYS);

        let inner = DfTract::new(params, &runtime)
            .map_err(|e| DspError::Load(format!("initialising the tract model: {e}")))?;

        if inner.hop_size != FRAME_SAMPLES {
            return Err(DspError::Load(format!(
                "model hop size is {} but the pipeline works in {FRAME_SAMPLES}-sample frames",
                inner.hop_size
            )));
        }

        tracing::info!(
            model = %path.display(),
            hop = inner.hop_size,
            atten_lim_db = atten,
            "loaded DeepFilterNet3 via tract"
        );

        Ok(Self {
            inner,
            scratch_in: Array2::zeros((1, FRAME_SAMPLES)),
            scratch_out: Array2::zeros((1, FRAME_SAMPLES)),
        })
    }
}

impl Denoiser for TractDenoiser {
    fn process_frame(&mut self, frame: &mut [f32; FRAME_SAMPLES]) -> Result<()> {
        self.scratch_in
            .row_mut(0)
            .as_slice_mut()
            .expect("contiguous")
            .copy_from_slice(frame);

        let input: ArrayView2<f32> = self.scratch_in.view();
        self.inner
            .process(input, self.scratch_out.view_mut())
            .map_err(|e| DspError::Inference(format!("tract process: {e}")))?;

        frame.copy_from_slice(self.scratch_out.row(0).as_slice().expect("contiguous"));
        Ok(())
    }

    fn name(&self) -> &'static str {
        "DeepFilterNet3 (tract)"
    }
}
