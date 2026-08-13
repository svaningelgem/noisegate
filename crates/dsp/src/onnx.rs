//! Optional ONNX Runtime backend, for running a real model instead of the
//! built-in RNNoise — DeepFilterNet3 being the obvious one.
//!
//! Build with `--features onnx`. You'll need the ONNX Runtime DLL
//! (onnxruntime.dll); we use `load-dynamic` so it doesn't have to ship
//! inside the binary. Put it **next to noisegate.exe**, or point
//! `ORT_DYLIB_PATH` at it explicitly — see [`pin_dylib_path`] for why we
//! don't let the OS search for it.
//!
//! ## Expected model signature
//! Streaming, frame-at-a-time models that carry their own recurrent state,
//! which is what every usable real-time denoiser looks like:
//!
//! | tensor         | direction | shape       |
//! |----------------|-----------|-------------|
//! | `input_frame`  | in        | `[480]` f32 |
//! | `states`       | in        | `[S]` f32   |
//! | `atten_lim_db` | in        | `[1]` f32   |
//! | enhanced audio | out       | `[480]` f32 |
//! | new states     | out       | `[S]` f32   |
//!
//! Inputs are matched by name so export naming quirks don't matter; outputs
//! are positional (enhanced audio, then new state; anything after is ignored).
//! The state width `S` is read from the model, so exports with different state
//! sizes work without a recompile.
//!
//! Models with a different contract — spectral input, explicit STFT/iSTFT, a
//! multi-file encoder/decoder split — need their own front-end and aren't
//! handled here.

use std::path::Path;

use ort::session::Session;
use ort::value::TensorRef;

use crate::{Denoiser, DspError, Result, FRAME_SAMPLES};

pub struct OnnxDenoiser {
    session: Session,
    /// Input tensor names, resolved once at load so the hot path doesn't
    /// re-query them.
    frame_input: String,
    states_input: String,
    atten_input: Option<String>,
    /// Recurrent state, handed back to the model on every frame.
    states: Vec<f32>,
    /// Attenuation limit in dB. 0 = let the model suppress as much as it
    /// wants; a positive value deliberately leaves some noise in.
    atten_lim_db: f32,
    model_label: String,
    scratch_in: [f32; FRAME_SAMPLES],
}

/// Pin `onnxruntime.dll` to our own install directory unless the operator has
/// already chosen a path.
///
/// `ort`'s `load-dynamic` otherwise hands the bare filename to the OS loader,
/// whose search order includes the current working directory. Launch
/// NoiseGate from Downloads with a hostile `onnxruntime.dll` sitting there and
/// it executes inside a process that holds the microphone open. Naming the
/// full path removes the search entirely.
pub(crate) fn pin_dylib_path() {
    let already_set = std::env::var_os("ORT_DYLIB_PATH").is_some();
    let exe_dir = std::env::current_exe()
        .ok()
        .and_then(|p| p.parent().map(Path::to_path_buf));
    if let Some(path) = dylib_path_to_pin(already_set, exe_dir.as_deref()) {
        std::env::set_var("ORT_DYLIB_PATH", path);
    }
}

/// Where `ORT_DYLIB_PATH` should point, or `None` to leave it alone.
///
/// Separated from the environment so the rule itself can be tested: this is
/// the guard against loading a stray `onnxruntime.dll`, and getting it wrong
/// is a code-execution bug rather than a cosmetic one.
fn dylib_path_to_pin(already_set: bool, exe_dir: Option<&Path>) -> Option<std::path::PathBuf> {
    if already_set {
        return None; // Explicit operator choice wins.
    }
    Some(exe_dir?.join("onnxruntime.dll"))
}

/// Pick the input whose name contains `needle`, falling back to positional
/// order. Exports disagree on naming (`input_frame` vs `frame` vs `input`),
/// and guessing wrong here produces a baffling runtime error rather than a
/// clear one.
fn find_input(session: &Session, needle: &str, fallback_index: usize) -> Option<String> {
    session
        .inputs
        .iter()
        .find(|i| i.name.to_ascii_lowercase().contains(needle))
        .or_else(|| session.inputs.get(fallback_index))
        .map(|i| i.name.clone())
}

/// Number of elements a fixed-size input expects, if the model declares it.
fn declared_len(session: &Session, name: &str) -> Option<usize> {
    let input = session.inputs.iter().find(|i| i.name == name)?;
    let dims = input.input_type.tensor_shape()?;
    let total: i64 = dims.iter().filter(|d| **d > 0).product();
    (total > 0).then_some(total as usize)
}

impl OnnxDenoiser {
    /// Load an ONNX model from disk.
    pub fn load(path: impl AsRef<Path>) -> Result<Self> {
        pin_dylib_path();
        let path = path.as_ref();
        let label = path
            .file_name()
            .map(|s| s.to_string_lossy().into_owned())
            .unwrap_or_else(|| "onnx-model".into());

        let session = Session::builder()
            .map_err(|e| DspError::Load(format!("ort builder: {e}")))?
            .with_intra_threads(1)
            .map_err(|e| DspError::Load(format!("with_intra_threads: {e}")))?
            .commit_from_file(path)
            .map_err(|e| DspError::Load(format!("commit_from_file({}): {e}", path.display())))?;

        let frame_input = find_input(&session, "frame", 0)
            .ok_or_else(|| DspError::Load("model has no inputs".into()))?;
        let states_input = find_input(&session, "state", 1).ok_or_else(|| {
            DspError::Load(
                "model has no state input; this loader handles streaming models that carry \
                 recurrent state (e.g. DeepFilterNet3 exports)"
                    .into(),
            )
        })?;
        // Optional — not every export exposes an attenuation limit.
        let atten_input = session
            .inputs
            .iter()
            .find(|i| i.name.to_ascii_lowercase().contains("atten"))
            .map(|i| i.name.clone());

        // Outputs are positional, and `process_frame` indexes both. Check the
        // count here, where there is a user to tell, rather than panicking on
        // the audio thread — which aborts the process in a release build.
        if session.outputs.len() < 2 {
            return Err(DspError::Load(format!(
                "model has {} output(s); this loader needs two — the enhanced audio and \
                 the model's new state, in that order. A model without a state output is \
                 not a streaming model and cannot be run frame at a time",
                session.outputs.len()
            )));
        }

        let state_len = declared_len(&session, &states_input).ok_or_else(|| {
            DspError::Load(format!(
                "could not determine the size of state input '{states_input}'"
            ))
        })?;

        tracing::info!(
            model = %label,
            frame_input,
            states_input,
            state_len,
            atten_input = atten_input.as_deref().unwrap_or("<none>"),
            "loaded ONNX denoiser"
        );

        Ok(Self {
            session,
            frame_input,
            states_input,
            atten_input,
            states: vec![0.0; state_len],
            atten_lim_db: 0.0,
            model_label: label,
            scratch_in: [0.0; FRAME_SAMPLES],
        })
    }

    /// Attenuation limit in dB, if the model takes one. 0 means "suppress as
    /// much as you like"; a value like 12 leaves some noise in, which some
    /// people prefer on a call.
    pub fn set_attenuation_db(&mut self, db: f32) {
        self.atten_lim_db = db.max(0.0);
    }
}

impl Denoiser for OnnxDenoiser {
    fn process_frame(&mut self, frame: &mut [f32; FRAME_SAMPLES]) -> Result<()> {
        self.scratch_in.copy_from_slice(frame);

        let frame_tensor =
            TensorRef::from_array_view(([FRAME_SAMPLES as i64], &self.scratch_in[..]))
                .map_err(|e| DspError::Inference(format!("input_frame: {e}")))?;
        let states_tensor =
            TensorRef::from_array_view(([self.states.len() as i64], &self.states[..]))
                .map_err(|e| DspError::Inference(format!("states: {e}")))?;
        let atten = [self.atten_lim_db];
        let atten_tensor = TensorRef::from_array_view(([1_i64], &atten[..]))
            .map_err(|e| DspError::Inference(format!("atten_lim_db: {e}")))?;

        // The macro fixes the (name, value) types; the optional third input
        // is pushed onto the same vec.
        let mut inputs = ort::inputs![
            self.frame_input.as_str() => frame_tensor,
            self.states_input.as_str() => states_tensor,
        ];
        if let Some(name) = &self.atten_input {
            inputs.push((name.as_str().into(), atten_tensor.into()));
        }

        let outputs = self
            .session
            .run(inputs)
            .map_err(|e| DspError::Inference(format!("session.run: {e}")))?;

        // Outputs are positional: enhanced audio first, new state second.
        let (_, enhanced) = outputs[0]
            .try_extract_tensor::<f32>()
            .map_err(|e| DspError::Inference(format!("enhanced output: {e}")))?;
        if enhanced.len() != FRAME_SAMPLES {
            return Err(DspError::Inference(format!(
                "expected {FRAME_SAMPLES} output samples, got {}",
                enhanced.len()
            )));
        }
        frame.copy_from_slice(enhanced);

        let (_, new_states) = outputs[1]
            .try_extract_tensor::<f32>()
            .map_err(|e| DspError::Inference(format!("new states output: {e}")))?;
        if new_states.len() != self.states.len() {
            return Err(DspError::Inference(format!(
                "model returned {} state values, expected {}",
                new_states.len(),
                self.states.len()
            )));
        }
        self.states.copy_from_slice(new_states);
        Ok(())
    }

    fn name(&self) -> &'static str {
        // `&'static str` from a runtime-loaded label requires leaking the
        // string once at load. Cheap (a few bytes once per process) and
        // makes the trait simple. If you load multiple models, leak each
        // label once.
        Box::leak(self.model_label.clone().into_boxed_str())
    }
}

#[cfg(test)]
mod tests {
    use std::path::PathBuf;

    use super::*;

    // ---- the dylib guard, which needs neither runtime nor model ----------

    /// Naming the full path is what stops the OS loader searching the current
    /// working directory. Launch NoiseGate from Downloads with a hostile
    /// `onnxruntime.dll` sitting there and a bare filename would execute it
    /// inside a process holding the microphone open.
    #[test]
    fn the_runtime_is_pinned_next_to_our_own_executable() {
        let pinned = dylib_path_to_pin(false, Some(Path::new(r"C:\Program Files\NoiseGate")))
            .expect("should pin a path");
        assert_eq!(
            pinned,
            PathBuf::from(r"C:\Program Files\NoiseGate\onnxruntime.dll")
        );
        assert!(
            pinned.is_absolute(),
            "a bare filename would be searched for"
        );
        assert!(pinned.parent().is_some(), "must name a directory");
    }

    #[test]
    fn an_operators_own_choice_is_not_overridden() {
        assert_eq!(
            dylib_path_to_pin(true, Some(Path::new(r"C:\NoiseGate"))),
            None,
            "ORT_DYLIB_PATH was set deliberately; leave it alone"
        );
    }

    /// Without an executable path there is nothing safe to point at. Falling
    /// back to a bare filename would reintroduce the search we are avoiding.
    #[test]
    fn an_unknown_executable_location_pins_nothing() {
        assert_eq!(dylib_path_to_pin(false, None), None);
    }

    // ---- the loader, against a model that implements just the contract ---

    fn model_path() -> PathBuf {
        PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("testdata/streaming_contract.onnx")
    }

    /// The ONNX Runtime DLL is not redistributed in the source tree; CI
    /// downloads it and the installer ships it. Point `ort` at wherever it
    /// actually is, or report that these tests could not run.
    fn ort_available() -> bool {
        if std::env::var_os("ORT_DYLIB_PATH").is_some() {
            return true;
        }
        let root = PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../../target");
        for dir in ["debug", "release", "debug/deps"] {
            let candidate = root.join(dir).join("onnxruntime.dll");
            if candidate.exists() {
                std::env::set_var("ORT_DYLIB_PATH", &candidate);
                return true;
            }
        }
        // A skip is fine on a developer machine that has never fetched the
        // runtime. On CI it would mean these tests quietly stopped running
        // while the build stayed green, so make it fatal there instead.
        assert!(
            std::env::var_os("CI").is_none(),
            "onnxruntime.dll not found under target/. CI is supposed to download it \
             before running tests — see the 'Fetch ONNX Runtime' step."
        );
        eprintln!(
            "SKIP: onnxruntime.dll not found under target/. These tests need it; \
             fetch it with the same URL the release workflow uses."
        );
        false
    }

    #[test]
    fn the_state_width_is_read_from_the_model() {
        if !ort_available() {
            return;
        }
        let d = OnnxDenoiser::load(model_path()).expect("load the contract model");
        assert_eq!(
            d.states.len(),
            4,
            "the loader must take the state size from the model, not a constant"
        );
        assert_eq!(d.frame_input, "input_frame");
        assert_eq!(d.states_input, "states");
        assert_eq!(d.atten_input.as_deref(), Some("atten_lim_db"));
        assert!(d.name().contains("streaming_contract"));
    }

    /// The whole point of the streaming contract: the model's new state has to
    /// come back in on the next frame. A loader that dropped it would still
    /// produce audio, just with the model permanently reset — which sounds
    /// plausible and is completely wrong.
    ///
    /// The fixture returns `input * (states[0] + 1)` and increments the state,
    /// so a fed-back state shows up as a doubling and a dropped one does not.
    #[test]
    fn the_models_state_is_fed_back_on_the_next_frame() {
        if !ort_available() {
            return;
        }
        let mut d = OnnxDenoiser::load(model_path()).unwrap();
        let mut frame = [0.25f32; FRAME_SAMPLES];

        d.process_frame(&mut frame).unwrap();
        assert!(
            frame.iter().all(|&s| (s - 0.25).abs() < 1e-6),
            "first frame, state 0: expected the input back, got {}",
            frame[0]
        );

        d.process_frame(&mut frame).unwrap();
        assert!(
            frame.iter().all(|&s| (s - 0.5).abs() < 1e-6),
            "second frame should have seen state 1 and doubled; got {} \
             (0.25 means the state was not fed back)",
            frame[0]
        );

        d.process_frame(&mut frame).unwrap();
        assert!(
            frame.iter().all(|&s| (s - 1.5).abs() < 1e-6),
            "third frame should have seen state 2; got {}",
            frame[0]
        );
    }

    /// The attenuation limit is optional in the contract, so it is easy to
    /// wire up and never actually send. The fixture adds it to every sample.
    #[test]
    fn the_attenuation_limit_reaches_the_model() {
        if !ort_available() {
            return;
        }
        let mut d = OnnxDenoiser::load(model_path()).unwrap();
        d.set_attenuation_db(3.0);

        let mut frame = [0.0f32; FRAME_SAMPLES];
        d.process_frame(&mut frame).unwrap();
        assert!(
            (frame[0] - 3.0).abs() < 1e-6,
            "expected the attenuation to arrive, got {}",
            frame[0]
        );
    }

    /// "Suppress as much as you like" is 0; a negative limit is meaningless
    /// and must not reach the model as one.
    #[test]
    fn a_negative_attenuation_is_clamped_to_zero() {
        if !ort_available() {
            return;
        }
        let mut d = OnnxDenoiser::load(model_path()).unwrap();
        d.set_attenuation_db(-10.0);
        assert_eq!(d.atten_lim_db, 0.0);

        let mut frame = [0.0f32; FRAME_SAMPLES];
        d.process_frame(&mut frame).unwrap();
        assert_eq!(frame[0], 0.0);
    }

    /// Outputs are taken positionally, so a model that returns only the
    /// enhanced audio used to index `outputs[1]` and panic — on the audio
    /// thread, which aborts the process in a release build. Refuse it at load,
    /// where there is a user to tell.
    #[test]
    fn a_model_that_returns_no_new_state_is_refused_at_load() {
        if !ort_available() {
            return;
        }
        let path = PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("testdata/one_output.onnx");
        let msg = match OnnxDenoiser::load(&path) {
            Ok(_) => panic!("a model with one output must not load"),
            Err(e) => e.to_string(),
        };
        assert!(
            msg.contains("output"),
            "the message should say what is wrong with the model: {msg}"
        );
    }

    /// Pointing `model_path` at the wrong file is an ordinary mistake, and the
    /// message has to name the file rather than surface an ort internal.
    #[test]
    fn a_file_that_is_not_a_model_is_refused_by_name() {
        if !ort_available() {
            return;
        }
        let junk =
            std::env::temp_dir().join(format!("noisegate-not-a-model-{}.onnx", std::process::id()));
        std::fs::write(&junk, b"this is not a protobuf").unwrap();

        let msg = match OnnxDenoiser::load(&junk) {
            Ok(_) => panic!("a text file must not load as a model"),
            Err(e) => e.to_string(),
        };
        assert!(
            msg.contains("not-a-model"),
            "the message should name the file: {msg}"
        );
        let _ = std::fs::remove_file(&junk);
    }
}
