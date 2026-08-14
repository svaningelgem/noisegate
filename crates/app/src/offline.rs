//! Offline WAV helpers, so you can hear what the denoiser does without
//! installing VB-Cable or wiring up any virtual device: record a clip from
//! your mic, run it through the model, and A/B the two files.
//!
//! Everything here works on the pipeline's native format — mono f32 @ 48 kHz
//! in 480-sample frames — so what you hear is what the live path produces.

use std::path::Path;
use std::sync::{Arc, Mutex};

use anyhow::{bail, Context, Result};
use tracing::info;

use audio_io::wasapi_capture::{Frame, FrameSink};
use dsp::FRAME_SAMPLES;

const SAMPLE_RATE: u32 = 48_000;

fn wav_spec() -> hound::WavSpec {
    hound::WavSpec {
        channels: 1,
        sample_rate: SAMPLE_RATE,
        bits_per_sample: 16,
        sample_format: hound::SampleFormat::Int,
    }
}

/// f32 in [-1, 1] to i16, clamped rather than wrapped — a wrapped sample is a
/// full-scale click.
fn to_i16(s: f32) -> i16 {
    (s.clamp(-1.0, 1.0) * i16::MAX as f32) as i16
}

fn write_wav(path: &Path, samples: &[f32]) -> Result<()> {
    let mut w = hound::WavWriter::create(path, wav_spec())
        .with_context(|| format!("creating {}", path.display()))?;
    for &s in samples {
        w.write_sample(to_i16(s))?;
    }
    w.finalize().context("finalizing wav")?;
    Ok(())
}

fn read_wav(path: &Path) -> Result<Vec<f32>> {
    let mut r =
        hound::WavReader::open(path).with_context(|| format!("opening {}", path.display()))?;
    let spec = r.spec();
    if spec.channels != 1 || spec.sample_rate != SAMPLE_RATE {
        bail!(
            "{} is {} ch @ {} Hz; this tool wants mono 48 kHz (what `--record` produces)",
            path.display(),
            spec.channels,
            spec.sample_rate
        );
    }
    let samples: Vec<f32> = match spec.sample_format {
        hound::SampleFormat::Float => r.samples::<f32>().collect::<Result<_, _>>()?,
        hound::SampleFormat::Int => {
            let scale = 1.0 / (1i32 << (spec.bits_per_sample - 1)) as f32;
            r.samples::<i32>()
                .map(|s| s.map(|v| v as f32 * scale))
                .collect::<Result<_, _>>()?
        }
    };
    Ok(samples)
}

/// Capture `seconds` of audio from `device_id` straight to a WAV file.
///
/// Doubles as the only way to exercise the capture path end to end when no
/// virtual cable is installed.
pub fn record(device_name: &str, seconds: f32, out: &Path) -> Result<()> {
    // Names are what the config and CLI speak; ids are resolved here.
    let devices = audio_io::devices::DeviceList::enumerate().map_err(|e| anyhow::anyhow!(e))?;
    let device = devices
        .resolve_capture(device_name)
        .ok_or_else(|| anyhow::anyhow!("no microphone matching {device_name:?}"))?;
    let device_id = device.id.clone();
    info!(mic = %device.friendly_name, "recording from");
    let captured: Arc<Mutex<Vec<f32>>> = Arc::new(Mutex::new(Vec::new()));

    struct Sink(Arc<Mutex<Vec<f32>>>);
    impl FrameSink for Sink {
        fn on_frame(&mut self, frame: &Frame) {
            // Locking on the audio thread is a sin in the live path, but this
            // is an offline recorder — dropping frames to look principled
            // would just corrupt the file.
            if let Ok(mut buf) = self.0.lock() {
                buf.extend_from_slice(frame);
            }
        }
        fn on_glitch(&mut self, flags: u32) {
            tracing::warn!(flags, "capture glitch during recording");
        }
    }

    info!(seconds, path = %out.display(), "recording — make some noise");
    let capture = audio_io::WasapiCapture::start(&device_id, Box::new(Sink(captured.clone())))
        .map_err(|e| anyhow::anyhow!(e))
        .context("opening the capture device")?;

    std::thread::sleep(std::time::Duration::from_secs_f32(seconds));
    drop(capture); // stops and joins the capture thread

    let samples = captured.lock().unwrap().clone();
    if samples.is_empty() {
        bail!(
            "captured nothing. Check Settings → Privacy & Security → Microphone, \
             and that the mic isn't hardware-muted"
        );
    }
    write_wav(out, &samples)?;
    info!(
        samples = samples.len(),
        secs = samples.len() as f32 / SAMPLE_RATE as f32,
        peak = samples.iter().fold(0.0f32, |a, s| a.max(s.abs())),
        "recorded"
    );
    Ok(())
}

/// Run a WAV through the denoiser and write the result, reporting how much
/// energy was removed so there's a number to go with the listening test.
pub fn denoise_file(
    input: &Path,
    output: &Path,
    model: Option<&Path>,
    attenuation_db: f32,
) -> Result<()> {
    let samples = read_wav(input)?;
    let mut denoiser = dsp::build_denoiser(model, attenuation_db).context("loading denoiser")?;
    info!(denoiser = denoiser.name(), "processing");

    let mut out = Vec::with_capacity(samples.len());
    let mut frame = [0f32; FRAME_SAMPLES];
    let started = std::time::Instant::now();

    // Zero-pad the tail so the final partial frame still gets processed.
    for chunk in samples.chunks(FRAME_SAMPLES) {
        frame[..chunk.len()].copy_from_slice(chunk);
        frame[chunk.len()..].fill(0.0);
        denoiser.process_frame(&mut frame)?;
        out.extend_from_slice(&frame[..chunk.len()]);
    }

    let elapsed = started.elapsed();
    let audio_secs = samples.len() as f32 / SAMPLE_RATE as f32;
    write_wav(output, &out)?;

    // Whole-file RMS is a useless measure here: it's dominated by speech, so
    // scrubbing the noise floor to nothing barely moves it. Compare the quiet
    // parts and the loud parts separately instead — a good denoiser drops the
    // floor a long way and leaves the speech where it was.
    let before = LevelProfile::measure(&samples);
    let after = LevelProfile::measure(&out);
    info!(
        noise_floor = format!("{:.1} -> {:.1} dB ({:+.1})", before.floor_db, after.floor_db, after.floor_db - before.floor_db),
        speech = format!("{:.1} -> {:.1} dB ({:+.1})", before.speech_db, after.speech_db, after.speech_db - before.speech_db),
        rtf = format!("{:.3}", elapsed.as_secs_f32() / audio_secs),
        path = %output.display(),
        "done"
    );
    Ok(())
}

/// Quiet-part and loud-part levels of a signal, in dBFS.
#[derive(Debug, Clone, Copy)]
struct LevelProfile {
    /// 10th-percentile window level — the residual noise between words.
    floor_db: f32,
    /// 95th-percentile window level — how loud the speech itself is.
    speech_db: f32,
}

impl LevelProfile {
    const WINDOW: usize = 960; // 20 ms

    fn measure(samples: &[f32]) -> Self {
        let mut levels: Vec<f32> = samples
            .chunks(Self::WINDOW)
            // Skip digitally silent windows: an upstream gate that already
            // zeroed them would otherwise read as a spectacular noise floor.
            .filter(|w| w.iter().any(|&s| s != 0.0))
            .map(|w| db(rms(w)))
            .collect();
        if levels.is_empty() {
            return Self {
                floor_db: db(0.0),
                speech_db: db(0.0),
            };
        }
        levels.sort_by(f32::total_cmp);
        Self {
            floor_db: percentile(&levels, 0.10),
            speech_db: percentile(&levels, 0.95),
        }
    }
}

/// `sorted` must be ascending; `q` in [0, 1].
fn percentile(sorted: &[f32], q: f32) -> f32 {
    let i = ((sorted.len() as f32 * q) as usize).min(sorted.len() - 1);
    sorted[i]
}

fn rms(x: &[f32]) -> f32 {
    if x.is_empty() {
        return 0.0;
    }
    (x.iter().map(|s| s * s).sum::<f32>() / x.len() as f32).sqrt()
}

fn db(x: f32) -> f32 {
    20.0 * x.max(1e-10).log10()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn wav_round_trips_through_disk() {
        let dir = std::env::temp_dir().join(format!("roommute-wav-{}", std::process::id()));
        std::fs::create_dir_all(&dir).unwrap();
        let path = dir.join("tone.wav");

        // A quarter second of 1 kHz.
        let input: Vec<f32> = (0..12_000)
            .map(|i| (i as f32 * 2.0 * std::f32::consts::PI * 1000.0 / 48_000.0).sin() * 0.5)
            .collect();
        write_wav(&path, &input).unwrap();
        let read_back = read_wav(&path).unwrap();

        assert_eq!(read_back.len(), input.len());
        for (a, b) in input.iter().zip(&read_back) {
            assert!((a - b).abs() < 1e-3, "{a} vs {b}");
        }
        let _ = std::fs::remove_dir_all(&dir);
    }

    #[test]
    fn clipping_saturates_instead_of_wrapping() {
        // A wrapped sample flips sign and is audible as a full-scale click.
        assert_eq!(to_i16(2.0), i16::MAX);
        assert_eq!(to_i16(-2.0), -i16::MAX);
        assert_eq!(to_i16(0.0), 0);
    }

    #[test]
    fn rejects_wrong_sample_rate() {
        let dir = std::env::temp_dir().join(format!("roommute-wav-bad-{}", std::process::id()));
        std::fs::create_dir_all(&dir).unwrap();
        let path = dir.join("wrong.wav");

        let spec = hound::WavSpec {
            sample_rate: 44_100,
            ..wav_spec()
        };
        let mut w = hound::WavWriter::create(&path, spec).unwrap();
        w.write_sample(0i16).unwrap();
        w.finalize().unwrap();

        assert!(read_wav(&path).is_err());
        let _ = std::fs::remove_dir_all(&dir);
    }

    #[test]
    fn rms_and_db_are_sane() {
        assert_eq!(rms(&[]), 0.0);
        assert!((rms(&[1.0, -1.0, 1.0, -1.0]) - 1.0).abs() < 1e-6);
        assert!((db(1.0) - 0.0).abs() < 1e-6);
        assert!((db(0.5) + 6.02).abs() < 0.01);
    }

    /// Build alternating loud/quiet windows and check the profile separates
    /// them — this is the measurement the whole tool reports on.
    fn alternating(loud: f32, quiet: f32) -> Vec<f32> {
        let mut v = Vec::new();
        for i in 0..40 {
            let amp = if i % 2 == 0 { loud } else { quiet };
            // Square wave at `amp`, so window RMS is exactly `amp`.
            v.extend((0..LevelProfile::WINDOW).map(|j| if j % 2 == 0 { amp } else { -amp }));
        }
        v
    }

    #[test]
    fn profile_separates_floor_from_speech() {
        let p = LevelProfile::measure(&alternating(0.5, 0.005));
        assert!(
            (p.floor_db - db(0.005)).abs() < 0.5,
            "floor was {}",
            p.floor_db
        );
        assert!(
            (p.speech_db - db(0.5)).abs() < 0.5,
            "speech was {}",
            p.speech_db
        );
    }

    #[test]
    fn digital_silence_does_not_fake_a_good_floor() {
        // An upstream gate that zeroes the pauses must not read as -200 dB.
        let mut v = alternating(0.5, 0.005);
        for s in v.iter_mut().take(LevelProfile::WINDOW * 10) {
            *s = 0.0;
        }
        let p = LevelProfile::measure(&v);
        assert!(
            p.floor_db > -100.0,
            "silence leaked into the floor: {}",
            p.floor_db
        );
    }

    #[test]
    fn profile_of_nothing_is_not_a_panic() {
        let p = LevelProfile::measure(&[]);
        assert!(p.floor_db < -100.0 && p.speech_db < -100.0);
    }

    /// Float WAVs come out of most editors; ints come out of `--record`. Both
    /// have to load, and land at the same amplitude.
    #[test]
    fn float_wavs_load_at_the_same_scale_as_int_ones() {
        let dir = std::env::temp_dir().join(format!("roommute-f32-{}", std::process::id()));
        std::fs::create_dir_all(&dir).unwrap();
        let path = dir.join("float.wav");

        let spec = hound::WavSpec {
            bits_per_sample: 32,
            sample_format: hound::SampleFormat::Float,
            ..wav_spec()
        };
        let mut w = hound::WavWriter::create(&path, spec).unwrap();
        for s in [0.0f32, 0.5, -0.5, 1.0] {
            w.write_sample(s).unwrap();
        }
        w.finalize().unwrap();

        assert_eq!(read_wav(&path).unwrap(), vec![0.0, 0.5, -0.5, 1.0]);
        let _ = std::fs::remove_dir_all(&dir);
    }

    /// The `--denoise` path, end to end: a noisy file in, a quieter one out,
    /// same length. Runs on the built-in RNNoise so it needs no model file.
    #[test]
    fn denoising_a_file_drops_the_noise_floor_and_keeps_the_length() {
        let dir = std::env::temp_dir().join(format!("roommute-denoise-{}", std::process::id()));
        std::fs::create_dir_all(&dir).unwrap();
        let (input, output) = (dir.join("in.wav"), dir.join("out.wav"));

        // Two seconds of white noise. Deliberately not a tone: a denoiser is
        // free to keep a tone, but hiss is exactly what it exists to remove.
        // Deterministic LCG rather than a rand dependency for one test.
        let mut seed = 0x2545_F491_4F6C_DD1Du64;
        let noisy: Vec<f32> = (0..SAMPLE_RATE as usize * 2)
            .map(|_| {
                seed = seed.wrapping_mul(6364136223846793005).wrapping_add(1);
                ((seed >> 33) as f32 / (1u64 << 31) as f32 - 1.0) * 0.2
            })
            .collect();
        write_wav(&input, &noisy).unwrap();

        denoise_file(&input, &output, None, 0.0).unwrap();

        let cleaned = read_wav(&output).unwrap();
        assert_eq!(cleaned.len(), noisy.len(), "length must be preserved");

        let before = LevelProfile::measure(&noisy);
        let after = LevelProfile::measure(&cleaned);
        assert!(
            after.floor_db < before.floor_db - 6.0,
            "expected the hiss to drop: {:.1} -> {:.1} dB",
            before.floor_db,
            after.floor_db
        );
        let _ = std::fs::remove_dir_all(&dir);
    }

    /// A partial final frame must still reach the output. Zero-padding it and
    /// then writing the whole padded frame would lengthen the file.
    #[test]
    fn a_file_that_is_not_a_whole_number_of_frames_keeps_its_length() {
        let dir = std::env::temp_dir().join(format!("roommute-partial-{}", std::process::id()));
        std::fs::create_dir_all(&dir).unwrap();
        let (input, output) = (dir.join("in.wav"), dir.join("out.wav"));

        let odd = FRAME_SAMPLES * 3 + 7;
        write_wav(&input, &vec![0.1f32; odd]).unwrap();
        denoise_file(&input, &output, None, 0.0).unwrap();

        assert_eq!(read_wav(&output).unwrap().len(), odd);
        let _ = std::fs::remove_dir_all(&dir);
    }

    #[test]
    fn denoising_a_file_that_is_not_there_says_so() {
        let missing = std::env::temp_dir().join("roommute-nope/in.wav");
        let err = denoise_file(&missing, &missing.with_file_name("out.wav"), None, 0.0)
            .expect_err("must not pretend to succeed");
        assert!(
            format!("{err:#}").contains("in.wav"),
            "the message should name the file: {err:#}"
        );
    }
}
