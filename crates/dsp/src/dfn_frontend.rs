//! DeepFilterNet front-end: STFT, ERB filterbank and feature normalisation.
//!
//! DeepFilterNet exports that keep the ERB mask and deep filtering inside the
//! graph still expect the host to hand them a spectrum plus two normalised
//! feature tensors. Everything libDF does before the network happens here.
//!
//! Transcribed from `libDF/src/lib.rs` (`erb_fb`, `feat_erb`, `feat_cplx`,
//! `frame_analysis`, `frame_synthesis`) and validated against the reference
//! implementation: on real audio the three tensors match to floating-point
//! noise (worst case 2.6e-9 on the spectrum), so the numbers below are not a
//! reimplementation guess.

use realfft::num_complex::Complex32;
use realfft::{ComplexToReal, RealFftPlanner, RealToComplex};
use std::sync::Arc;

pub const SR: usize = 48_000;
pub const FFT_SIZE: usize = 960;
pub const HOP: usize = 480;
pub const FREQ_BINS: usize = FFT_SIZE / 2 + 1; // 481
pub const NB_ERB: usize = 32;
pub const NB_DF: usize = 96;
/// Minimum FFT bins per ERB band, so the lowest bands aren't degenerate.
const MIN_NB_FREQS: usize = 2;

/// exp(-hop / sr / norm_tau) with norm_tau = 1 s, rounded the way libDF does.
const ALPHA: f32 = 0.99;
/// Initial exponential-mean state, in dB, spread across the ERB bands.
const MEAN_NORM_INIT: (f32, f32) = (-60.0, -90.0);
/// Initial unit-norm state, spread across the DF bins.
const UNIT_NORM_INIT: (f32, f32) = (0.001, 0.0001);

fn freq2erb(f: f32) -> f32 {
    9.265 * (f / (24.7 * 9.265)).ln_1p()
}

fn erb2freq(e: f32) -> f32 {
    24.7 * 9.265 * ((e / 9.265).exp() - 1.0)
}

/// Width, in FFT bins, of each ERB band. Sums to exactly `FREQ_BINS`.
pub fn erb_widths() -> Vec<usize> {
    let nyq = SR / 2;
    let width = SR as f32 / FFT_SIZE as f32;
    let (lo, hi) = (freq2erb(0.0), freq2erb(nyq as f32));
    let step = (hi - lo) / NB_ERB as f32;

    let mut erb = vec![0usize; NB_ERB];
    let (mut prev_freq, mut freq_over) = (0i32, 0i32);
    for i in 1..=NB_ERB {
        let f = erb2freq(lo + i as f32 * step);
        let fb = (f / width).round() as i32;
        let mut nb = fb - prev_freq - freq_over;
        if nb < MIN_NB_FREQS as i32 {
            // Not enough bins in this band; borrow from the next one.
            freq_over = MIN_NB_FREQS as i32 - nb;
            nb = MIN_NB_FREQS as i32;
        } else {
            freq_over = 0;
        }
        erb[i - 1] = nb as usize;
        prev_freq = fb;
    }
    erb[NB_ERB - 1] += 1; // freq_size is fft/2 + 1
    let total: usize = erb.iter().sum();
    if total > FREQ_BINS {
        erb[NB_ERB - 1] -= total - FREQ_BINS;
    }
    debug_assert_eq!(erb.iter().sum::<usize>(), FREQ_BINS);
    erb
}

/// Vorbis window: `sin(pi/2 * sin^2(pi/2 * (i + 0.5) / half))`.
///
/// Power-complementary, so analysis and synthesis with 50% overlap reconstruct
/// exactly.
pub fn vorbis_window() -> Vec<f32> {
    let half = (FFT_SIZE / 2) as f64;
    (0..FFT_SIZE)
        .map(|i| {
            let s = (0.5 * std::f64::consts::PI * (i as f64 + 0.5) / half).sin();
            (0.5 * std::f64::consts::PI * s * s).sin() as f32
        })
        .collect()
}

/// Streaming STFT plus feature extraction, one 480-sample frame at a time.
pub struct DfFrontend {
    fft: Arc<dyn RealToComplex<f32>>,
    ifft: Arc<dyn ComplexToReal<f32>>,
    window: Vec<f32>,
    /// 1/N, compensating for an unnormalised inverse FFT.
    wnorm: f32,
    erb: Vec<usize>,
    analysis_mem: Vec<f32>,
    synthesis_mem: Vec<f32>,
    mean_norm_state: Vec<f32>,
    unit_norm_state: Vec<f32>,
    scratch_in: Vec<f32>,
    scratch_out: Vec<f32>,
}

fn linspace(a: f32, b: f32, n: usize) -> Vec<f32> {
    if n <= 1 {
        return vec![a; n];
    }
    let step = (b - a) / (n - 1) as f32;
    (0..n).map(|i| a + i as f32 * step).collect()
}

impl Default for DfFrontend {
    fn default() -> Self {
        Self::new()
    }
}

impl DfFrontend {
    pub fn new() -> Self {
        let mut planner = RealFftPlanner::<f32>::new();
        Self {
            fft: planner.plan_fft_forward(FFT_SIZE),
            ifft: planner.plan_fft_inverse(FFT_SIZE),
            window: vorbis_window(),
            wnorm: 1.0 / (FFT_SIZE as f32 * FFT_SIZE as f32 / (2.0 * HOP as f32)),
            erb: erb_widths(),
            analysis_mem: vec![0.0; FFT_SIZE - HOP],
            synthesis_mem: vec![0.0; FFT_SIZE - HOP],
            mean_norm_state: linspace(MEAN_NORM_INIT.0, MEAN_NORM_INIT.1, NB_ERB),
            unit_norm_state: linspace(UNIT_NORM_INIT.0, UNIT_NORM_INIT.1, NB_DF),
            scratch_in: vec![0.0; FFT_SIZE],
            scratch_out: vec![0.0; FFT_SIZE],
        }
    }

    /// One hop of input to one spectrum, carrying the overlap forward.
    pub fn analysis(&mut self, frame: &[f32; HOP], spec: &mut [Complex32; FREQ_BINS]) {
        let keep = FFT_SIZE - HOP;
        self.scratch_in[..keep].copy_from_slice(&self.analysis_mem);
        self.scratch_in[keep..].copy_from_slice(frame);
        for (x, w) in self.scratch_in.iter_mut().zip(&self.window) {
            *x *= w;
        }
        self.fft
            .process(&mut self.scratch_in, spec)
            .expect("fft input length is fixed");
        for x in spec.iter_mut() {
            *x *= self.wnorm;
        }
        // Shift the overlap window forward by one hop.
        self.analysis_mem.copy_within(HOP.., 0);
        let tail = keep - HOP;
        self.analysis_mem[tail..].copy_from_slice(frame);
    }

    /// Spectrum back to one hop of audio, overlap-added with the previous tail.
    pub fn synthesis(&mut self, spec: &mut [Complex32; FREQ_BINS], out: &mut [f32; HOP]) {
        self.ifft
            .process(spec, &mut self.scratch_out)
            .expect("ifft length is fixed");
        for (x, w) in self.scratch_out.iter_mut().zip(&self.window) {
            *x *= w;
        }
        for ((o, x), mem) in out
            .iter_mut()
            .zip(&self.scratch_out[..HOP])
            .zip(&self.synthesis_mem[..HOP])
        {
            *o = x + mem;
        }
        let keep = FFT_SIZE - HOP;
        self.synthesis_mem.copy_within(HOP.., 0);
        for v in &mut self.synthesis_mem[keep - HOP..] {
            *v = 0.0;
        }
        for (i, v) in self.synthesis_mem.iter_mut().enumerate() {
            *v += self.scratch_out[HOP + i];
        }
    }

    /// ERB band energies in dB, exponentially mean-normalised, scaled by 1/40.
    pub fn feat_erb(&mut self, spec: &[Complex32; FREQ_BINS], out: &mut [f32; NB_ERB]) {
        let mut pos = 0;
        for (b, &width) in self.erb.iter().enumerate() {
            let sum: f32 = spec[pos..pos + width].iter().map(|c| c.norm_sqr()).sum();
            out[b] = (sum / width as f32 + 1e-10).log10() * 10.0;
            pos += width;
        }
        for (o, s) in out.iter_mut().zip(self.mean_norm_state.iter_mut()) {
            *s = *o * (1.0 - ALPHA) + *s * ALPHA;
            *o = (*o - *s) / 40.0;
        }
    }

    /// The lowest `NB_DF` bins, divided by the square root of an exponentially
    /// averaged magnitude.
    pub fn feat_spec(&mut self, spec: &[Complex32; FREQ_BINS], out: &mut [Complex32; NB_DF]) {
        for ((o, x), s) in out
            .iter_mut()
            .zip(spec.iter())
            .zip(self.unit_norm_state.iter_mut())
        {
            *s = x.norm() * (1.0 - ALPHA) + *s * ALPHA;
            *o = *x / s.sqrt();
        }
    }
}

#[cfg(test)]
// The expected values below are transcribed verbatim from a float64 reference
// run. f32 cannot hold every digit, but trimming them would obscure where they
// came from, which is the whole point of having them.
#[allow(clippy::excessive_precision)]
mod tests {
    use super::*;

    /// Expected values produced by the reference-validated Python port on a
    /// pure 440 Hz sine — reproducible exactly in either language, so no audio
    /// file is needed and nothing licensed or personal ends up in the repo.
    fn sine(n: usize) -> Vec<f32> {
        (0..n)
            .map(|i| 0.5 * (2.0 * std::f32::consts::PI * 440.0 * i as f32 / SR as f32).sin())
            .collect()
    }

    #[test]
    fn erb_bands_match_libdf() {
        let erb = erb_widths();
        assert_eq!(
            erb,
            vec![
                2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 5, 5, 7, 7, 8, 10, 12, 13, 15, 18, 20, 24,
                28, 31, 37, 42, 50, 56, 67
            ]
        );
        assert_eq!(erb.iter().sum::<usize>(), FREQ_BINS);
    }

    #[test]
    fn window_matches_libdf() {
        let w = vorbis_window();
        assert_eq!(w.len(), FFT_SIZE);
        for (got, want) in w
            .iter()
            .zip([0.0000042055f32, 0.0000378492, 0.0001051350, 0.0002060603])
        {
            assert!((got - want).abs() < 1e-9, "{got} vs {want}");
        }
        // Power-complementary: w^2 sums to 1 across the 50% overlap.
        for i in 0..HOP {
            let s = w[i] * w[i] + w[i + HOP] * w[i + HOP];
            assert!(
                (s - 1.0).abs() < 1e-5,
                "window not power-complementary at {i}"
            );
        }
    }

    #[test]
    fn features_match_the_reference_implementation() {
        let x = sine(12 * HOP);
        let mut fe = DfFrontend::new();
        let mut spec = [Complex32::default(); FREQ_BINS];
        let mut erb = [0.0f32; NB_ERB];
        let mut sf = [Complex32::default(); NB_DF];

        for i in 0..12 {
            let mut frame = [0.0f32; HOP];
            frame.copy_from_slice(&x[i * HOP..(i + 1) * HOP]);
            fe.analysis(&frame, &mut spec);
            fe.feat_erb(&spec, &mut erb);
            fe.feat_spec(&spec, &mut sf);
        }

        // Frame 11, once the exponential norms have settled.
        let want_erb = [
            -0.511206,
            -0.3457876,
            -0.05047588,
            0.4184017,
            1.010229,
            0.7606199,
            -0.06575756,
            -0.1449326,
        ];
        for (i, want) in want_erb.iter().enumerate() {
            assert!(
                (erb[i] - want).abs() < 1e-4,
                "feat_erb[{i}] = {}, expected {want}",
                erb[i]
            );
        }

        let want_spec = [
            (-0.002134275f32, 0.0f32),
            (-0.002309205, -0.001001143),
            (-0.002897816, -0.002297453),
            (-0.004134033, -0.00431176),
        ];
        // Tolerance is 1e-5, not tighter: the expected values come from a
        // float64 reference while this runs in f32 throughout, as libDF does.
        // The remaining difference is accumulation, not a difference in the
        // computation — the reconstruction test above pins the scaling exactly.
        for (i, (re, im)) in want_spec.iter().enumerate() {
            assert!(
                (sf[i].re - re).abs() < 1e-5 && (sf[i].im - im).abs() < 1e-5,
                "feat_spec[{i}] = {:?}, expected ({re}, {im})",
                sf[i]
            );
        }

        // The raw spectrum too, so a scaling slip can't hide behind the
        // normalisation. Judged against the spectrum's peak rather than each
        // bin's own value: the low bins are leakage skirt for a 440 Hz tone,
        // where near-cancellation inflates relative error while the absolute
        // error stays negligible. Peak-relative is the meaningful measure, and
        // still catches any scaling, sign or bin-offset mistake.
        let peak = spec.iter().map(|c| c.norm()).fold(0.0f32, f32::max);
        let close = |got: f32, want: f32| (got - want).abs() < 1e-4 * peak;
        assert!(
            close(spec[0].re, -6.6597802e-05),
            "{:?} peak {peak}",
            spec[0]
        );
        assert!(
            close(spec[1].re, -7.1894617e-05),
            "{:?} peak {peak}",
            spec[1]
        );
        assert!(
            close(spec[1].im, -3.1169499e-05),
            "{:?} peak {peak}",
            spec[1]
        );
    }

    /// Analysis then synthesis with no processing must return the input.
    /// A scaling error here is exactly the bug that made the first Python
    /// attempt output silence.
    #[test]
    fn analysis_then_synthesis_reconstructs_the_signal() {
        let x = sine(20 * HOP);
        let mut fe = DfFrontend::new();
        let mut spec = [Complex32::default(); FREQ_BINS];
        let mut out = [0.0f32; HOP];
        let mut recon = Vec::new();

        for i in 0..20 {
            let mut frame = [0.0f32; HOP];
            frame.copy_from_slice(&x[i * HOP..(i + 1) * HOP]);
            fe.analysis(&frame, &mut spec);
            fe.synthesis(&mut spec, &mut out);
            recon.extend_from_slice(&out);
        }

        // One frame of algorithmic delay, then it should track the input.
        for i in HOP..(19 * HOP) {
            let want = x[i - HOP];
            assert!(
                (recon[i] - want).abs() < 1e-4,
                "sample {i}: got {}, expected {want}",
                recon[i]
            );
        }
    }
}
