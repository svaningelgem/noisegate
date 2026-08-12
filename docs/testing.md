# Testing and coverage

## Running things locally

```powershell
cargo fmt --all -- --check          # CI gates on this
cargo clippy --all-targets --all-features
cargo test --all-features
cargo llvm-cov --all-features --summary-only
```

`cargo llvm-cov` needs the `llvm-tools-preview` component:

```powershell
rustup component add llvm-tools-preview
cargo install cargo-llvm-cov
```

## The coverage ratchet

CI enforces a **minimum** line coverage, currently **29%**, which is the level
already reached. A pull request may raise that number; it must never lower it.
When coverage improves, bump `--fail-under-lines` in
`.github/workflows/ci.yml` in the same PR.

**The goal is 100%.** It is worth being honest about the distance:

| area | covered | why |
|---|---|---|
| `devices.rs` | 89% | pure logic — device matching, priority, cable detection |
| `error.rs` | 82% | HRESULT translation is a pure function |
| `format.rs` | 76% | mix-format validation is pure |
| `tray.rs` | 26% | labels and icons are testable; the event loop is not |
| `wasapi_capture.rs` | 0% | needs a real microphone and the audio engine |
| `wasapi_render.rs` | 0% | needs a real endpoint |
| `pipeline.rs` | 0% | starts three threads against live devices |
| `dsp/onnx.rs` | 0% | needs `onnxruntime.dll` and a model file |
| `mmcss.rs` | 0% | asks the OS scheduler for Pro Audio priority |

The uncovered majority is not untested-because-lazy; it is code whose entire
job is talking to Windows. Reaching 100% needs one of:

1. **A hardware abstraction** — put a trait in front of WASAPI so the capture
   and render loops can run against a fake device in tests. This is the real
   answer, and it is a substantial refactor of `audio-io`.
2. **Loopback tests on a machine with a virtual cable** — CI runners have no
   audio devices at all, so these could only run locally or on a self-hosted
   runner.
3. **A checked-in ONNX model** — would cover `dsp/onnx.rs`, but the licence
   for the DeepFilterNet weights is unresolved (see the project README).

Until one of those lands, the honest target is "everything that can be tested
without hardware is tested", and the ratchet is how that gets enforced.

## What is deliberately covered

The tests concentrate on the things that were actually wrong at some point, or
that would fail silently:

- **Mix-format validation** — including the 16-bit case that would have caused
  an out-of-bounds read in capture and an overrun in render.
- **Device matching** — cable lookalikes, the `CABLE In 16ch` sibling endpoint,
  the capture side of a cable (which would feed the pipeline into itself), and
  instance prefixes (`4- fifine`) that change on replug.
- **Microphone priority** — falling through to the next device when the
  preferred one is unplugged, with the Windows default as an unremovable floor.
- **Error translation** — that known HRESULTs read as English and unmapped ones
  never render a dangling colon.
- **Icon states** — that on/off differ in hue rather than only brightness, since
  a brightness-only difference is invisible at tray size.
- **Autostart** — a real registry round-trip that restores whatever was there.

## Fixtures

`samples/` (gitignored) holds real recordings used to compare denoiser
backends, with a README describing provenance and the commands to regenerate
the processed versions. They are not part of the automated suite — they contain
real voices and are several MB each — but they are how any DSP change gets
judged.
