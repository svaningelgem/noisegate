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

CI enforces a **minimum** line coverage, currently **65%**, which is just under
the level already reached (66.9%). A pull request may raise that number; it must
never lower it. When coverage improves, bump `--fail-under-lines` in
`.github/workflows/ci.yml` **and** `.github/workflows/release.yml` in the same
PR — a tag must never publish something a pull request would have failed.

**The goal is 100%.** It is worth being honest about the distance:

| area | covered | why |
|---|---|---|
| `rnnoise.rs` | 100% | pure DSP |
| `log_format.rs` | 95% | the formatter runs against a buffer instead of a terminal |
| `dsp/dfn_frontend.rs` | 94% | pure DSP, checked against the reference implementation |
| `devices.rs` | 94% | pure logic — device matching, priority, cable detection |
| `autostart.rs` | 94% | round-trips through a real registry key |
| `config.rs` | 91% | load/save take a path, so tests never touch `%APPDATA%` |
| `pipeline.rs` | 85% | driven through a fake audio backend, including device selection |
| `dsp/lib.rs` | 85% | backend selection |
| `error.rs` | 82% | HRESULT translation is a pure function |
| `offline.rs` | 81% | `--denoise` runs end to end on a generated WAV |
| `banner.rs` | 81% | the art has to fit 80 columns |
| `format.rs` | 80% | mix-format validation is pure |
| `dsp/onnx.rs` | 76% | runs a real session against `testdata/streaming_contract.onnx` |
| `mmcss.rs` | 77% | asks the OS scheduler for Pro Audio priority |
| `firstrun.rs` | 75% | the dialog *text* is testable; the message box is not |
| `wasapi_capture.rs` | 46% | the resamplers and frame accumulator; the COM loop is not |
| `wasapi_render.rs` | 45% | same — `UpConverter` is covered, the engine calls are not |
| `tray.rs` | 47% | the watchdog, menu labels and icons; the event loop is not |
| `main.rs` | 44% | argument parsing and the single-instance lock; `real_main` is not |
| `console.rs` | 39% | the redirection-preserving branch runs; `AttachConsole` needs a parent console |

## The ONNX tests

`crates/dsp/testdata/streaming_contract.onnx` is a 700-byte model that
implements the loader's contract and nothing else: three named inputs, a state
tensor whose width has to be read from the model, and two positional outputs.
It is deliberately not an identity — it returns `input * (states[0] + 1)` and
increments the state — so a loader that forgets to feed the state back fails
with a wrong sample value instead of quietly passing audio through.

Regenerate it with `python scripts/make_test_model.py <path>` if the expected
signature ever changes.

These tests need `onnxruntime.dll`, which is not in the source tree. Both
workflows download it into `target/debug` before running tests. Locally they
skip with a message if it is missing; when `CI` is set they fail instead, so
they cannot silently stop running while the build stays green.

The uncovered majority is not untested-because-lazy; it is code whose entire
job is talking to Windows. Reaching 100% needs one of:

1. **More hardware abstraction.** `Pipeline` already takes an `AudioIo` trait,
   so the ring buffers, DSP thread, bypass and shutdown paths are all tested
   against a fake device. The same treatment for `wasapi_capture`/
   `wasapi_render` internals is the remaining chunk, and a larger one — those
   modules are mostly `unsafe` COM calls with little logic to separate out.
2. **Loopback tests on a machine with a virtual cable** — CI runners have no
   audio devices at all, so these could only run locally or on a self-hosted
   runner.
3. **A checked-in ONNX model** — would cover `dsp/onnx.rs`. Redistributing the
   DeepFilterNet weights is not ours to do, which is why we are training our
   own; once those land in the repo this stops being a blocker.

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
