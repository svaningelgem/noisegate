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

CI enforces a **minimum** line coverage, currently **68%**, just under the
70.2% actually reached. A pull request may raise that number; it must
never lower it. When coverage improves, bump `--fail-under-lines` in
`.github/workflows/ci.yml` **and** `.github/workflows/release.yml` in the same
PR — a tag must never publish something a pull request would have failed.

**The goal is 100%.** It is worth being honest about the distance:

| area | covered | why |
|---|---|---|
| `rnnoise.rs` | 100% | pure DSP |
| `log_format.rs` | 96% | the formatter runs against a buffer instead of a terminal |
| `devices.rs` | 94% | pure logic — device matching, priority, cable detection |
| `dsp/dfn_frontend.rs` | 92% | pure DSP, checked against `df.enhance.df_features()` |
| `event.rs` | 91% | the RAII handle wrapper, round-tripped |
| `config.rs` | 90% | load/save take a path, so tests never touch `%APPDATA%` |
| `pipeline.rs` | 83% | driven through a fake audio backend, including device selection |
| `error.rs` | 82% | HRESULT translation is a pure function |
| `offline.rs` | 81% | `--denoise` runs end to end on a generated WAV |
| `banner.rs` | 81% | the art has to fit 80 columns |
| `dsp/lib.rs` | 78% | backend selection |
| `mmcss.rs` | 77% | asks the OS scheduler for Pro Audio priority |
| `firstrun.rs` | 75% | the dialog *text* is testable; the message box is not |
| `dsp/onnx.rs` | 75% | runs a real session against `testdata/streaming_contract.onnx` |
| `format.rs` | 69% | mix-format validation is pure; the rejection paths are thin |
| `autostart.rs` | 68% | round-trips a scratch registry key, never the real one |
| `wasapi_capture.rs` | 67% | the pump runs against a scripted engine; only the COM setup is left |
| `wasapi_render.rs` | 66% | same |
| `tray.rs` | 52% | the watchdog, menu labels and icons; the event loop is not |
| `main.rs` | 48% | argument parsing and the single-instance lock; `real_main` is not |
| `console.rs` | 39% | the redirection-preserving branch runs; `AttachConsole` needs a parent console |
| **`dsp/tract.rs`** | **0%** | **the shipping backend, and none of it is tested — see below** |

`tract.rs` is the gap that matters. It is what the app actually runs, and every
line of it is uncovered: constructing it needs the real 8 MB model, which makes
it an integration test rather than a unit one. `models/dfn3_ours.tar.gz` is in
the repo, so nothing blocks writing it except the work.

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

## Things no test guards

Two fixes in this codebase have no test behind them, and it is worth saying so
rather than leaving a plausible-looking one in place.

**The tray event loop must keep sleeping between ticks.** `ControlFlow::wait_duration`
is `WaitUntil(now + d)` — a deadline, not a period — so it has to be re-armed on
every `about_to_wait`. Set once, the loop stops sleeping after the first tick and
a background app quietly pins a core. Checking this needs a real winit event loop
and a wall clock; the test would be slow, flaky, and the first one anyone
disables. The guard is the comment at the re-arm site.

**A fresh `Instant` must never have a `Duration` subtracted from it.** That panics
when uptime is below the offset, which is exactly when RoomMute starts, since it
runs at login. This cannot be tested by observation: `Instant` is opaque, has no
constructor, and the panic is only reachable on a machine that has genuinely just
booted — so a test asserting "the pipeline builds" passes everywhere and never
fails. `no_time_arithmetic_on_a_fresh_instant` in `main.rs` checks the property
that *is* checkable, by scanning the source for the pattern. Prefer
`Option<Instant>` (`None` = never) or `checked_sub`.

A regression test that has never been observed to fail is not a regression test.
When adding one for a bug already fixed, put the bug back, watch the test go red
for the *right reason*, then restore.

## Encoding a regression instead of describing one

`CaptureEngine` and `RenderEngine` exist so audio failures can be written down.
Neither can be reproduced on demand against real hardware — a device that
disappears when a dock is unplugged, an engine that signals and then has
nothing, a buffer flagged silent, an event that never fires — but each is a
short sequence of `Tick`s against the scripted engine in the test module.

When a bug report arrives, the first move is to express it as a script:

```rust
let (sink, result) = run(
    vec![
        Tick::Buffers(vec![buffer(FRAME_SAMPLES, 0.5)]),
        Tick::Invalidated,
    ],
    1,
    None,
);
```

The abstraction buys little coverage on its own. What it buys is that the next
report becomes a test rather than a comment.

The uncovered remainder is not untested-because-lazy; it is code whose entire
job is talking to Windows. Reaching 100% needs one of:

1. **COM setup coverage.** What is left in `wasapi_capture`/`wasapi_render` is
   the one-shot ceremony: activate the device, negotiate the mix format, create
   the event, start the client. It is a sequence of `unsafe` calls with almost
   no logic between them, and a fake would check our call ordering against our
   own assumptions rather than against WASAPI.
2. **Loopback tests on a machine with a virtual cable** — CI runners have no
   audio devices at all, so these could only run locally or on a self-hosted
   runner.
3. **Exercising the real model.** `dsp/tract.rs` needs
   `models/dfn3_ours.tar.gz`, which is in the repo — this one is only waiting
   on someone writing it.

The honest target is "everything that can be tested without hardware is
tested", and the ratchet is how that gets enforced.

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

`samples/demo_*` is committed: a 60-second mixture built from LibriSpeech and
DEMAND by `scripts/make_demo_sample.py`, byte-identical on any machine, plus
the MP3s and videos the README shows. `scripts/analyse_demo.py` measures it.
That is how a DSP change gets judged, and anyone can reproduce it.

Everything else in `samples/` is gitignored — real recordings of real people,
several MB each. Useful locally, never committed.

None of it runs in the automated suite: the sources are a 1.3 GB download.
