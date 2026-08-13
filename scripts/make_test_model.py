#!/usr/bin/env python3
"""Generate the tiny ONNX model the dsp tests run against.

The real denoiser weights are someone else's work and are far too large to
check in, but the loader in `crates/dsp/src/onnx.rs` has a contract of its own
worth testing: input names, a state tensor whose width is read from the model,
an optional attenuation input, and outputs in a fixed order. This model
implements that contract and nothing else, so a failure points at the loader
rather than at a denoiser.

It is deliberately *not* an identity: the output depends on the incoming state
and the state increments on every call, so a loader that forgets to feed the
state back shows up as a wrong sample value rather than as a silent pass.

    enhanced   = input_frame * (states[0] + 1) + atten_lim_db
    new_states = states + 1

Regenerate with:

    python scripts/make_test_model.py crates/dsp/testdata/streaming_contract.onnx

Needs torch and onnx; any recent versions will do. The checked-in file is a
couple of kilobytes and rarely needs regenerating — only if the loader's
expected signature changes.
"""

import sys
from pathlib import Path

import torch

FRAME_SAMPLES = 480
STATE_LEN = 4


class StreamingContract(torch.nn.Module):
    def forward(self, input_frame, states, atten_lim_db):
        enhanced = input_frame * (states[0] + 1.0) + atten_lim_db
        new_states = states + 1.0
        return enhanced, new_states


def main() -> None:
    out = Path(sys.argv[1] if len(sys.argv) > 1 else "streaming_contract.onnx")
    out.parent.mkdir(parents=True, exist_ok=True)

    torch.onnx.export(
        StreamingContract(),
        (
            torch.zeros(FRAME_SAMPLES),
            torch.zeros(STATE_LEN),
            torch.zeros(1),
        ),
        str(out),
        input_names=["input_frame", "states", "atten_lim_db"],
        output_names=["enhanced_audio", "new_states"],
        # Fixed shapes on purpose: the loader reads the state width from the
        # model, so the dimensions have to be declared rather than dynamic.
        dynamic_axes=None,
        opset_version=17,
    )
    print(f"wrote {out} ({out.stat().st_size} bytes)")


if __name__ == "__main__":
    main()
