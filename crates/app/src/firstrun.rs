//! First-run conversations: the two things NoiseGate can't do for the user.
//!
//! Both are deliberately plain-language. Someone installing a noise-cancelling
//! app has no reason to know what a virtual audio endpoint is, and telling
//! them "no virtual cable found" is the kind of message that gets an app
//! uninstalled.

use tracing::{info, warn};

/// Vendor page rather than a direct installer link: the licence terms and the
/// current signed build both live there.
pub const CABLE_URL: &str = "https://vb-audio.com/Cable/";

/// Where the ONNX model comes from, shown before anything is fetched.
pub const MODEL_SOURCE_URL: &str = "https://github.com/Rikorose/DeepFilterNet";

#[derive(Debug, PartialEq, Eq)]
pub enum CableChoice {
    OpenSite,
    Close,
}

/// Explain the missing cable without jargon, and offer the only two useful
/// actions.
///
/// "Close" must also switch off start-with-Windows. Otherwise someone who
/// isn't ready to install a driver gets this dialog on every single boot,
/// which is a popup loop they can only escape through the registry.
pub fn cable_missing() -> CableChoice {
    let text = "NoiseGate cleans up your microphone, but it needs one more \
free program to pass the cleaned sound on to Zoom, Teams, Discord and so on.\n\n\
That program is called VB-Cable. It acts like a second microphone that other \
apps can listen to. Without it, NoiseGate has nowhere to send your voice.\n\n\
It takes about two minutes to install and it's free.\n\n\
    Yes  —  open the VB-Cable download page\n\
    No   —  close NoiseGate for now\n\n\
(Choosing No also stops NoiseGate starting with Windows, so you won't see \
this message every time you switch on your PC.)";

    if crate::message_box_yes_no(text) {
        CableChoice::OpenSite
    } else {
        CableChoice::Close
    }
}

/// Ask before using the high-quality model, naming where it comes from.
///
/// Shown whenever the model is deliberately chosen, not once and remembered:
/// it's a licence acceptance, and it's a decision the user just made, so it
/// isn't a loop.
pub fn model_licence() -> bool {
    let text = format!(
        "The high-quality noise removal uses DeepFilterNet3, a speech model \
published by its authors at:\n\n    {MODEL_SOURCE_URL}\n\n\
It is a separate work from NoiseGate, with its own licence terms. NoiseGate \
does not include or redistribute it — you obtain it from the source above and \
accept those terms yourself.\n\n\
    Yes  —  I accept the model's terms and want to use it\n\
    No   —  stay on the simpler built-in noise removal\n\n\
You can change this later from the tray menu."
    );
    let accepted = crate::message_box_yes_no(&text);
    info!(accepted, "model licence prompt answered");
    accepted
}

/// Tell the user where to put the model, since we can't fetch it for them yet.
pub fn model_missing(expected: &std::path::Path) {
    warn!(path = %expected.display(), "ONNX model not present");
    crate::message_box(&format!(
        "The high-quality model isn't installed yet.\n\n\
         Download the DeepFilterNet3 ONNX export from:\n\n    {}\n\n\
         and save it as:\n\n    {}\n\n\
         NoiseGate will use it the next time you pick it from the Denoiser menu. \
         Until then it stays on the built-in noise removal.",
        MODEL_SOURCE_URL,
        expected.display()
    ));
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn links_point_where_we_say_they_do() {
        // A typo sends people somewhere arbitrary to download software.
        assert_eq!(CABLE_URL, "https://vb-audio.com/Cable/");
        assert!(MODEL_SOURCE_URL.starts_with("https://github.com/Rikorose/"));
        for url in [CABLE_URL, MODEL_SOURCE_URL] {
            assert!(url.starts_with("https://"), "{url} must be https");
        }
    }
}
