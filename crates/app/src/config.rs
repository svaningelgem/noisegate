use std::path::PathBuf;

use serde::{Deserialize, Serialize};

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Config {
    /// Microphone to capture from, by the name Windows shows. Empty = follow
    /// the Windows default.
    ///
    /// A name rather than an endpoint id: ids are opaque GUIDs that nobody can
    /// edit by hand, and they don't survive replugging the device — Windows
    /// re-enumerates it with a fresh one. The name is stable, readable, and
    /// what the tray menu shows. Resolving name to id is our problem, not the
    /// config file's.
    #[serde(default, alias = "input_device_name")]
    pub input_device: String,
    /// Microphones in order of preference, by name, best first.
    ///
    /// A list rather than one choice because devices come and go: unplug the
    /// USB mic mid-call and the next one down takes over instead of the app
    /// stopping to ask. Windows' own default is always the final fallback, so
    /// the list can never strand you.
    #[serde(default)]
    pub microphones: Vec<String>,
    /// Where cleaned audio goes, by name. Empty = auto-detect a virtual cable.
    #[serde(default)]
    pub output_device: String,
    /// Master enable. When false, the pipeline runs in bypass mode (passes
    /// audio through without DSP) so toggling is instant.
    #[serde(default = "default_true")]
    pub enabled: bool,
    /// Attenuation limit in dB for DeepFilterNet. 6.0 = subtle, 100.0 = max.
    #[serde(default = "default_atten")]
    pub attenuation_db: f32,
    /// Auto-start at user login.
    #[serde(default)]
    pub auto_start: bool,
    /// Path to an ONNX denoise model. Only honoured by builds made with
    /// `--features onnx`.
    #[serde(default)]
    pub model_path: String,
    /// Run the ONNX model instead of the built-in RNNoise. On by default: on
    /// competing speech DeepFilterNet3 removes ~19 dB where RNNoise removes
    /// ~0.5 dB, which is the difference between usable and not. Falls back to
    /// RNNoise on its own if no model is present.
    #[serde(default = "default_true")]
    pub use_onnx: bool,
}

fn default_true() -> bool {
    true
}
fn default_atten() -> f32 {
    100.0
}

impl Default for Config {
    fn default() -> Self {
        Self {
            input_device: String::new(),
            microphones: Vec::new(),
            output_device: String::new(),
            enabled: true,
            attenuation_db: default_atten(),
            auto_start: false,
            model_path: String::new(),
            use_onnx: true,
        }
    }
}

impl Config {
    pub fn load_or_default() -> Self {
        match Self::load() {
            Ok(c) => c,
            Err(e) => {
                tracing::warn!(error = %e, "config load failed; using defaults");
                Self::default()
            }
        }
    }

    pub fn load() -> anyhow::Result<Self> {
        let path = config_path();
        if !path.exists() {
            return Ok(Self::default());
        }
        let text = std::fs::read_to_string(&path)?;
        let mut cfg: Self = toml::from_str(&text)?;
        // Fold a single legacy choice into the priority list.
        if cfg.microphones.is_empty() && !cfg.input_device.is_empty() {
            cfg.microphones.push(std::mem::take(&mut cfg.input_device));
        }
        Ok(cfg)
    }

    /// Move `name` to the front, keeping the rest in order and capping the
    /// list so it doesn't grow forever with devices seen once.
    pub fn prefer_microphone(&mut self, name: &str) {
        const MAX_REMEMBERED: usize = 5;
        self.microphones.retain(|m| m != name);
        self.microphones.insert(0, name.to_string());
        self.microphones.truncate(MAX_REMEMBERED);
    }

    pub fn save(&self) -> anyhow::Result<()> {
        let path = config_path();
        if let Some(parent) = path.parent() {
            std::fs::create_dir_all(parent)?;
        }
        let text = toml::to_string_pretty(self)?;
        std::fs::write(&path, text)?;
        Ok(())
    }
}

impl Config {
    /// The ONNX model to offer in the tray, if there is one: whatever
    /// `model_path` points at, else a `model.onnx` sitting next to the
    /// executable so "drop the file in and pick it" works with no config
    /// editing at all.
    pub fn available_model(&self) -> Option<PathBuf> {
        if !self.model_path.is_empty() {
            let p = PathBuf::from(&self.model_path);
            return p.exists().then_some(p);
        }
        let beside = std::env::current_exe().ok()?.parent()?.join("model.onnx");
        beside.exists().then_some(beside)
    }

    /// The model to actually load, honouring the on/off switch.
    pub fn active_model(&self) -> Option<PathBuf> {
        self.use_onnx.then(|| self.available_model()).flatten()
    }
}

pub fn config_path() -> PathBuf {
    base_dir().join("config.toml")
}

pub fn log_dir() -> PathBuf {
    base_dir().join("logs")
}

fn base_dir() -> PathBuf {
    dirs::config_dir()
        .unwrap_or_else(|| PathBuf::from("."))
        .join("NoiseGate")
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn preferring_a_microphone_moves_it_to_the_front() {
        let mut c = Config::default();
        c.prefer_microphone("A");
        c.prefer_microphone("B");
        assert_eq!(c.microphones, vec!["B", "A"]);

        // Choosing an existing one promotes rather than duplicates.
        c.prefer_microphone("A");
        assert_eq!(c.microphones, vec!["A", "B"]);
    }

    #[test]
    fn the_remembered_list_does_not_grow_without_bound() {
        let mut c = Config::default();
        for i in 0..12 {
            c.prefer_microphone(&format!("mic {i}"));
        }
        assert_eq!(c.microphones.len(), 5);
        assert_eq!(c.microphones[0], "mic 11", "most recent first");
    }

    /// The CLI override and the tray both have to write where the pipeline
    /// reads. Writing to the legacy single field made --mic a no-op.
    #[test]
    fn a_legacy_single_choice_becomes_the_first_preference() {
        let toml = r#"
            input_device = "Microphone (fifine Microphone)"
            enabled = true
        "#;
        let mut cfg: Config = toml::from_str(toml).unwrap();
        assert!(cfg.microphones.is_empty(), "not migrated until load() runs");

        // Same fold that Config::load performs.
        if cfg.microphones.is_empty() && !cfg.input_device.is_empty() {
            cfg.microphones.push(std::mem::take(&mut cfg.input_device));
        }
        assert_eq!(cfg.microphones, vec!["Microphone (fifine Microphone)"]);
    }

    #[test]
    fn an_onnx_model_is_only_active_when_switched_on() {
        let mut c = Config {
            model_path: "nonexistent.onnx".into(),
            use_onnx: false,
            ..Config::default()
        };
        assert!(c.active_model().is_none(), "off means off");
        // With use_onnx on, a path that does not exist still yields nothing,
        // so the pipeline falls back to RNNoise rather than failing.
        c.use_onnx = true;
        assert!(c.active_model().is_none());
    }
}
