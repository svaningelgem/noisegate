//! Custom tracing-subscriber event formatter. Emits each log line in green
//! ANSI, matching the banner. Errors and warnings get a brighter green +
//! a small glyph so they stand out without breaking the theme.

use std::fmt;
use std::time::Instant;

use tracing::{Event, Level, Subscriber};
use tracing_subscriber::fmt::{format::Writer, FmtContext, FormatEvent, FormatFields};
use tracing_subscriber::registry::LookupSpan;

const GREEN: &str = "\x1b[32m";
const BRIGHT_GREEN: &str = "\x1b[1;92m";
const DIM_GREEN: &str = "\x1b[2;32m";
const YELLOW: &str = "\x1b[33m";
const RED: &str = "\x1b[31m";
const RESET: &str = "\x1b[0m";

/// Green-themed event formatter. Stateless apart from a process-relative
/// "uptime" reference so log timestamps look like `[12.345s]` instead of
/// the full system clock.
pub struct GreenFormat {
    start: Instant,
}

impl GreenFormat {
    pub fn new() -> Self {
        Self {
            start: Instant::now(),
        }
    }
}

impl<S, N> FormatEvent<S, N> for GreenFormat
where
    S: Subscriber + for<'a> LookupSpan<'a>,
    N: for<'a> FormatFields<'a> + 'static,
{
    fn format_event(
        &self,
        ctx: &FmtContext<'_, S, N>,
        mut writer: Writer<'_>,
        event: &Event<'_>,
    ) -> fmt::Result {
        let meta = event.metadata();
        let elapsed = self.start.elapsed();
        let secs = elapsed.as_secs_f64();

        let (level_color, glyph) = match *meta.level() {
            Level::ERROR => (RED, "✗"),
            Level::WARN => (YELLOW, "!"),
            Level::INFO => (BRIGHT_GREEN, "›"),
            Level::DEBUG => (GREEN, "·"),
            Level::TRACE => (DIM_GREEN, "."),
        };

        write!(
            writer,
            "{DIM_GREEN}[{:>7.3}s]{RESET} {}{} {:<5}{RESET} {GREEN}",
            secs,
            level_color,
            glyph,
            meta.level().as_str(),
        )?;
        ctx.field_format().format_fields(writer.by_ref(), event)?;
        writeln!(writer, "{RESET}")
    }
}

#[cfg(test)]
mod tests {
    use std::sync::{Arc, Mutex};

    use tracing_subscriber::fmt::MakeWriter;
    use tracing_subscriber::prelude::*;

    use super::*;

    /// Collects everything the subscriber writes so the formatted line can be
    /// asserted on.
    #[derive(Clone, Default)]
    struct Buffer(Arc<Mutex<Vec<u8>>>);

    impl std::io::Write for Buffer {
        fn write(&mut self, buf: &[u8]) -> std::io::Result<usize> {
            self.0.lock().unwrap().extend_from_slice(buf);
            Ok(buf.len())
        }
        fn flush(&mut self) -> std::io::Result<()> {
            Ok(())
        }
    }

    impl<'a> MakeWriter<'a> for Buffer {
        type Writer = Self;
        fn make_writer(&'a self) -> Self::Writer {
            self.clone()
        }
    }

    /// Run `f` with our formatter installed and return what it printed.
    fn captured(f: impl FnOnce()) -> String {
        let buf = Buffer::default();
        let layer = tracing_subscriber::fmt::layer()
            .with_ansi(false)
            .event_format(GreenFormat::new())
            .with_writer(buf.clone());
        tracing::subscriber::with_default(tracing_subscriber::registry().with(layer), f);
        let bytes = buf.0.lock().unwrap().clone();
        String::from_utf8(bytes).expect("formatter must emit valid UTF-8")
    }

    #[test]
    fn a_line_carries_the_level_the_message_and_an_uptime() {
        let out = captured(|| tracing::error!(device = "Yeti", "capture failed"));

        assert!(out.contains("ERROR"), "{out:?}");
        assert!(out.contains("capture failed"), "{out:?}");
        assert!(out.contains("device"), "fields must survive: {out:?}");
        assert!(out.contains("s]"), "expected an uptime stamp: {out:?}");
        assert!(out.ends_with('\n'), "one line per event");
    }

    /// Every level must be visually distinct — the point of the glyphs is
    /// spotting a warning in a wall of green.
    #[test]
    fn each_level_gets_its_own_glyph() {
        let lines = [
            captured(|| tracing::error!("x")),
            captured(|| tracing::warn!("x")),
            captured(|| tracing::info!("x")),
            captured(|| tracing::debug!("x")),
            captured(|| tracing::trace!("x")),
        ];
        let glyphs: Vec<&str> = lines
            .iter()
            .map(|l| {
                ["✗", "!", "›", "·", "."]
                    .into_iter()
                    .find(|g| l.contains(g))
                    .unwrap_or("?")
            })
            .collect();
        assert_eq!(glyphs, ["✗", "!", "›", "·", "."], "lines were {lines:#?}");
    }

    /// Colour has to be closed off, or the escape sequence bleeds into
    /// whatever the terminal prints next.
    #[test]
    fn colour_is_always_reset_before_the_newline() {
        let out = captured(|| tracing::info!("tinted"));
        assert!(out.starts_with(DIM_GREEN), "{out:?}");
        assert!(out.trim_end().ends_with(RESET), "{out:?}");
    }
}
