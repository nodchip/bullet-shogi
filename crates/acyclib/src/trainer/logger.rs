use std::{
    fmt::Display,
    io::{Write, stdout},
    sync::atomic::{AtomicBool, Ordering::SeqCst},
    time::Instant,
};

use super::schedule::TrainingSteps;

static CBCS: AtomicBool = AtomicBool::new(false);
static COLOURS_ENABLED: AtomicBool = AtomicBool::new(true);

pub fn ansi<T: Display, U: Display>(x: T, y: U) -> String {
    if COLOURS_ENABLED.load(SeqCst) { format!("\x1b[{y}m{x}\x1b[0m{}", esc()) } else { x.to_string() }
}

pub fn set_colour<U: Display>(x: U) {
    if COLOURS_ENABLED.load(SeqCst) {
        print!("\x1b[{x}m");
    }
}

pub fn clear_colours() {
    print!("{}", esc());
}

pub fn set_colours_enabled(val: bool) {
    COLOURS_ENABLED.store(val, SeqCst)
}

pub fn set_cbcs(val: bool) {
    CBCS.store(val, SeqCst)
}

pub fn num_cs() -> i32 {
    if CBCS.load(SeqCst) { 35 } else { 36 }
}

fn esc() -> &'static str {
    if COLOURS_ENABLED.load(SeqCst) && CBCS.load(SeqCst) { "\x1b[38;5;225m" } else { "" }
}

fn progress_line_suffix() -> &'static str {
    if COLOURS_ENABLED.load(SeqCst) { "\x1b[F" } else { "" }
}

pub fn report_superbatch_progress(
    superbatch: usize,
    batches: usize,
    finished_batches: usize,
    superbatch_timer: &Instant,
    superbatch_positions: usize,
) {
    let num_cs = num_cs();
    let superbatch_time = superbatch_timer.elapsed().as_secs_f32();
    let pct = finished_batches as f32 / batches as f32;
    let pos_per_sec = superbatch_positions as f32 / superbatch_time;

    let seconds = superbatch_time / pct - superbatch_time;

    print!(
        "superbatch {} [{}% ({}/{} batches, {} pos/sec)]\n\
        Estimated time to end of superbatch: {}s     {}",
        ansi(superbatch, num_cs),
        ansi(format!("{:.1}", pct * 100.0), 35),
        ansi(finished_batches, num_cs),
        ansi(batches, num_cs),
        ansi(format!("{pos_per_sec:.0}"), num_cs),
        ansi(format!("{seconds:.1}"), num_cs),
        progress_line_suffix(),
    );
    let _ = stdout().flush();
}

pub fn report_superbatch_finished(
    superbatch: usize,
    error: f32,
    superbatch_time: f32,
    total_time: f32,
    positions: usize,
) {
    let num_cs = num_cs();
    let pos_per_sec = positions as f32 / superbatch_time;

    println!(
        "superbatch {} | time {}s | running loss {} | {} pos/sec | total time {}s",
        ansi(superbatch, num_cs),
        ansi(format!("{superbatch_time:.1}"), num_cs),
        ansi(format!("{error:.6}"), num_cs),
        ansi(format!("{pos_per_sec:.0}"), num_cs),
        ansi(format!("{total_time:.1}"), num_cs),
    );
}

pub fn report_time_left(steps: TrainingSteps, superbatch: usize, total_time: f32) {
    let num_cs = num_cs();
    let finished_superbatches = superbatch - steps.start_superbatch + 1;
    let total_superbatches = steps.end_superbatch - steps.start_superbatch + 1;
    let pct = finished_superbatches as f32 / total_superbatches as f32;
    let time_left = total_time / pct - total_time;

    let (hours, minutes, seconds) = seconds_to_hms(time_left as u32);

    println!(
        "Estimated time remaining in training: {}h {}m {}s",
        ansi(hours, num_cs),
        ansi(minutes, num_cs),
        ansi(seconds, num_cs),
    );
}

pub fn seconds_to_hms(mut seconds: u32) -> (u32, u32, u32) {
    let mut minutes = seconds / 60;
    let hours = minutes / 60;
    seconds -= minutes * 60;
    minutes -= hours * 60;

    (hours, minutes, seconds)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn ansi_omits_escape_sequences_when_colours_are_disabled() {
        set_colours_enabled(false);

        assert_eq!(ansi("value", 31), "value");

        set_colours_enabled(true);
    }
}
