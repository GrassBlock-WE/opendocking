// SPDX-License-Identifier: GPL-3.0-or-later
//! Cooperative cancellation, pause and resume for long-running searches.
//!
//! A docking run is a tight, CPU-bound loop that may take minutes on a large
//! flexible ligand, and the desktop workbench (module D, "任务调度与执行监控")
//! must be able to **pause**, **resume** and **abort** it from the GUI thread.
//! Killing the worker threads is not an option — it would leak the `rayon` pool
//! state and lose every pose found so far — so the search polls a shared,
//! lock-free token at every generation / step boundary and unwinds normally.
//!
//! # Design
//!
//! * A single [`AtomicU8`] holds the state, so `pause()` / `resume()` /
//!   `cancel()` are wait-free and can be called from a signal handler, a GUI
//!   callback or another Python thread.
//! * Cloning a [`CancelToken`] is an `Arc` bump: every search worker of every
//!   `rayon` task sees the very same word.
//! * The token is **latched**: once [`CancelState::Cancelled`] it can never go
//!   back to `Running` by accident. `pause()` and `resume()` only move between
//!   `Running` and `Paused`, so a cancel racing with a resume always wins.
//! * A paused worker sleeps (1 ms by default) instead of spinning, so a paused
//!   search costs no measurable CPU.
//!
//! # Contract for search code
//!
//! ```no_run
//! # use dock_core::cancel::CancelToken;
//! # fn search(token: &CancelToken) {
//! for _generation in 0..1000 {
//!     if token.checkpoint() {
//!         break; // cancelled: return the best pose found so far
//!     }
//!     // ... one generation of work ...
//! }
//! # }
//! ```
//!
//! [`checkpoint`](CancelToken::checkpoint) blocks while the token is paused, so
//! "paused" really means "does not advance" rather than "busy-waits", and it
//! returns as soon as the token is cancelled. Everything is safe Rust.

use std::sync::atomic::{AtomicU8, Ordering};
use std::sync::Arc;
use std::time::Duration;

/// How long a paused worker sleeps between polls of the token.
pub const DEFAULT_POLL_INTERVAL: Duration = Duration::from_millis(1);

/// The state of a [`CancelToken`].
///
/// The discriminants are the values stored in the atomic word; the ordering
/// (`Running < Paused < Cancelled`) is not used for decisions, but the explicit
/// values keep the mapping obvious and stable.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, PartialOrd, Ord)]
#[repr(u8)]
pub enum CancelState {
    /// The search is (or may start) doing work.
    Running = 0,
    /// The search must not advance; it waits until it is resumed or cancelled.
    Paused = 1,
    /// Terminal: the search must stop and return what it has.
    Cancelled = 2,
}

impl CancelState {
    /// Decode the raw atomic value. Unknown values are treated as `Running`
    /// (fail open: a corrupt word must never silently kill a run).
    #[inline]
    pub fn from_raw(value: u8) -> CancelState {
        match value {
            1 => CancelState::Paused,
            2 => CancelState::Cancelled,
            _ => CancelState::Running,
        }
    }

    /// Lower-case name, used by the CLI / Python / GUI.
    pub fn name(self) -> &'static str {
        match self {
            CancelState::Running => "running",
            CancelState::Paused => "paused",
            CancelState::Cancelled => "cancelled",
        }
    }
}

impl std::fmt::Display for CancelState {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str(self.name())
    }
}

/// A cheap, clonable, thread-safe pause / cancel flag.
///
/// All clones share one state word; the token is `Send + Sync` and can be
/// handed to every worker of a `rayon` pool.
#[derive(Debug, Clone)]
pub struct CancelToken {
    state: Arc<AtomicU8>,
}

impl Default for CancelToken {
    fn default() -> Self {
        CancelToken::new()
    }
}

impl CancelToken {
    /// A fresh, running token.
    pub fn new() -> CancelToken {
        CancelToken {
            state: Arc::new(AtomicU8::new(CancelState::Running as u8)),
        }
    }

    /// The current state (an `Acquire` load).
    #[inline]
    pub fn state(&self) -> CancelState {
        CancelState::from_raw(self.state.load(Ordering::Acquire))
    }

    /// `true` when the search must stop.
    #[inline]
    pub fn is_cancelled(&self) -> bool {
        self.state() == CancelState::Cancelled
    }

    /// `true` while the token is paused.
    #[inline]
    pub fn is_paused(&self) -> bool {
        self.state() == CancelState::Paused
    }

    /// `true` when the token is neither paused nor cancelled.
    #[inline]
    pub fn is_running(&self) -> bool {
        self.state() == CancelState::Running
    }

    /// Request a pause.
    ///
    /// Only a running token can be paused: pausing a cancelled token is a
    /// no-op, so a `cancel()` racing with a `pause()` always wins. Returns
    /// `true` when the token is now paused.
    pub fn pause(&self) -> bool {
        self.state
            .compare_exchange(
                CancelState::Running as u8,
                CancelState::Paused as u8,
                Ordering::AcqRel,
                Ordering::Acquire,
            )
            .is_ok()
    }

    /// Resume a paused token.
    ///
    /// Never resurrects a cancelled token. Returns `true` when the token is now
    /// running.
    pub fn resume(&self) -> bool {
        self.state
            .compare_exchange(
                CancelState::Paused as u8,
                CancelState::Running as u8,
                Ordering::AcqRel,
                Ordering::Acquire,
            )
            .is_ok()
    }

    /// Cancel the search. Idempotent, unconditional, and always wins over a
    /// concurrent `pause()` / `resume()`.
    pub fn cancel(&self) {
        // A plain store is enough: the state is a single word and `Cancelled`
        // is terminal, so it can only be overwritten by another `cancel()`.
        self.state.store(CancelState::Cancelled as u8, Ordering::Release);
    }

    /// Re-arm a cancelled token so that the same [`crate::docking::Docking`]
    /// object can be run again.
    ///
    /// This deliberately does **not** touch a running or paused token, so it is
    /// safe to call at the start of every run. Returns `true` when the token
    /// had been cancelled.
    pub fn rearm(&self) -> bool {
        self.state
            .compare_exchange(
                CancelState::Cancelled as u8,
                CancelState::Running as u8,
                Ordering::AcqRel,
                Ordering::Acquire,
            )
            .is_ok()
    }

    /// Non-blocking poll: `true` only when the token is cancelled.
    ///
    /// Use this in inner loops that must not sleep (e.g. while holding a spin
    /// lock); use [`CancelToken::checkpoint`] everywhere else so that a pause
    /// is honoured.
    #[inline]
    pub fn should_stop(&self) -> bool {
        self.is_cancelled()
    }

    /// Cooperative checkpoint: blocks while the token is paused, returns `true`
    /// when the search must unwind.
    ///
    /// The call is a single atomic load in the common case, so it is cheap
    /// enough for the innermost search loop.
    #[inline]
    pub fn checkpoint(&self) -> bool {
        self.checkpoint_with(DEFAULT_POLL_INTERVAL)
    }

    /// [`CancelToken::checkpoint`] with an explicit poll interval.
    ///
    /// A paused worker sleeps in `poll` steps rather than spinning, which is
    /// what keeps a paused search at ~0 % CPU; the sleep bounds the latency
    /// with which a resume (or a cancel) is noticed.
    pub fn checkpoint_with(&self, poll: Duration) -> bool {
        loop {
            match self.state() {
                CancelState::Running => return false,
                CancelState::Cancelled => return true,
                CancelState::Paused => std::thread::sleep(poll),
            }
        }
    }

    /// Block until the token is running again or cancelled; returns `true` when
    /// it was cancelled.
    pub fn wait_until_running(&self) {
        let _ = self.checkpoint();
    }
}

/// `true` when `option` asks the search to stop.
///
/// The search code carries `Option<&CancelToken>` so that the original public
/// signatures keep working; this helper keeps the call sites a one-liner.
#[inline]
pub fn stopped(token: Option<&CancelToken>) -> bool {
    match token {
        Some(t) => t.checkpoint(),
        None => false,
    }
}

/// `true` when `option` is cancelled, without sleeping on a pause.
///
/// Used inside the local optimisers, which must be able to unwind even while
/// the token is merely paused.
#[inline]
pub fn abort_requested(token: Option<&CancelToken>) -> bool {
    match token {
        Some(t) => t.is_cancelled(),
        None => false,
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::sync::atomic::AtomicUsize;
    use std::sync::mpsc;
    use std::thread;
    use std::time::Instant;

    #[test]
    fn a_fresh_token_is_running() {
        let t = CancelToken::new();
        assert_eq!(t.state(), CancelState::Running);
        assert!(t.is_running());
        assert!(!t.is_paused());
        assert!(!t.is_cancelled());
        assert!(!t.should_stop());
        assert!(!t.checkpoint());
    }

    #[test]
    fn clones_share_one_state_word() {
        let a = CancelToken::new();
        let b = a.clone();
        a.pause();
        assert!(b.is_paused());
        b.resume();
        assert!(a.is_running());
        b.cancel();
        assert!(a.is_cancelled());
    }

    #[test]
    fn pause_resume_round_trip() {
        let t = CancelToken::new();
        assert!(t.pause());
        assert!(!t.pause(), "pausing twice must be a no-op");
        assert!(t.resume());
        assert!(!t.resume(), "resuming a running token must be a no-op");
        assert!(t.is_running());
    }

    #[test]
    fn cancel_is_terminal() {
        let t = CancelToken::new();
        t.cancel();
        t.cancel(); // idempotent
        assert!(t.is_cancelled());
        assert!(!t.resume(), "resume must not resurrect a cancelled token");
        assert!(!t.pause(), "pause must not downgrade a cancelled token");
        assert_eq!(t.state(), CancelState::Cancelled);
        assert!(t.should_stop());
        assert!(t.checkpoint(), "a cancelled checkpoint returns immediately");
    }

    #[test]
    fn rearm_only_touches_a_cancelled_token() {
        let t = CancelToken::new();
        assert!(!t.rearm(), "a running token is not re-armed");
        t.pause();
        assert!(!t.rearm(), "a paused token is not re-armed");
        assert!(t.is_paused());
        t.cancel();
        assert!(t.rearm());
        assert!(t.is_running());
    }

    #[test]
    fn cancellation_wins_a_race_with_pause_and_resume() {
        // Hammer pause/resume from one thread while another cancels; whatever
        // the interleaving, the token must end up cancelled.
        for _ in 0..200 {
            let t = CancelToken::new();
            let other = t.clone();
            let h = thread::spawn(move || {
                for _ in 0..64 {
                    other.pause();
                    other.resume();
                }
            });
            t.cancel();
            h.join().unwrap();
            assert!(
                t.is_cancelled(),
                "a cancel lost the race against pause/resume"
            );
        }
    }

    #[test]
    fn paused_worker_does_not_advance_and_resumes() {
        let token = CancelToken::new();
        let counter = Arc::new(AtomicUsize::new(0));
        let (tx, rx) = mpsc::channel::<()>();

        let worker = {
            let token = token.clone();
            let counter = Arc::clone(&counter);
            thread::spawn(move || {
                loop {
                    if token.checkpoint() {
                        break;
                    }
                    counter.fetch_add(1, Ordering::Relaxed);
                    thread::sleep(Duration::from_millis(1));
                }
                let _ = tx.send(());
            })
        };

        // Let it run for a moment, then freeze it. One more iteration may slip
        // through before the worker reaches its next checkpoint, so allow a
        // single extra count rather than demanding exact equality (the worker
        // may also be descheduled by a loaded machine).
        thread::sleep(Duration::from_millis(20));
        assert!(token.pause());
        thread::sleep(Duration::from_millis(5));
        let before = counter.load(Ordering::Relaxed);
        thread::sleep(Duration::from_millis(25));
        let during = counter.load(Ordering::Relaxed);
        assert!(
            during <= before + 1,
            "a paused search advanced by {} iterations",
            during - before
        );

        // ... and it must carry on after a resume.
        assert!(token.resume());
        thread::sleep(Duration::from_millis(15));
        let after = counter.load(Ordering::Relaxed);
        assert!(
            after > during,
            "the resumed search did not advance ({during} -> {after})"
        );

        let started = Instant::now();
        token.cancel();
        rx.recv_timeout(Duration::from_secs(2))
            .expect("a cancelled worker must unwind promptly");
        assert!(started.elapsed() < Duration::from_millis(100));
        worker.join().unwrap();
    }

    #[test]
    fn cancel_latency_is_below_a_hundred_milliseconds() {
        let token = CancelToken::new();
        let worker = {
            let token = token.clone();
            thread::spawn(move || {
                let mut iterations = 0u64;
                while !token.checkpoint() {
                    iterations += 1;
                    // Simulate a chunk of search work.
                    thread::sleep(Duration::from_millis(1));
                    if iterations > 100_000 {
                        break;
                    }
                }
                iterations
            })
        };
        thread::sleep(Duration::from_millis(10));
        let started = Instant::now();
        token.cancel();
        worker.join().unwrap();
        let elapsed = started.elapsed();
        assert!(
            elapsed < Duration::from_millis(100),
            "cancellation took {elapsed:?}"
        );
    }

    #[test]
    fn token_is_send_and_sync() {
        fn assert_send_sync<T: Send + Sync>() {}
        assert_send_sync::<CancelToken>();
    }
}
